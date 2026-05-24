"""Contrastive pretraining + supervised finetuning loops for SpectralCLIP."""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from typing import List

from .config import CFG, CKPT_DIR, LOG_DIR, DATASETS, wavelengths_for


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wavelengths_for_batch(ds_ids: torch.Tensor, Bmax: int, device) -> torch.Tensor:
    """Build a (B, Bmax) wavelength tensor given per-sample dataset_id."""
    names = list(DATASETS.keys())
    out = torch.zeros(len(ds_ids), Bmax, dtype=torch.float32)
    for i, di in enumerate(ds_ids.tolist()):
        wl = wavelengths_for(names[di])
        n = min(len(wl), Bmax)
        out[i, :n] = torch.from_numpy(wl[:n])
    return out.to(device)


def info_nce_loss(z_spec: torch.Tensor, z_text: torch.Tensor,
                  cap_idx: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. z_text indexed by cap_idx gives the positives.

    z_spec: (B, d)   normalized
    z_text: (M, d)   normalized (all caption embeddings)
    cap_idx: (B,)    long; cap_idx[i] is the positive caption for spec_i
    """
    scale = logit_scale.exp().clamp_max(100.0)
    logits = scale * z_spec @ z_text.t()       # (B, M)
    # InfoNCE: target = cap_idx[i] for each row
    loss_s2t = F.cross_entropy(logits, cap_idx)
    # text->spec: for each unique caption, the positives are its assigned spectra.
    # We use a simplified "best-matching" loss: take z_text indexed by cap_idx
    # as the "text view" and contrast back.
    text_view = z_text[cap_idx]                  # (B, d) corresponding text emb
    logits2 = scale * text_view @ z_spec.t()     # (B, B)
    target2 = torch.arange(z_spec.size(0), device=z_spec.device)
    loss_t2s = F.cross_entropy(logits2, target2)
    return 0.5 * (loss_s2t + loss_t2s)


def physical_consistency_loss(z_spec: torch.Tensor, spec: torch.Tensor,
                              mask: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    """Penalize the encoder for relying on intensity similarity rather than shape."""
    Bsz = z_spec.size(0)
    s = spec.float()
    m = mask.float()
    # intensity similarity = cosine of raw spectra (masked)
    s_norm = s / (s.norm(dim=-1, keepdim=True) + 1e-6)
    sigma_in = s_norm @ s_norm.t()                   # (B,B)
    # shape similarity = cosine of first derivative
    ds = s[:, 1:] - s[:, :-1]
    ds_mask = m[:, 1:] * m[:, :-1]
    ds = ds * ds_mask
    ds_norm = ds / (ds.norm(dim=-1, keepdim=True) + 1e-6)
    sigma_sh = ds_norm @ ds_norm.t()                # (B,B)
    # embedding similarity
    z_sim = z_spec @ z_spec.t()
    # ReLU activates only on shortcut pairs (intensity >> shape)
    shortcut = F.relu(sigma_in - sigma_sh - margin)
    # squared deviation from shape similarity, weighted by shortcut activation
    loss = (shortcut * (z_sim - sigma_sh) ** 2).mean()
    return loss


def pretrain_contrastive(model, loader, text_embeddings, epochs, lr, wd,
                         lambda_phys, tag="pretrain", device="cuda:0"):
    """Contrastive pretraining loop."""
    model.to(device)
    text_embeddings = text_embeddings.to(device)
    # Project + normalize text embeddings ONCE per epoch (since text_proj is trainable)
    opt = torch.optim.AdamW(
        list(model.spec_enc.parameters()) + list(model.text_proj.parameters())
        + [model.logit_scale], lr=lr, weight_decay=wd,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log_path = os.path.join(LOG_DIR, f"{tag}.jsonl")
    flog = open(log_path, "w")
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        sum_loss, sum_n = 0.0, 0
        sum_nce, sum_phys = 0.0, 0.0
        for spec, mask, cap_idx, ds_id, _ in loader:
            spec = spec.to(device); mask = mask.to(device); cap_idx = cap_idx.to(device)
            Bmax = spec.size(1)
            wls = _wavelengths_for_batch(ds_id, Bmax, device)
            z_spec = model.encode_spectrum(spec, mask, wls)        # (B, d)
            z_text = model.encode_text(text_embeddings)            # (M, d)
            loss_nce = info_nce_loss(z_spec, z_text, cap_idx, model.logit_scale)
            if lambda_phys > 0:
                loss_phys = physical_consistency_loss(z_spec, spec, mask,
                                                      margin=CFG.phys_margin)
            else:
                loss_phys = torch.tensor(0.0, device=device)
            loss = loss_nce + lambda_phys * loss_phys
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_loss += float(loss.item()); sum_n += 1
            sum_nce += float(loss_nce.item()); sum_phys += float(loss_phys.item())
        sched.step()
        rec = dict(
            epoch=ep + 1,
            loss=sum_loss / max(1, sum_n),
            loss_nce=sum_nce / max(1, sum_n),
            loss_phys=sum_phys / max(1, sum_n),
            logit_scale=float(model.logit_scale.exp().item()),
            t=time.time() - t0,
        )
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  [{tag}] ep{ep+1}/{epochs} loss={rec['loss']:.4f} "
                  f"nce={rec['loss_nce']:.4f} phys={rec['loss_phys']:.4f} "
                  f"scale={rec['logit_scale']:.2f} t={rec['t']:.1f}s")
    flog.close()
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@torch.no_grad()
def evaluate_supervised(model, classifier_head, loader, device):
    """Evaluate a trained linear/prototype classifier head on a loader."""
    model.eval(); classifier_head.eval()
    names = list(DATASETS.keys())
    preds, labs = [], []
    for spec, ds_id, lab in loader:
        spec = spec.to(device)
        mask = torch.ones_like(spec, dtype=torch.bool)
        wls = _wavelengths_for_batch(ds_id, spec.size(1), device)
        z = model.encode_spectrum(spec, mask, wls)
        logits = classifier_head(z)
        preds.append(logits.argmax(-1).cpu().numpy())
        labs.append(lab.numpy())
    p = np.concatenate(preds); y = np.concatenate(labs)
    oa = float((p == y).mean())
    per = []
    for c in np.unique(y):
        m = y == c
        if m.sum() > 0:
            per.append(float((p[m] == c).mean()))
    aa = float(np.mean(per)) if per else 0.0
    k = float(cohen_kappa_score(y, p))
    return dict(OA=oa, AA=aa, kappa=k, per_class=per)


def finetune_supervised(model, tr_loader, va_loader, n_classes, epochs,
                        lr, wd, tag="finetune", device="cuda:0", patience=20):
    """Closed-set supervised finetuning: train a linear classifier on top of
    a (possibly frozen) SpecMamba encoder."""
    model.to(device)
    head = nn.Linear(CFG.d_emb, n_classes).to(device)
    params = list(model.spec_enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log_path = os.path.join(LOG_DIR, f"{tag}.jsonl")
    flog = open(log_path, "w")
    t0 = time.time()
    best_va_oa = -1.0
    best_state = None
    bad = 0

    for ep in range(epochs):
        model.train(); head.train()
        sum_loss, n_correct, n_total = 0.0, 0, 0
        for spec, ds_id, lab in tr_loader:
            spec = spec.to(device); lab = lab.to(device)
            mask = torch.ones_like(spec, dtype=torch.bool)
            wls = _wavelengths_for_batch(ds_id, spec.size(1), device)
            z = model.encode_spectrum(spec, mask, wls)
            logits = head(z)
            loss = nn.functional.cross_entropy(logits, lab)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sum_loss += float(loss.item())
            n_correct += int((logits.argmax(-1) == lab).sum().item())
            n_total += lab.size(0)
        sched.step()
        tr_oa = n_correct / max(1, n_total)
        va_m = evaluate_supervised(model, head, va_loader, device)
        rec = dict(epoch=ep + 1, tr_loss=sum_loss / max(1, len(tr_loader)),
                   tr_oa=tr_oa, va_oa=va_m["OA"], va_aa=va_m["AA"],
                   va_kappa=va_m["kappa"], lr=opt.param_groups[0]["lr"],
                   t=time.time() - t0)
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if va_m["OA"] > best_va_oa:
            best_va_oa = va_m["OA"]
            best_state = {
                "spec": {k: v.detach().cpu().clone() for k, v in model.spec_enc.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            flog.write(json.dumps(dict(early_stop=True, epoch=ep + 1)) + "\n")
            break
    flog.close()
    model.spec_enc.load_state_dict(best_state["spec"])
    head.load_state_dict(best_state["head"])
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pt")
    torch.save({"spec": best_state["spec"], "head": best_state["head"]}, ckpt_path)
    return head, ckpt_path, dict(best_va_oa=best_va_oa, epochs_run=ep + 1)