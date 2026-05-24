"""Source-domain trainer with joint classification + spectral mask reconstruction."""
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from typing import Tuple
from .config import CFG, CKPT_DIR, LOG_DIR


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spectral_band_mask(x, ratio):
    Bsz, B, H, W = x.shape
    keep = torch.rand(Bsz, B, 1, 1, device=x.device) >= ratio
    return keep.float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_lab = [], []
    for patch, lab, _ in loader:
        patch = patch.to(device, non_blocking=True)
        logits = model(patch)
        all_pred.append(logits.argmax(-1).cpu().numpy())
        all_lab.append(lab.numpy())
    pred = np.concatenate(all_pred); lab = np.concatenate(all_lab)
    oa = float((pred == lab).mean())
    per = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() > 0:
            per.append(float((pred[m] == c).mean()))
    aa = float(np.mean(per)) if per else 0.0
    kappa = float(cohen_kappa_score(lab, pred))
    return dict(OA=oa, AA=aa, kappa=kappa, per_class=per)


def train_source(model, train_loader, val_loader, device, epochs, lr, wd,
                 use_recon=True, mask_ratio=0.30, rec_w=0.10,
                 log_prefix="exp", patience=25) -> Tuple[dict, str]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_va_oa = -1.0
    best_state = None
    bad = 0
    log_path = os.path.join(LOG_DIR, f"{log_prefix}.jsonl")
    f_log = open(log_path, "w")
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        sum_cls, sum_rec, n_batch, n_correct, n_total = 0.0, 0.0, 0, 0, 0
        for patch, lab, _ in train_loader:
            patch = patch.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            if use_recon and hasattr(model, "reconstruct_spectrum"):
                keep = spectral_band_mask(patch, mask_ratio)
                patch_in = patch * keep
                logits = model(patch_in)
                rec_full = model.reconstruct_spectrum(patch_in)
                Bsz, B, H, W = patch.shape
                center = patch[:, :, H // 2, W // 2]
                mask_b = (keep.squeeze(-1).squeeze(-1) == 0).float()
                rec_loss = ((rec_full - center) ** 2 * mask_b).sum() / (mask_b.sum() + 1e-6)
            else:
                logits = model(patch)
                rec_loss = torch.tensor(0.0, device=device)
            cls_loss = F.cross_entropy(logits, lab)
            loss = cls_loss + rec_w * rec_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sum_cls += float(cls_loss.item()); sum_rec += float(rec_loss.item())
            n_batch += 1
            n_correct += int((logits.argmax(-1) == lab).sum().item())
            n_total += lab.size(0)
        sched.step()
        tr_oa = n_correct / max(1, n_total)
        va_m = evaluate(model, val_loader, device)
        rec = dict(epoch=ep + 1,
                   tr_loss_cls=sum_cls / max(1, n_batch),
                   tr_loss_rec=sum_rec / max(1, n_batch),
                   tr_oa=tr_oa,
                   va_oa=va_m["OA"], va_aa=va_m["AA"], va_kappa=va_m["kappa"],
                   lr=opt.param_groups[0]["lr"], t=time.time() - t0)
        f_log.write(json.dumps(rec) + "\n"); f_log.flush()
        if va_m["OA"] > best_va_oa:
            best_va_oa = va_m["OA"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            f_log.write(json.dumps(dict(early_stop=True, epoch=ep + 1)) + "\n"); f_log.flush()
            break
    f_log.close()
    ckpt_path = os.path.join(CKPT_DIR, f"{log_prefix}.pt")
    torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    return dict(best_va_oa=best_va_oa, log=log_path, epochs_run=ep + 1), ckpt_path