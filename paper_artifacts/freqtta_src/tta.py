"""Test-time adaptation: source_only / tent / eata / cotta_lite / smtta / phitta."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from .config import CFG
from .trainer import spectral_band_mask


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    p, y = [], []
    for patch, lab, _ in loader:
        patch = patch.to(device)
        out = model(patch)
        p.append(out.argmax(-1).cpu().numpy()); y.append(lab.numpy())
    return np.concatenate(p), np.concatenate(y)


def metrics_from(pred, lab):
    oa = float((pred == lab).mean())
    per = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() > 0:
            per.append(float((pred[m] == c).mean()))
    aa = float(np.mean(per)) if per else 0.0
    kappa = float(cohen_kappa_score(lab, pred))
    return dict(OA=oa, AA=aa, kappa=kappa, per_class=per)


def _entropy(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def _gather_params(model, method: str):
    """Return list of nn.Parameters to be adapted under each method."""
    if method == "phitta":
        params = list
def _gather_params(model, method: str):
    """Return list of nn.Parameters to be adapted under each method."""
    if method == "phitta":
        params = list(model.phi_params())
        if CFG.phitta_include_film:
            params += list(model.film_params())
        if not params:
            # baseline that has no phi_params -> fall back to LN affines
            params = list(model.all_ln_params())
        return params
    if method in ("smtta", "tent", "eata", "cotta_lite"):
        return list(model.all_ln_params())
    raise ValueError(method)


def _configure_for_tta(model, method):
    for p in model.parameters():
        p.requires_grad_(False)
    params = _gather_params(model, method)
    for p in params:
        p.requires_grad_(True)
    return params


def adapt_and_eval(model, loader, device, method="phitta",
                   steps=10, lr=1e-4, alpha=1.0, tau=0.7, mask_ratio=0.3):
    state_backup = copy.deepcopy(model.state_dict())
    if method == "source_only":
        pred, lab = collect_preds(model, loader, device)
        return metrics_from(pred, lab)

    params = _configure_for_tta(model, method)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.train()
    all_pred, all_lab = [], []

    for patch, lab, _ in loader:
        patch = patch.to(device)
        lab_np = lab.numpy()
        for s in range(steps):
            if method in ("smtta", "phitta"):
                keep = spectral_band_mask(patch, mask_ratio)
                patch_in = patch * keep
                logits = model(patch_in)
                ent = _entropy(logits)
                with torch.no_grad():
                    prob = F.softmax(logits, dim=-1)
                    conf = prob.max(-1).values
                    keep_mask = (conf > tau).float()
                loss_e = (ent * keep_mask).sum() / (keep_mask.sum() + 1e-6)
                rec = model.reconstruct_spectrum(patch_in)
                Bsz, B, H, W = patch.shape
                center = patch[:, :, H // 2, W // 2]
                mask_b = (keep.squeeze(-1).squeeze(-1) == 0).float()
                loss_r = ((rec - center) ** 2 * mask_b).sum() / (mask_b.sum() + 1e-6)
                loss = loss_r + alpha * loss_e
            elif method == "tent":
                logits = model(patch)
                loss = _entropy(logits).mean()
            elif method == "eata":
                logits = model(patch)
                ent = _entropy(logits)
                with torch.no_grad():
                    conf = F.softmax(logits, dim=-1).max(-1).values
                    km = (conf > tau).float()
                loss = (ent * km).sum() / (km.sum() + 1e-6)
            elif method == "cotta_lite":
                logits = model(patch)
                logits_aug = model(patch + 0.01 * torch.randn_like(patch))
                p_ = F.softmax(logits, dim=-1)
                p_aug = F.softmax(logits_aug, dim=-1)
                avg = 0.5 * (p_ + p_aug)
                loss = -(avg * torch.log(avg + 1e-8)).sum(-1).mean()
            else:
                raise ValueError(method)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        with torch.no_grad():
            model.eval()
            out = model(patch)
            all_pred.append(out.argmax(-1).cpu().numpy())
            all_lab.append(lab_np)
            model.train()
    pred = np.concatenate(all_pred)
    lab = np.concatenate(all_lab)
    m = metrics_from(pred, lab)
    model.load_state_dict(state_backup)
    return m