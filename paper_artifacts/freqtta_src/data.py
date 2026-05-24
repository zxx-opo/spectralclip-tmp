"""Dataset loading and patch sampling for HSI classification."""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from .config import DATASETS, TRANSFER_PAIRS, CFG


def load_hsi(name: str):
    meta = DATASETS[name]
    img = sio.loadmat(meta["img"])[meta["img_key"]].astype(np.float32)
    gt = sio.loadmat(meta["gt"])[meta["gt_key"]].astype(np.int32)
    return img, gt, meta["n_classes"]


def standardize_per_band(img):
    h, w, b = img.shape
    flat = img.reshape(-1, b)
    sc = StandardScaler()
    flat = sc.fit_transform(flat)
    return flat.reshape(h, w, b).astype(np.float32)


def align_bands(src_img, tgt_img, n_bands):
    return src_img[..., :n_bands], tgt_img[..., :n_bands]


def pad_image(img, patch_size):
    p = patch_size // 2
    return np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")


class HSIPatchDataset(Dataset):
    def __init__(self, img, gt, indices, patch_size):
        self.img_pad = pad_image(img, patch_size)
        self.gt = gt
        self.indices = indices
        self.patch_size = patch_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        p = self.patch_size
        patch = self.img_pad[r:r + p, c:c + p, :]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        label = int(self.gt[r, c]) - 1
        return patch, label, np.array([r, c], dtype=np.int64)


def split_indices(gt, train_ratio, val_ratio, rng):
    train_idx, val_idx, test_idx = [], [], []
    labels = np.unique(gt)
    labels = labels[labels > 0]
    for c in labels:
        rs, cs = np.where(gt == c)
        coords = np.stack([rs, cs], axis=1)
        n = len(coords)
        if n == 0:
            continue
        perm = rng.permutation(n)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio)))
        train_idx.append(coords[perm[:n_tr]])
        val_idx.append(coords[perm[n_tr:n_tr + n_va]])
        test_idx.append(coords[perm[n_tr + n_va:]])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def build_in_domain_loaders(name, seed):
    img, gt, n_cls = load_hsi(name)
    img = standardize_per_band(img)
    rng = np.random.default_rng(seed)
    tr, va, te = split_indices(gt, CFG.train_ratio, CFG.val_ratio, rng)
    tr_ds = HSIPatchDataset(img, gt, tr, CFG.patch_size)
    va_ds = HSIPatchDataset(img, gt, va, CFG.patch_size)
    te_ds = HSIPatchDataset(img, gt, te, CFG.patch_size)
    tr_ld = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=2)
    return tr_ld, va_ld, te_ld, img.shape[-1], n_cls


def build_transfer_loaders(pair_name, seed):
    pair = TRANSFER_PAIRS[pair_name]
    src_img, src_gt, n_cls = load_hsi(pair["source"])
    tgt_img, tgt_gt, _ = load_hsi(pair["target"])
    if pair["source"] != pair["target"]:
        src_img, tgt_img = align_bands(src_img, tgt_img, pair["shared_bands"])
    # random class-balanced split: keep same image, partition labeled pixels into src/tgt
    sp = pair.get("spatial_split", None)
    if sp is not None:
        # build per-class random mask: sp fraction -> source, (1-sp) -> target
        rng_sp = np.random.default_rng(seed + 7777)
        src_mask = np.zeros_like(src_gt, dtype=bool)
        tgt_mask = np.zeros_like(tgt_gt, dtype=bool)
        for c in np.unique(src_gt):
            if c == 0: continue
            rs, cs = np.where(src_gt == c)
            perm = rng_sp.permutation(len(rs))
            n_src = int(len(rs) * sp)
            src_mask[rs[perm[:n_src]], cs[perm[:n_src]]] = True
            tgt_mask[rs[perm[n_src:]], cs[perm[n_src:]]] = True
        # keep full image but mask gt to retain only assigned pixels (background = 0)
        src_gt_new = np.where(src_mask, src_gt, 0)
        tgt_gt_new = np.where(tgt_mask, tgt_gt, 0)
        src_gt = src_gt_new
        tgt_gt = tgt_gt_new
    src_img = standardize_per_band(src_img)
    tgt_img = standardize_per_band(tgt_img)
    # Inject synthetic noise to target if configured (creates a controlled cross-scene)
    tn = pair.get("target_noise", 0.0)
    if tn > 0:
        rng_n = np.random.default_rng(seed + 2000)
        noise = rng_n.normal(0, tn, size=tgt_img.shape).astype(np.float32)
        tgt_img = tgt_img + noise
    rng = np.random.default_rng(seed)
    s_tr, s_va, _ = split_indices(src_gt, CFG.train_ratio, CFG.val_ratio, rng)
    rng_t = np.random.default_rng(seed + 1000)
    t_idx = np.argwhere(tgt_gt > 0)
    t_idx = t_idx[rng_t.permutation(len(t_idx))]
    if hasattr(CFG, 'tta_max_target_samples') and CFG.tta_max_target_samples > 0:
        t_idx = t_idx[:CFG.tta_max_target_samples]
    s_tr_ds = HSIPatchDataset(src_img, src_gt, s_tr, CFG.patch_size)
    s_va_ds = HSIPatchDataset(src_img, src_gt, s_va, CFG.patch_size)
    t_te_ds = HSIPatchDataset(tgt_img, tgt_gt, t_idx, CFG.patch_size)
    s_tr_ld = DataLoader(s_tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    s_va_ld = DataLoader(s_va_ds, batch_size=256, shuffle=False, num_workers=2)
    t_te_ld = DataLoader(t_te_ds, batch_size=256, shuffle=False, num_workers=2)
    return s_tr_ld, s_va_ld, t_te_ld, src_img.shape[-1], n_cls