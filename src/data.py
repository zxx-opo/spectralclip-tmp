"""Dataset loading for SpectralCLIP.

Provides two kinds of datasets:
  - SpectrumCaptionDataset: paired (spectrum, caption_text_idx) for contrastive
    pretraining. Spectra come from sampled labelled pixels of all 4 benchmarks.
  - SinglePixelDataset: spectrum-only loader used for closed-set fine-tuning
    and zero-shot evaluation.
"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

from .config import DATASETS, CFG, wavelengths_for
from .captions import CAPTIONS, get_caption


def load_hsi(name: str):
    meta = DATASETS[name]
    img = sio.loadmat(meta["img"])[meta["img_key"]].astype(np.float32)
    gt = sio.loadmat(meta["gt"])[meta["gt_key"]].astype(np.int32)
    return img, gt, meta["n_classes"]


def standardize_per_band(img: np.ndarray) -> np.ndarray:
    h, w, b = img.shape
    flat = img.reshape(-1, b)
    sc = StandardScaler()
    flat = sc.fit_transform(flat)
    return flat.reshape(h, w, b).astype(np.float32)


def split_indices(gt: np.ndarray, train_ratio: float, val_ratio: float,
                  rng: np.random.Generator):
    """Stratified split returning (train_idx, val_idx, test_idx) of shape (N,2)."""
    tr, va, te = [], [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        rs, cs = np.where(gt == c)
        coords = np.stack([rs, cs], axis=1)
        n = len(coords)
        if n == 0:
            continue
        perm = rng.permutation(n)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio)))
        tr.append(coords[perm[:n_tr]])
        va.append(coords[perm[n_tr:n_tr + n_va]])
        te.append(coords[perm[n_tr + n_va:]])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


class SinglePixelDataset(Dataset):
    """A spectrum-only dataset: each item is (spectrum, dataset_id, class_id)."""

    def __init__(self, img: np.ndarray, gt: np.ndarray, indices: np.ndarray,
                 dataset_id: int):
        self.img = img
        self.gt = gt
        self.indices = indices
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        spec = torch.from_numpy(self.img[r, c, :]).float()    # (B,)
        label = int(self.gt[r, c]) - 1                         # 0..C-1
        return spec, self.dataset_id, label


class SpectrumCaptionDataset(Dataset):
    """Paired dataset for contrastive pretraining.

    Each item: (spectrum, caption_idx) where caption_idx points into the
    flat caption table built across all (dataset, class) pairs.
    """

    def __init__(self, dataset_names: List[str], per_class: int = 200,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.items = []            # (spec_array, caption_idx, dataset_id, class_id)
        self.captions: List[str] = []
        self.cap_keys: List[Tuple[str, int]] = []   # (dataset, class_id)
        self.cube_cache = {}

        cap_index = {}
        for ds_idx, name in enumerate(dataset_names):
            img, gt, n_cls = load_hsi(name)
            img = standardize_per_band(img)
            self.cube_cache[name] = (img, gt)
            for c in range(1, n_cls + 1):
                if c not in CAPTIONS[name]:
                    continue
                cap = get_caption(name, c)
                if (name, c) not in cap_index:
                    cap_index[(name, c)] = len(self.captions)
                    self.captions.append(cap)
                    self.cap_keys.append((name, c))
                cap_idx = cap_index[(name, c)]
                rs, cs = np.where(gt == c)
                if len(rs) == 0:
                    continue
                perm = rng.permutation(len(rs))[:per_class]
                for k in perm:
                    spec = img[rs[k], cs[k]].astype(np.float32)
                    self.items.append((spec, cap_idx, ds_idx, c - 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        spec, cap_idx, ds_id, cls = self.items[i]
        return torch.from_numpy(spec).float(), int(cap_idx), int(ds_id), int(cls)


def collate_variable_length(batch):
    """Collate spectra of (potentially) different band counts by padding to max.

    Returns:
        specs:  (B, B_max) float
        mask:   (B, B_max) bool (True = valid)
        cap_idx, ds_id, cls: (B,) long
    """
    B = len(batch)
    Bmax = max(x[0].shape[0] for x in batch)
    specs = torch.zeros(B, Bmax, dtype=torch.float32)
    mask = torch.zeros(B, Bmax, dtype=torch.bool)
    cap_idx = torch.zeros(B, dtype=torch.long)
    ds_id = torch.zeros(B, dtype=torch.long)
    cls = torch.zeros(B, dtype=torch.long)
    for i, (s, ci, di, c) in enumerate(batch):
        n = s.shape[0]
        specs[i, :n] = s
        mask[i, :n] = True
        cap_idx[i] = ci
        ds_id[i] = di
        cls[i] = c
    return specs, mask, cap_idx, ds_id, cls


def build_pretrain_loader(per_class: int = 200, seed: int = 42, batch_size: int = None):
    bs = batch_size or CFG.batch_size
    ds = SpectrumCaptionDataset(
        dataset_names=list(DATASETS.keys()), per_class=per_class, seed=seed,
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2,
                        collate_fn=collate_variable_length, drop_last=True)
    return ds, loader


def build_indomain_loaders(name: str, seed: int):
    img, gt, n_cls = load_hsi(name)
    img = standardize_per_band(img)
    rng = np.random.default_rng(seed)
    tr, va, te = split_indices(gt, CFG.train_ratio, CFG.val_ratio, rng)
    ds_id = list(DATASETS.keys()).index(name)
    tr_ds = SinglePixelDataset(img, gt, tr, ds_id)
    va_ds = SinglePixelDataset(img, gt, va, ds_id)
    te_ds = SinglePixelDataset(img, gt, te, ds_id)

    def _col(batch):
        specs = torch.stack([b[0] for b in batch])
        ds_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return specs, ds_ids, labels

    tr_ld = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True,
                       num_workers=2, collate_fn=_col)
    va_ld = DataLoader(va_ds, batch_size=512, shuffle=False,
                       num_workers=2, collate_fn=_col)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False,
                       num_workers=2, collate_fn=_col)
    return tr_ld, va_ld, te_ld, img.shape[-1], n_cls


def build_fewshot_loaders(name: str, k_per_class: int, seed: int):
    """Build (support, query) loaders for few-shot. Support has k_per_class
    pixels per class; query has the rest."""
    img, gt, n_cls = load_hsi(name)
    img = standardize_per_band(img)
    rng = np.random.default_rng(seed)
    support, query = [], []
    for c in range(1, n_cls + 1):
        rs, cs = np.where(gt == c)
        if len(rs) == 0:
            continue
        coords = np.stack([rs, cs], axis=1)
        perm = rng.permutation(len(coords))
        n_sup = min(k_per_class, len(coords))
        support.append(coords[perm[:n_sup]])
        query.append(coords[perm[n_sup:]])
    support = np.concatenate(support) if support else np.zeros((0, 2), int)
    query = np.concatenate(query) if query else np.zeros((0, 2), int)
    ds_id = list(DATASETS.keys()).index(name)
    sup_ds = SinglePixelDataset(img, gt, support, ds_id)
    qry_ds = SinglePixelDataset(img, gt, query, ds_id)

    def _col(batch):
        specs = torch.stack([b[0] for b in batch])
        ds_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return specs, ds_ids, labels

    sup_ld = DataLoader(sup_ds, batch_size=512, shuffle=False, num_workers=2, collate_fn=_col)
    qry_ld = DataLoader(qry_ds, batch_size=512, shuffle=False, num_workers=2, collate_fn=_col)
    return sup_ld, qry_ld, img.shape[-1], n_cls