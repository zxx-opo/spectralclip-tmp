"""Spectral-Spatial MAE pretraining for FAM-SSM."""
import os, json, time, sys
sys.path.insert(0, "/root/freqtta-hsi")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from src.config import CFG, DATASETS, CKPT_DIR, LOG_DIR
from src.data import load_hsi, standardize_per_band, pad_image, HSIPatchDataset
from src.model import FAMSSMHSI


def build_pretrain_dataset(patch_size=11, max_per_ds=20000):
    """Aggregate patches from all datasets. Each dataset capped to max_per_ds samples."""
    all_ds = []
    for name in ["PaviaU", "PaviaC", "IndianPines", "Salinas"]:
        img, gt, _ = load_hsi(name)
        img = standardize_per_band(img)
        # use ALL pixels (including background) for self-supervised pretrain
        H, W = img.shape[:2]
        coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
        rng = np.random.default_rng(0)
        coords = coords[rng.permutation(len(coords))]
        coords = coords[:max_per_ds]
        ds = HSIPatchDataset(img, gt, coords, patch_size)
        print(f"  pretrain[{name}]: {len(ds)} patches, B={img.shape[-1]}")
        all_ds.append((name, ds, img.shape[-1]))
    return all_ds


def pretrain_one(ds_name, dataset, n_bands, n_classes_dummy, epochs=30, lr=5e-4, mask_ratio=0.30, log_prefix="pretrain"):
    """Pretrain FAMSSMHSI on one dataset (different #bands)."""
    device = CFG.device
    model = FAMSSMHSI(n_bands=n_bands, n_classes=n_classes_dummy,
                      d_model=CFG.d_model, d_state=CFG.d_state,
                      d_phi=CFG.d_phi, n_layers=CFG.n_layers, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    log_path = os.path.join(LOG_DIR, f"{log_prefix}_{ds_name}.jsonl")
    flog = open(log_path, "w")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        sum_loss, n = 0.0, 0
        for patch, _, _ in loader:
            patch = patch.to(device, non_blocking=True)
            B = patch.shape[1]
            keep = (torch.rand(patch.size(0), B, 1, 1, device=device) >= mask_ratio).float()
            patch_in = patch * keep
            rec = model.reconstruct_spectrum(patch_in)
            H, W = patch.shape[-2], patch.shape[-1]
            center = patch[:, :, H//2, W//2]
            mask_b = (keep.squeeze(-1).squeeze(-1) == 0).float()
            loss = ((rec - center)**2 * mask_b).sum() / (mask_b.sum() + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sum_loss += float(loss.item()); n += 1
        sched.step()
        rec = dict(epoch=ep+1, loss=sum_loss/max(1,n), t=time.time()-t0, lr=opt.param_groups[0]["lr"])
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if (ep+1) % 5 == 0:
            print(f"  [{ds_name}] ep{ep+1}/{epochs} loss={sum_loss/max(1,n):.4f} t={time.time()-t0:.1f}s")
    flog.close()
    # save encoder weights (excluding classifier which is dummy)
    sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    ckpt = os.path.join(CKPT_DIR, f"mae_{ds_name}.pt")
    torch.save(sd, ckpt)
    print(f"  saved -> {ckpt}")
    return ckpt


def main():
    print("=== MAE PRETRAINING ===")
    all_ds = build_pretrain_dataset(patch_size=CFG.patch_size, max_per_ds=20000)
    for ds_name, ds, n_bands in all_ds:
        n_cls = DATASETS[ds_name]["n_classes"]
        ckpt = pretrain_one(ds_name, ds, n_bands, n_cls, epochs=25, lr=5e-4, mask_ratio=0.30)
        print(f"finished {ds_name} -> {ckpt}")


if __name__ == "__main__":
    main()
