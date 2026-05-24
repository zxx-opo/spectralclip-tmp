"""Global configuration for FAM-SSM experiments."""
from dataclasses import dataclass
from typing import Dict, Tuple
import os

ROOT = "/root/freqtta-hsi"
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")
for d in (CKPT_DIR, LOG_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

DATASETS: Dict[str, Dict] = {
    "PaviaU": dict(
        img=os.path.join(DATA_DIR, "PaviaU.mat"),
        gt=os.path.join(DATA_DIR, "PaviaU_gt.mat"),
        img_key="paviaU", gt_key="paviaU_gt",
        n_classes=9, bands=103),
    "PaviaC": dict(
        img=os.path.join(DATA_DIR, "Pavia.mat"),
        gt=os.path.join(DATA_DIR, "Pavia_gt.mat"),
        img_key="pavia", gt_key="pavia_gt",
        n_classes=9, bands=102),
    "IndianPines": dict(
        img=os.path.join(DATA_DIR, "Indian_pines_corrected.mat"),
        gt=os.path.join(DATA_DIR, "Indian_pines_gt.mat"),
        img_key="indian_pines_corrected", gt_key="indian_pines_gt",
        n_classes=16, bands=200),
    "Salinas": dict(
        img=os.path.join(DATA_DIR, "Salinas_corrected.mat"),
        gt=os.path.join(DATA_DIR, "Salinas_gt.mat"),
        img_key="salinas_corrected", gt_key="salinas_gt",
        n_classes=16, bands=204),
}

TRANSFER_PAIRS: Dict[str, Dict] = {
    "S1_PaviaC2U":  dict(source="PaviaC", target="PaviaU",  shared_bands=102, n_classes=9),
    "S2_PaviaC_noise": dict(source="PaviaC", target="PaviaC", shared_bands=102, n_classes=9, target_noise=3.0),
    "P1_PaviaU_split":      dict(source="PaviaU",      target="PaviaU",      shared_bands=103, n_classes=9,  spatial_split=0.5),
    "P2_PaviaC_split":      dict(source="PaviaC",      target="PaviaC",      shared_bands=102, n_classes=9,  spatial_split=0.5),
    "P3_IndianPines_split": dict(source="IndianPines", target="IndianPines", shared_bands=200, n_classes=16, spatial_split=0.5),
    "P4_Salinas_split":     dict(source="Salinas",     target="Salinas",     shared_bands=204, n_classes=16, spatial_split=0.5),
    "D1_PaviaU_noise":      dict(source="PaviaU",      target="PaviaU",      shared_bands=103, n_classes=9,  spatial_split=0.5, target_noise=0.3),
    "D2_Salinas_noise":     dict(source="Salinas",     target="Salinas",     shared_bands=204, n_classes=16, spatial_split=0.5, target_noise=0.3),
    "S5_IPself":    dict(source="IndianPines", target="IndianPines", shared_bands=200, n_classes=16),
}


@dataclass
class TrainCfg:
    # data / training
    patch_size: int = 11
    train_ratio: float = 0.05
    val_ratio: float = 0.05
    batch_size: int = 64
    lr: float = 5e-4
    wd: float = 1e-4
    epochs: int = 80
    warmup_epochs: int = 5
    seeds: Tuple[int, ...] = (42, 43, 44)
    device: str = "cuda:0"
    # FAM-SSM architecture
    d_model: int = 64
    d_state: int = 16
    d_phi: int = 64
    n_radial_bins: int = 16
    n_layers: int = 2
    dropout: float = 0.2
    # spectral mask reconstruction
    mask_ratio: float = 0.30
    rec_weight: float = 0.10
    # TTA
    tta_steps: int = 10
    tta_lr: float = 1e-4
    tta_alpha: float = 1.0
    tta_tau: float = 0.7
    tta_max_target_samples: int = 5000   # subsample target for fast TTA eval
    # phitta-specific
    phitta_include_film: bool = True   # also adapt FiLM projection matrices


CFG = TrainCfg()