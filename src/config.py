"""Global configuration for SpectralCLIP experiments."""
from dataclasses import dataclass
from typing import Dict, Tuple
import os

ROOT = "/root/spectralclip-tmp"
DATA_DIR = os.path.join(ROOT, "data")
CORPUS_DIR = os.path.join(ROOT, "corpus")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")
for d in (CORPUS_DIR, CKPT_DIR, LOG_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

DATASETS: Dict[str, Dict] = {
    "PaviaU": dict(
        img=os.path.join(DATA_DIR, "PaviaU.mat"),
        gt=os.path.join(DATA_DIR, "PaviaU_gt.mat"),
        img_key="paviaU", gt_key="paviaU_gt",
        n_classes=9, bands=103,
        wl_start=430.0, wl_end=860.0,
    ),
    "PaviaC": dict(
        img=os.path.join(DATA_DIR, "Pavia.mat"),
        gt=os.path.join(DATA_DIR, "Pavia_gt.mat"),
        img_key="pavia", gt_key="pavia_gt",
        n_classes=9, bands=102,
        wl_start=430.0, wl_end=860.0,
    ),
    "IndianPines": dict(
        img=os.path.join(DATA_DIR, "Indian_pines_corrected.mat"),
        gt=os.path.join(DATA_DIR, "Indian_pines_gt.mat"),
        img_key="indian_pines_corrected", gt_key="indian_pines_gt",
        n_classes=16, bands=200,
        wl_start=400.0, wl_end=2500.0,
    ),
    "Salinas": dict(
        img=os.path.join(DATA_DIR, "Salinas_corrected.mat"),
        gt=os.path.join(DATA_DIR, "Salinas_gt.mat"),
        img_key="salinas_corrected",
        gt_key="salinas_gt",
        n_classes=16, bands=204,
        wl_start=400.0, wl_end=2500.0,
    ),
}


def wavelengths_for(name: str):
    """Return the per-band centre wavelengths (nm) for a dataset."""
    import numpy as np
    m = DATASETS[name]
    return np.linspace(m["wl_start"], m["wl_end"], m["bands"], dtype="float32")


@dataclass
class TrainCfg:
    # data
    patch_size: int = 1            # single-pixel spectra (no spatial context)
    train_ratio: float = 0.05
    val_ratio: float = 0.05
    seeds: Tuple[int, ...] = (42, 43, 44)
    device: str = "cuda:0"

    # SpecMamba encoder
    d_in: int = 48                 # per-band token dim (1 reflectance + 32 fourier wl + 15 neighbour)
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 4
    d_emb: int = 256

    # text tower
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    d_text: int = 768

    # training
    batch_size: int = 256
    lr: float = 2e-4
    wd: float = 1e-4
    pretrain_epochs: int = 100
    finetune_epochs: int = 60
    warmup_epochs: int = 5
    temperature: float = 0.07
    lambda_phys: float = 0.20
    phys_margin: float = 0.10

    # paraphrase augmentation
    n_paraphrases: int = 1         # set to 1 when not generating LLM paraphrases
    n_hard_negatives: int = 0      # disable for first version

    # few-shot inference
    proto_alpha: float = 0.7       # mixing weight for textual-visual prototype


CFG = TrainCfg()