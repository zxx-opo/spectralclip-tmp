"""Zero-shot and few-shot evaluation for SpectralCLIP."""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from .config import CFG, DATASETS, RESULTS_DIR, wavelengths_for
from .captions import all_prompts
from .text_encoder import encode_captions
from .data import build_indomain_loaders, build_fewshot_loaders


def _wavelengths(name: str, Bmax: int, device):
    wl = wavelengths_for(name)
    out = torch.zeros(1, Bmax, dtype=torch.float32)
    n = min(len(wl), Bmax)
    out[0, :n] = torch.from_numpy(wl[:n])
    return out.to(device)


@torch.no_grad()
def encode_loader(model, loader, device):
    """Return (N, d_emb) embeddings and (N,) labels for a loader."""
    model.eval()
    embs, labs, ds_ids = [], [], []
    for spec, ds_id, lab in loader:
        spec = spec.to(device); B = spec.size(0); Bmax = spec.size(1)
        mask = torch.ones_like(spec, dtype=torch.bool)
        # all samples in this loader come from the same dataset
        name = list(DATASETS.keys())[int(ds_id[0])]
        wls = _wavelengths(name, Bmax, device).expand(B, -1)
        z = model.encode_spectrum(spec, mask, wls)
        embs.append(z.cpu()); labs.append(lab); ds_ids.append(ds_id)
    return torch.cat(embs), torch.cat(labs), torch.cat(ds_ids)


def _metrics(pred, lab):
    pred = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    lab = lab.numpy() if isinstance(lab, torch.Tensor) else lab
    oa = float((pred == lab).mean())
    per = []
    for c in np.unique(lab):
        m = lab == c
        if m.sum() > 0:
            per.append(float((pred[m] == c).mean()))
    aa = float(np.mean(per)) if per else 0.0
    k = float(cohen_kappa_score(lab, pred))
    return dict(OA=oa, AA=aa, kappa=k)


def zero_shot_eval(model, dataset: str, seed: int = 42, device: str = "cuda:0"):
    """Encode all test pixels of `dataset`, compare to text prompts of every
    class, predict by argmax cosine similarity."""
    # text prompts
    ids, names, caps = all_prompts(dataset)
    text_emb = encode_captions(caps, device="cpu").to(device)         # (C, d_text)
    with torch.no_grad():
        z_text = model.encode_text(text_emb)                          # (C, d_emb)

    # spectra test loader (full label set, no training)
    _, _, te_loader, _, n_cls = build_indomain_loaders(dataset, seed)
    z_spec, y, _ = encode_loader(model, te_loader, device)
    z_spec = z_spec.to(device)

    # argmax cosine similarity
    sim = z_spec @ z_text.t()                                         # (N, C)
    pred = sim.argmax(-1).cpu()
    m = _metrics(pred, y)
    m["dataset"] = dataset
    m["seed"] = seed
    m["mode"] = "zero_shot"
    m["n_test"] = int(y.numel())
    return m


def few_shot_eval(model, dataset: str, k_per_class: int, seed: int = 42,
                  alpha: float = None, device: str = "cuda:0"):
    """Few-shot inference with textual-visual prototype:
        p_c = alpha * mean(z_spec | y==c) + (1-alpha) * z_text_c
    """
    alpha = CFG.proto_alpha if alpha is None else alpha
    # text prompts
    ids, names, caps = all_prompts(dataset)
    text_emb = encode_captions(caps, device="cpu").to(device)
    with torch.no_grad():
        z_text = model.encode_text(text_emb)                          # (C, d_emb)

    sup_loader, qry_loader, _, n_cls = build_fewshot_loaders(dataset, k_per_class, seed)
    z_sup, y_sup, _ = encode_loader(model, sup_loader, device)
    z_qry, y_qry, _ = encode_loader(model, qry_loader, device)
    z_sup = z_sup.to(device); z_qry = z_qry.to(device)

    # build per-class centroids from support
    centroids = torch.zeros(n_cls, z_sup.size(-1), device=device)
    counts = torch.zeros(n_cls, device=device)
    for c in range(n_cls):
        m = (y_sup == c)
        if m.any():
            centroids[c] = z_sup[m].mean(0)
            counts[c] = m.sum().float()
    # for classes with zero support, fall back to text embedding only
    counts = counts.unsqueeze(-1)
    # mix textual-visual prototype where support exists; else pure text
    proto = torch.where(counts > 0,
                        alpha * F.normalize(centroids, dim=-1) +
                        (1 - alpha) * z_text,
                        z_text)
    proto = F.normalize(proto, dim=-1)
    sim = z_qry @ proto.t()
    pred = sim.argmax(-1).cpu()
    m = _metrics(pred, y_qry)
    m["dataset"] = dataset
    m["seed"] = seed
    m["mode"] = "few_shot"
    m["k_per_class"] = k_per_class
    m["n_query"] = int(y_qry.numel())
    return m


def cross_sensor_zero_shot(model, src_dataset: str, tgt_dataset: str,
                            seed: int = 42, device: str = "cuda:0"):
    """Use src-finetuned model and tgt prompt library to zero-shot predict
    on tgt benchmark. The function just calls zero_shot_eval on tgt;
    the caller is responsible for ensuring `model` was fine-tuned on src.
    """
    m = zero_shot_eval(model, tgt_dataset, seed=seed, device=device)
    m["mode"] = "cross_sensor_zero_shot"
    m["src_dataset"] = src_dataset
    m["tgt_dataset"] = tgt_dataset
    return m


def save_result(tag: str, result: dict):
    p = os.path.join(RESULTS_DIR, tag + ".json")
    with open(p, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return p