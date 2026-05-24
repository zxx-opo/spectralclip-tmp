"""Text-side helper: load sentence-transformer, encode caption list ONCE,
cache to disk, and return tensors usable by SpectralCLIP.

We use the publicly available 'sentence-transformers/all-mpnet-base-v2'
model (110M params), but we ONLY use it at preprocessing time -- the
model itself is not loaded into GPU memory during training.
"""
import os
import json
import hashlib
import numpy as np
import torch
from typing import List

from .config import CFG, CORPUS_DIR


_CACHE_DIR = os.path.join(CORPUS_DIR, "text_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(captions: List[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    for c in captions:
        h.update(c.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode_captions(captions: List[str], model_name: str = None,
                    device: str = "cpu") -> torch.Tensor:
    """Encode a list of caption strings with a sentence-transformer.

    Caches the result to disk keyed by (model_name, captions).
    Returns: (N, d_text) float tensor on CPU.
    """
    model_name = model_name or CFG.text_model_name
    key = _cache_key(captions, model_name)
    cache_path = os.path.join(_CACHE_DIR, f"emb_{key}.pt")
    meta_path = os.path.join(_CACHE_DIR, f"meta_{key}.json")
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        return torch.load(cache_path, map_location="cpu")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for text encoding.\n"
            "  pip install sentence-transformers"
        ) from e

    print(f"[text_encoder] Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    print(f"[text_encoder] Encoding {len(captions)} captions...")
    emb = model.encode(
        captions, batch_size=32, convert_to_tensor=True,
        normalize_embeddings=False, show_progress_bar=False,
    )
    emb = emb.cpu()
    torch.save(emb, cache_path)
    with open(meta_path, "w") as f:
        json.dump({"model": model_name, "n": len(captions),
                   "d_text": emb.shape[-1]}, f, indent=2)
    print(f"[text_encoder] Cached -> {cache_path}  shape={tuple(emb.shape)}")
    # free memory immediately
    del model
    return emb