"""SpecMamba: sensor-agnostic spectral tower for SpectralCLIP.

Architecture:
  - Wavelength-aware tokenization: each spectral band becomes one token with
    (reflectance, fourier wavelength embedding, local neighborhood) features.
  - L=4 selective state-space blocks (lightweight Mamba) process the sequence.
  - Mean-pool over valid bands -> projection head -> L2-normalized embedding.

Handles variable band counts: any (B,) spectrum -> fixed d_emb embedding.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_wavelength_embedding(wavelengths: torch.Tensor, dim: int = 32,
                                 wl_min: float = 350.0, wl_max: float = 2500.0):
    """Sinusoidal embedding of wavelengths (nm) -> (B, dim)."""
    # Normalize wavelength to [0, 1]
    wl = (wavelengths - wl_min) / (wl_max - wl_min)         # (B,)
    half = dim // 2
    # log-spaced frequencies
    freqs = torch.exp(torch.arange(half, dtype=wl.dtype, device=wl.device)
                      * (-math.log(10000.0) / max(1, half - 1)))
    args = wl.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def neighborhood_feature(spec: torch.Tensor, k: int = 7):
    """Gaussian-weighted local spectral neighborhood (Bsz, B) -> (Bsz, B, k)."""
    Bsz, B = spec.shape
    half = k // 2
    # pad with edge values
    padded = F.pad(spec.unsqueeze(1), (half, half), mode="replicate").squeeze(1)
    out = torch.zeros(Bsz, B, k, device=spec.device, dtype=spec.dtype)
    for off in range(k):
        out[..., off] = padded[:, off:off + B]
    return out


class SelectiveScan(nn.Module):
    """Simplified selective scan: O(N) recurrence with input-dependent B,C,delta."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1, bias=False)
        self.A_log = nn.Parameter(torch.empty(d_model))
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, L, D); mask: (B, L) bool
        Bsz, L, D = x.shape
        proj = self.x_proj(x)
        dt = F.softplus(proj[..., :1])                  # (B,L,1)
        Bp = proj[..., 1:1 + self.d_state]              # (B,L,N)
        Cp = proj[..., 1 + self.d_state:]               # (B,L,N)
        A = -torch.exp(self.A_log).mean()               # scalar global decay
        h = x.new_zeros(Bsz, D, self.d_state)
        outs = []
        for t in range(L):
            decay = torch.exp(dt[:, t, 0:1].unsqueeze(-1) * A)  # (B,1,1)
            step = 1.0 - decay
            x_t = x[:, t, :].unsqueeze(-1)                       # (B,D,1)
            Bp_t = Bp[:, t, :].unsqueeze(1)                       # (B,1,N)
            Cp_t = Cp[:, t, :].unsqueeze(1)                       # (B,1,N)
            h = decay * h + step * (Bp_t * x_t)                  # (B,D,N)
            y_t = (Cp_t * h).sum(-1)                              # (B,D)
            if mask is not None:
                y_t = y_t * mask[:, t:t + 1].float()
            outs.append(y_t)
        y = torch.stack(outs, dim=1)                            # (B,L,D)
        return self.out_proj(y)


class SpecMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.scan = SelectiveScan(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x, mask=None):
        x = x + self.scan(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SpecMamba(nn.Module):
    """Sensor-agnostic spectral tower.

    Input:  (B, B_max) raw reflectance + (B, B_max) mask + per-sample wavelength tensor.
    Output: (B, d_emb) L2-normalized embedding.
    """

    def __init__(self, d_model: int = 128, d_state: int = 16, n_layers: int = 4,
                 d_emb: int = 256, fourier_dim: int = 32, neighbor_k: int = 7,
                 wl_min: float = 350.0, wl_max: float = 2500.0):
        super().__init__()
        self.d_model = d_model
        self.fourier_dim = fourier_dim
        self.neighbor_k = neighbor_k
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.d_in = 1 + fourier_dim + neighbor_k
        self.input_proj = nn.Linear(self.d_in, d_model)
        self.blocks = nn.ModuleList([
            SpecMambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_emb),
        )

    def tokenize(self, spec: torch.Tensor, wavelengths: torch.Tensor):
        # spec: (B, B_max); wavelengths: (B, B_max) in nm
        Bsz, Bn = spec.shape
        ref = spec.unsqueeze(-1)                          # (B,B,1)
        wl_emb = fourier_wavelength_embedding(
            wavelengths.flatten(),
            dim=self.fourier_dim, wl_min=self.wl_min, wl_max=self.wl_max,
        ).reshape(Bsz, Bn, self.fourier_dim)              # (B,B,fourier_dim)
        nb = neighborhood_feature(spec, k=self.neighbor_k)  # (B,B,k)
        tok = torch.cat([ref, wl_emb, nb], dim=-1)        # (B,B,d_in)
        return self.input_proj(tok)                        # (B,B,d_model)

    def forward(self, spec: torch.Tensor, mask: torch.Tensor,
                wavelengths: torch.Tensor) -> torch.Tensor:
        """spec: (B, Bmax); mask: (B, Bmax) bool; wavelengths: (B, Bmax)."""
        x = self.tokenize(spec, wavelengths)              # (B, Bmax, d_model)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        # masked mean pool
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        emb = self.head(pooled)
        emb = F.normalize(emb, dim=-1)
        return emb


class TextProjection(nn.Module):
    """Linear projection on frozen sentence-transformer embeddings."""

    def __init__(self, d_text: int = 768, d_emb: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_text, d_emb)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.proj(text_emb)
        return F.normalize(z, dim=-1)


class SpectralCLIP(nn.Module):
    """Two-tower SpectralCLIP model.

    The text tower (sentence-transformer body) is frozen and encoded ONCE
    per training run; only TextProjection is trainable on the text side.
    """

    def __init__(self, d_model: int = 128, d_state: int = 16, n_layers: int = 4,
                 d_emb: int = 256, d_text: int = 768, wl_min: float = 350.0,
                 wl_max: float = 2500.0):
        super().__init__()
        self.spec_enc = SpecMamba(
            d_model=d_model, d_state=d_state, n_layers=n_layers,
            d_emb=d_emb, wl_min=wl_min, wl_max=wl_max,
        )
        self.text_proj = TextProjection(d_text=d_text, d_emb=d_emb)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_spectrum(self, spec, mask, wavelengths):
        return self.spec_enc(spec, mask, wavelengths)

    def encode_text(self, text_emb):
        return self.text_proj(text_emb)