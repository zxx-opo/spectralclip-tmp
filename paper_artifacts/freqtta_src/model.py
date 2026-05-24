"""FAM-SSM: Frequency-Aware Modulated Selective State-Space Model for HSI."""
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Frequency-Spectrum Modulator
# =============================================================================
class FSM(nn.Module):
    def __init__(self, n_bands: int, n_radial_bins: int = 16,
                 d_phi: int = 64, hidden: int = 128):
        super().__init__()
        self.n_bands = n_bands
        self.K = n_radial_bins
        in_dim = n_bands + n_radial_bins
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_phi),
        )
        self.d_phi = d_phi
        self._bin_cache = {}

    def _radial_bins(self, H, W, B, device):
        key = (H, W, B, str(device))
        if key in self._bin_cache:
            return self._bin_cache[key]
        kh = torch.fft.fftfreq(H, d=1.0).to(device)
        kw = torch.fft.fftfreq(W, d=1.0).to(device)
        kb = torch.fft.fftfreq(B, d=1.0).to(device)
        Kh, Kw, Kb = torch.meshgrid(kh, kw, kb, indexing="ij")
        rho = torch.sqrt(Kh ** 2 + Kw ** 2 + Kb ** 2)
        rho_max = rho.max() + 1e-6
        edges = torch.logspace(-3, math.log10(float(rho_max)), self.K + 1, device=device)
        bins = torch.bucketize(rho.flatten(), edges, right=False).clamp_(max=self.K) - 1
        bins.clamp_(min=0)
        self._bin_cache[key] = bins
        return bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Bsz, B, H, W)
        Bsz, Bn, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # (Bsz,H,W,B)
        # spectral energy profile
        spec = torch.fft.fft(x_perm, dim=-1)
        s = spec.abs().pow(2).mean(dim=(1, 2))  # (Bsz, B)
        # radial energy profile via 3D FFT
        X3 = torch.fft.fftn(x_perm, dim=(-3, -2, -1))
        pwr = X3.abs().pow(2).reshape(Bsz, -1)  # (Bsz, H*W*B)
        bins = self._radial_bins(H, W, Bn, x.device)
        r = x.new_zeros(Bsz, self.K)
        # scatter_add per-batch sum into K bins
        bins_b = bins.unsqueeze(0).expand(Bsz, -1)
        r.scatter_add_(1, bins_b, pwr)
        # normalize by bin counts
        counts = torch.zeros(self.K, device=x.device).scatter_add_(
            0, bins, torch.ones_like(pwr[0]))
        r = r / counts.clamp_min(1.0).unsqueeze(0)
        # log-compress
        s_log = torch.log1p(s)
        r_log = torch.log1p(r)
        feat = torch.cat([s_log, r_log], dim=-1)
        feat = self.norm(feat)
        phi = self.mlp(feat)
        return phi  # (Bsz, d_phi)


# =============================================================================
# FiLM projection from phi -> per-channel (gamma, beta)
# =============================================================================
class FiLMHead(nn.Module):
    def __init__(self, d_phi: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_phi, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, phi):
        gb = self.proj(phi)  # (Bsz, 2*D)
        gamma, beta = gb.chunk(2, dim=-1)  # (Bsz, D), (Bsz, D)
        return gamma, beta


# =============================================================================
# Modulated Selective Scan (FiLM-conditioned)
# =============================================================================
class ModulatedSelectiveScan(nn.Module):
    """A selective state-space scan whose input-dependent projections B_t, C_t,
    Delta_t are additively modulated by a per-sample descriptor phi:

        Delta_t = softplus( (1 + gamma_d(phi)) * W_d z_t + beta_d(phi) )
        B_t     = (1 + gamma_B(phi)) * W_B z_t + beta_B(phi)
        C_t     = (1 + gamma_C(phi)) * W_C z_t + beta_C(phi)

    When phi==0 (and FiLM heads zero-init), this reduces to a standard Mamba
    selective scan (B/C are linear in z; Delta = softplus(W_d z)).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_phi: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.W_d = nn.Linear(d_model, d_model, bias=True)
        self.W_B = nn.Linear(d_model, d_state, bias=True)
        self.W_C = nn.Linear(d_model, d_state, bias=True)
        self.film_d = FiLMHead(d_phi, d_model)
        self.film_B = FiLMHead(d_phi, d_state)
        self.film_C = FiLMHead(d_phi, d_state)
        # A: per-channel negative real scalar (log-parameterized)
        self.A_log = nn.Parameter(torch.zeros(d_model))
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # z: (Bsz, L, D), phi: (Bsz, d_phi)
        Bsz, L, D = z.shape
        N = self.d_state
        gd, bd = self.film_d(phi)  # (Bsz, D)
        gB, bB = self.film_B(phi)  # (Bsz, N)
        gC, bC = self.film_C(phi)  # (Bsz, N)
        # broadcast over L
        gd = gd.unsqueeze(1); bd = bd.unsqueeze(1)
        gB = gB.unsqueeze(1); bB = bB.unsqueeze(1)
        gC = gC.unsqueeze(1); bC = bC.unsqueeze(1)
        # input-dependent projections, FiLM-modulated
        Delta = F.softplus((1 + gd) * self.W_d(z) + bd)        # (Bsz, L, D)
        Bproj = (1 + gB) * self.W_B(z) + bB                    # (Bsz, L, N)
        Cproj = (1 + gC) * self.W_C(z) + bC                    # (Bsz, L, N)
        A = -torch.exp(self.A_log)                              # (D,) negative
        A_mean = A.mean()                                       # scalar for stability
        # SSM scan: h_t = decay * h_{t-1} + step * (Bproj_t * z_t)
        h = z.new_zeros(Bsz, D, N)
        outputs = []
        for t in range(L):
            dt = Delta[:, t, :]                                 # (Bsz, D)
            decay = torch.exp(dt * A_mean).unsqueeze(-1)        # (Bsz, D, 1)
            step = 1.0 - decay                                  # (Bsz, D, 1)
            Bt = Bproj[:, t, :].unsqueeze(1)                    # (Bsz, 1, N)
            Ct = Cproj[:, t, :].unsqueeze(1)                    # (Bsz, 1, N)
            zt = z[:, t, :].unsqueeze(-1)                       # (Bsz, D, 1)
            h = decay * h + step * (Bt * zt)                    # (Bsz, D, N)
            y_t = (Ct * h).sum(-1)                              # (Bsz, D)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)                         # (Bsz, L, D)
        return self.out_proj(y)


# =============================================================================
# FAM-SSM block: norm -> modulated scan -> mlp, residual
# =============================================================================
class FAMSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_phi: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.scan = ModulatedSelectiveScan(d_model, d_state, d_phi)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, z, phi):
        z = z + self.drop(self.scan(self.norm1(z), phi))
        z = z + self.drop(self.mlp(self.norm2(z)))
        return z


# =============================================================================
# Full FAM-SSM model
# =============================================================================
class FAMSSMHSI(nn.Module):
    def __init__(self, n_bands: int, n_classes: int, d_model: int = 64,
                 d_state: int = 16, d_phi: int = 64, n_layers: int = 2,
                 n_radial_bins: int = 16, dropout: float = 0.2):
        super().__init__()
        self.n_bands = n_bands
        self.fsm = FSM(n_bands=n_bands, n_radial_bins=n_radial_bins,
                       d_phi=d_phi)
        self.stem = nn.Sequential(
            nn.Conv2d(n_bands, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FAMSSMBlock(d_model, d_state, d_phi, dropout)
            for _ in range(n_layers)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.spec_decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, n_bands),
        )
        self.d_model = d_model

    def _backbone(self, x):
        """Returns (center_feature, phi) -- center pixel after FAM-SSM blocks."""
        phi = self.fsm(x)
        f = self.stem(x)                                # (Bsz, D, H, W)
        Bsz, D, H, W = f.shape
        z = f.flatten(2).transpose(1, 2)                # (Bsz, L=H*W, D)
        for blk in self.blocks:
            z = blk(z, phi)
        z = self.head_norm(z)
        center = z[:, (H * W) // 2, :]                  # (Bsz, D)
        return center, phi

    def forward(self, x, return_phi: bool = False):
        center, phi = self._backbone(x)
        logits = self.classifier(center)
        if return_phi:
            return logits, phi
        return logits

    def reconstruct_spectrum(self, x):
        center, _ = self._backbone(x)
        return self.spec_decoder(center)

    # -------- TTA-related parameter views --------
    def phi_params(self) -> List[nn.Parameter]:
        """Parameters of the FSM (the descriptor producer)."""
        return list(self.fsm.parameters())

    def film_params(self) -> List[nn.Parameter]:
        """Parameters of all FiLM projection heads (phi -> SSM-projection bias/gain)."""
        out = []
        for blk in self.blocks:
            for film in (blk.scan.film_d, blk.scan.film_B, blk.scan.film_C):
                out.extend(list(film.parameters()))
        return out

    def all_ln_params(self) -> List[nn.Parameter]:
        out = []
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None: out.append(m.weight)
                if m.bias is not None:   out.append(m.bias)
        return out

    # Legacy alias kept for backward compatibility with old tta.py
    def low_freq_ln_params(self):
        return self.all_ln_params()


# -----------------------------------------------------------------------------
# Backward-compatible export so old code using FreqTTAHSI keeps working.
FreqTTAHSI = FAMSSMHSI