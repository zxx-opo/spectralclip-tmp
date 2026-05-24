"""Baselines: HybridSN, SpectralFormerLite, S2MambaLite."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import ModulatedSelectiveScan  # only used for shape compatibility
# NOTE: We reimplement a vanilla SelectiveScan here so baselines do NOT depend
# on the modulated variant.


class _VanillaScan(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.A_log = nn.Parameter(torch.zeros(d_model))
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, z):
        Bsz, L, D = z.shape
        proj = self.x_proj(z)
        dt = F.softplus(proj[..., :1])
        Bp = proj[..., 1:1 + self.d_state]
        Cp = proj[..., 1 + self.d_state:]
        A = -torch.exp(self.A_log)
        h = z.new_zeros(Bsz, D, self.d_state)
        outs = []
        Am = A.mean()
        for t in range(L):
            decay = torch.exp(dt[:, t, 0].unsqueeze(-1).unsqueeze(-1) * Am)
            step = 1.0 - decay
            Bt = Bp[:, t, :].unsqueeze(1)
            Ct = Cp[:, t, :].unsqueeze(1)
            zt = z[:, t, :].unsqueeze(-1)
            h = decay * h + step * (Bt * zt)
            outs.append((Ct * h).sum(-1))
        y = torch.stack(outs, dim=1)
        return self.out_proj(y)


# ----------------------------------------------------------------------------
class HybridSN(nn.Module):
    def __init__(self, n_bands, n_classes, patch_size=11):
        super().__init__()
        self.conv3d1 = nn.Conv3d(1, 8, (7, 3, 3))
        self.conv3d2 = nn.Conv3d(8, 16, (5, 3, 3))
        self.conv3d3 = nn.Conv3d(16, 32, (3, 3, 3))
        out_b = n_bands - 7 - 5 - 3 + 3
        out_s = patch_size - 2 - 2 - 2
        self.conv2d = nn.Conv2d(32 * out_b, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * (out_s - 2) * (out_s - 2), 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.4)
        self.act = nn.ReLU(inplace=True)
        self.n_bands = n_bands

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act(self.conv3d1(x))
        x = self.act(self.conv3d2(x))
        x = self.act(self.conv3d3(x))
        b, c, b_, h, w = x.shape
        x = x.reshape(b, c * b_, h, w)
        x = self.act(self.conv2d(x))
        x = x.flatten(1)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)

    def reconstruct_spectrum(self, x):
        return torch.zeros(x.size(0), self.n_bands, device=x.device)

    def all_ln_params(self):
        out = []
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None: out.append(m.weight)
                if m.bias is not None:   out.append(m.bias)
        return out

    def phi_params(self):  return []
    def film_params(self): return []
    def low_freq_ln_params(self): return self.all_ln_params()


class SpectralFormerLite(nn.Module):
    def __init__(self, n_bands, n_classes, patch_size=11,
                 group=5, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.n_bands = n_bands
        self.group = group
        self.n_tokens = n_bands // group
        self.embed = nn.Linear(group * patch_size * patch_size, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        Bsz, B, H, W = x.shape
        g = self.group; nt = self.n_tokens
        x = x[:, :nt * g, :, :].reshape(Bsz, nt, g, H, W).reshape(Bsz, nt, g * H * W)
        tok = self.embed(x)
        cls = self.cls_token.expand(Bsz, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = self.encoder(tok)
        return self.head(tok[:, 0])

    def reconstruct_spectrum(self, x):
        return torch.zeros(x.size(0), self.n_bands, device=x.device)

    def all_ln_params(self):
        out = []
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None: out.append(m.weight)
                if m.bias is not None:   out.append(m.bias)
        return out

    def phi_params(self):  return []
    def film_params(self): return []
    def low_freq_ln_params(self): return self.all_ln_params()


class S2MambaLite(nn.Module):
    def __init__(self, n_bands, n_classes, d_model=64, d_state=16, patch_size=11):
        super().__init__()
        self.n_bands = n_bands
        self.conv = nn.Conv2d(n_bands, d_model, kernel_size=3, padding=1)
        self.norm0 = nn.GroupNorm(8, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.scan = _VanillaScan(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.head = nn.Linear(d_model, n_classes)
        self.spec_decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, n_bands),
        )

    def _backbone(self, x):
        f = F.gelu(self.norm0(self.conv(x)))
        Bsz, D, H, W = f.shape
        tok = f.flatten(2).transpose(1, 2)
        tok = tok + self.scan(self.norm1(tok))
        tok = tok + self.mlp(self.norm2(tok))
        return tok[:, (H * W) // 2, :]

    def forward(self, x):
        return self.head(self._backbone(x))

    def reconstruct_spectrum(self, x):
        return self.spec_decoder(self._backbone(x))

    def all_ln_params(self):
        out = []
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None: out.append(m.weight)
                if m.bias is not None:   out.append(m.bias)
        return out

    def phi_params(self):  return []
    def film_params(self): return []
    def low_freq_ln_params(self): return self.all_ln_params()


def build_baseline(name, n_bands, n_classes, patch_size=11):
    if name == "hybridsn":       return HybridSN(n_bands, n_classes, patch_size)
    if name == "spectralformer": return SpectralFormerLite(n_bands, n_classes, patch_size)
    if name == "s2mamba":        return S2MambaLite(n_bands, n_classes)
    raise ValueError(name)