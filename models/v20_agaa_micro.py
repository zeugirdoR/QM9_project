# models/v20_agaa_micro.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1️⃣ safe_norm
def safe_norm(v, dim=-1, eps=1e-8):
    return torch.sqrt(torch.clamp((v * v).sum(dim=dim), min=eps))


# 2️⃣ GaussianRBF  ← COPY THIS FROM YOUR GOOD NOTEBOOK
class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=20, cutoff=5.0):
        super().__init__()
        # 👉 in your original code you probably had a specific way of
        #    defining centers and widths. Copy that here.
        # Below is just a *placeholder* if you can’t find the original:
        centers = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full_like(centers, cutoff / n_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, d):
        d = d.unsqueeze(-1)  # (..., 1)
        x = (d - self.centers) / (self.widths + 1e-8)
        return torch.exp(-0.5 * x * x)


# 3️⃣ V20_Block_Motor  ← THIS MUST COME FROM YOUR ORIGINAL WORKING CODE
class V20_Block_Motor(nn.Module):
    def __init__(self, d_model, n_heads, n_rbf=20):
        super().__init__()
        # ❗❗ IMPORTANT:
        # Replace this entire block with the real V20_Block_Motor from
        # xBR3m_v2. This is just a dumb placeholder to make code run.
        self.lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, rbf_feat, pos, mask=None):
        # placeholder "motor signal"
        sig = h.abs().mean()
        h2 = self.lin(h)
        h2 = F.relu(h2)
        h2 = self.norm(h2)
        return h2, sig


# 4️⃣ Your full V20_AGAA_Motor from earlier
class V20_AGAA_Motor(nn.Module):
    def __init__(self, num_layers=7, d_model=192, n_heads=16, max_z=100, n_rbf=20):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_rbf   = n_rbf

        self.rbf   = GaussianRBF(n_rbf=n_rbf, cutoff=5.0)
        self.emb_z = nn.Embedding(max_z, d_model)
        self.emb_geo  = nn.Linear(4, d_model)
        self.emb_fuse = nn.Linear(2 * d_model, d_model)

        self.layers = nn.ModuleList(
            [V20_Block_Motor(d_model, n_heads, n_rbf=n_rbf) for _ in range(num_layers)]
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, z, pos, mask=None):
        device = z.device
        z_clamped = z.clamp(min=0)

        # (B, N, d_model) atom-type embeddings
        h_z = self.emb_z(z_clamped)

        # Pairwise distances and RBF features
        delta = pos.unsqueeze(2) - pos.unsqueeze(1)      # (B, N, N, 3)
        dist  = safe_norm(delta, dim=-1)                # (B, N, N)
        rbf_feat = self.rbf(dist)                       # (B, N, N, n_rbf)

        # Simple per-atom geometric summary: mean/min/max/std of distances
        if mask is not None:
            neigh_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            dist_masked = dist * neigh_mask
            valid_counts = neigh_mask.sum(dim=-1).clamp(min=1.0)

            mean_d = dist_masked.sum(dim=-1) / valid_counts
            min_d  = torch.where(
                neigh_mask.bool(),
                dist,
                torch.full_like(dist, 1e6)
            ).min(dim=-1).values
            max_d  = (dist * neigh_mask).max(dim=-1).values
            std_d  = torch.sqrt(
                torch.clamp((dist_masked ** 2).sum(dim=-1) / valid_counts - mean_d ** 2, min=0.0)
            )
        else:
            mean_d = dist.mean(dim=-1)
            min_d  = dist.min(dim=-1).values
            max_d  = dist.max(dim=-1).values
            std_d  = dist.std(dim=-1)

        geo_feat = torch.stack([mean_d, min_d, max_d, std_d], dim=-1)  # (B, N, 4)
        h_geo = self.emb_geo(geo_feat)

        h = torch.cat([h_z, h_geo], dim=-1)
        h = self.emb_fuse(h)

        sigs = []
        for layer in self.layers:
            h, sig = layer(h, rbf_feat, pos, mask)
            sigs.append(sig)

        h = self.norm_final(h)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            h_pool = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            h_pool = h.mean(dim=1)

        pred = self.head(h_pool)  # (B, 1)
        sig_motor = torch.stack(sigs).mean()
        return pred, sig_motor
