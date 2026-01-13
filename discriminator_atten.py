import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


def ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    if x.dim() == 5 and x.size(-1) == 2:
        return torch.complex(x[..., 0], x[..., 1])
    raise ValueError(f"Unsupported input shape/dtype: shape={tuple(x.shape)}, dtype={x.dtype}")


def cv_features(xc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input (complex):  [B,17,32,48]
    View A (real):    [B,34,32,48]  (34 as channels)
    View B (real):    [B,17,64,48]  (17 as channels; 64 is spatial dim)
    """
    xc = ensure_complex(xc)
    xr, xi = xc.real, xc.imag
    feat_34ch = torch.cat([xr, xi], dim=1)  # [B,34,32,48]
    feat_64h  = torch.cat([xr, xi], dim=2)  # [B,17,64,48]
    return feat_34ch, feat_64h


class ConvStem(nn.Module):
    """
    Turn a 2D feature map into a token sequence for attention.

    Input:  [B, Cin, H, W]
    Output: tokens [B, L, D] where L = (H/4)*(W/4) by default (2 strided convs)
    """
    def __init__(self, cin: int, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, d // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d // 2, d, 4, 2, 1),   # /2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d, d, 4, 2, 1),        # /4
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                      # [B, D, h, w]
        B, D, hH, hW = h.shape
        return h.flatten(2).transpose(1, 2)  # [B, L, D]


class CrossAttentionFusion(nn.Module):
    """
    Two-way cross attention:
      - A attends to B -> A'
      - B attends to A -> B'
    Then pool and fuse to a single embedding.
    """
    def __init__(self, d: int = 256, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

        self.norm_a = nn.LayerNorm(d)
        self.norm_b = nn.LayerNorm(d)

        # small FFN after attention (optional but helps stability)
        self.ff_a = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ff_b = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        self.norm_a2 = nn.LayerNorm(d)
        self.norm_b2 = nn.LayerNorm(d)

        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, tok_a: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        # tok_a: [B, La, D], tok_b: [B, Lb, D]

        # A <- attend(B)
        a_attn, _ = self.attn_a2b(query=tok_a, key=tok_b, value=tok_b, need_weights=False)
        tok_a = self.norm_a(tok_a + a_attn)
        tok_a = self.norm_a2(tok_a + self.ff_a(tok_a))

        # B <- attend(A)
        b_attn, _ = self.attn_b2a(query=tok_b, key=tok_a, value=tok_a, need_weights=False)
        tok_b = self.norm_b(tok_b + b_attn)
        tok_b = self.norm_b2(tok_b + self.ff_b(tok_b))

        # pool each side then fuse
        emb_a = tok_a.mean(dim=1)  # [B,D]
        emb_b = tok_b.mean(dim=1)  # [B,D]
        return self.fuse(torch.cat([emb_a, emb_b], dim=-1))  # [B,D]


class CVDiscriminatorCrossAttn(nn.Module):
    """
    Critic: complex input [B,17,32,48] -> score [B]

    Views:
      A: [B,34,32,48] -> tokens
      B: [B,17,64,48] -> tokens
    Fusion:
      cross attention between token sets, then pooled to embedding -> scalar score.
    """
    def __init__(self, d: int = 256, heads: int = 8):
        super().__init__()
        self.stem_a = ConvStem(cin=34, d=d)
        self.stem_b = ConvStem(cin=17, d=d)

        self.fusion = CrossAttentionFusion(d=d, heads=heads)

        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d, 1),
        )

    def forward(self, xc: torch.Tensor) -> torch.Tensor:
        f34, f64 = cv_features(xc)      # [B,34,32,48], [B,17,64,48]
        tok_a = self.stem_a(f34)        # [B, La, d]
        tok_b = self.stem_b(f64)        # [B, Lb, d]
        emb = self.fusion(tok_a, tok_b) # [B, d]
        return self.head(emb).squeeze(1)


@torch.no_grad()
def _rand_eps_like(xc: torch.Tensor) -> torch.Tensor:
    B = xc.shape[0]
    return torch.rand(B, 1, 1, 1, device=xc.device, dtype=xc.real.dtype)


def gradient_penalty_wgan_gp(D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real = ensure_complex(real)
    fake = ensure_complex(fake)

    eps = _rand_eps_like(real)
    x_hat = (eps * real + (1.0 - eps) * fake).requires_grad_(True)

    d_hat = D(x_hat)  # [B]
    grad = autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # complex [B,17,32,48]

    # complex -> real vector for norm
    grad_ri = torch.view_as_real(grad).reshape(grad.shape[0], -1)  # [B, ...*2]
    grad_norm = grad_ri.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def d_loss_wgan_gp(D: nn.Module, y_real: torch.Tensor, y_fake: torch.Tensor, lambda_gp: float) -> tuple[torch.Tensor, torch.Tensor]:
    d_real = D(y_real)
    d_fake = D(y_fake)
    gp = gradient_penalty_wgan_gp(D, y_real, y_fake)
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    return loss, gp


# ------------------------- minimal usage example -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = CVDiscriminatorCrossAttn(d=256, heads=8).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    B = 8
    y_real = torch.randn(B, 17, 32, 48, device=device) + 1j * torch.randn(B, 17, 32, 48, device=device)
    y_fake = torch.randn(B, 17, 32, 48, device=device) + 1j * torch.randn(B, 17, 32, 48, device=device)

    optD.zero_grad(set_to_none=True)
    lossD, gp = d_loss_wgan_gp(D, y_real, y_fake, lambda_gp=10.0)
    lossD.backward()
    optD.step()

    print(f"lossD={lossD.item():.4f}, gp={gp.item():.4f}")
