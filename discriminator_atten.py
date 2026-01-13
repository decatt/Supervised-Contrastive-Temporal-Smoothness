import torch
import torch.nn as nn
from torch import autograd


# ---------- real-form utilities ----------
def to_ri(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex [B,17,32,48] to real-imag [B,17,32,48,2].
    If already real-imag, return as-is.
    """
    if torch.is_complex(x):
        return torch.view_as_real(x)  # [...,2]
    if x.dim() == 5 and x.size(-1) == 2:
        return x
    raise ValueError(f"Expect complex or [...,2] real-imag, got shape={tuple(x.shape)}, dtype={x.dtype}")


def cv_features_from_ri(x_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input (real): x_ri [B,17,32,48,2] where last dim is (real, imag)

    View A: [B,34,32,48]  (34 as channels)  = concat along channel(17) with real/imag
    View B: [B,17,64,48]  (17 as channels)  = concat along height(32) with real/imag
    """
    xr = x_ri[..., 0]  # [B,17,32,48]
    xi = x_ri[..., 1]  # [B,17,32,48]

    feat_34ch = torch.cat([xr, xi], dim=1)  # [B,34,32,48]
    feat_64h  = torch.cat([xr, xi], dim=2)  # [B,17,64,48]
    return feat_34ch, feat_64h


# ---------- model ----------
class ConvStem(nn.Module):
    """[B,C,H,W] -> tokens [B,L,D]"""
    def __init__(self, cin: int, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, d // 2, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d // 2, d, 4, 2, 1),   nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d, 4, 2, 1),        nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                      # [B,D,h,w]
        return h.flatten(2).transpose(1, 2)  # [B,L,D]


class CrossAttentionFusion(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(d); self.norm_a2 = nn.LayerNorm(d)
        self.norm_b1 = nn.LayerNorm(d); self.norm_b2 = nn.LayerNorm(d)
        self.ff_a = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ff_b = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.fuse = nn.Sequential(nn.Linear(2*d, d), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, tok_a: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        a_attn, _ = self.attn_a2b(tok_a, tok_b, tok_b, need_weights=False)
        tok_a = self.norm_a1(tok_a + a_attn)
        tok_a = self.norm_a2(tok_a + self.ff_a(tok_a))

        b_attn, _ = self.attn_b2a(tok_b, tok_a, tok_a, need_weights=False)
        tok_b = self.norm_b1(tok_b + b_attn)
        tok_b = self.norm_b2(tok_b + self.ff_b(tok_b))

        emb_a = tok_a.mean(dim=1)
        emb_b = tok_b.mean(dim=1)
        return self.fuse(torch.cat([emb_a, emb_b], dim=-1))  # [B,d]


class CVDiscriminatorCrossAttnReal(nn.Module):
    """
    Critic (real-form): input x_ri [B,17,32,48,2] -> score [B]
    """
    def __init__(self, d: int = 256, heads: int = 8):
        super().__init__()
        self.stem_a = ConvStem(cin=34, d=d)
        self.stem_b = ConvStem(cin=17, d=d)
        self.fusion = CrossAttentionFusion(d=d, heads=heads)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d, 1),
        )

    def forward(self, x_ri: torch.Tensor) -> torch.Tensor:
        x_ri = to_ri(x_ri)  # enforce real-form
        f34, f64 = cv_features_from_ri(x_ri)
        tok_a = self.stem_a(f34)
        tok_b = self.stem_b(f64)
        emb = self.fusion(tok_a, tok_b)
        return self.head(emb).squeeze(1)  # [B]


# ---------- WGAN-GP loss in real space ----------
def gradient_penalty_wgan_gp_real(D: nn.Module, real_ri: torch.Tensor, fake_ri: torch.Tensor) -> torch.Tensor:
    """
    real_ri/fake_ri: [B,17,32,48,2] real tensors
    GP = E[(||∇_{x_hat} D(x_hat)||2 - 1)^2] computed in real space
    """
    real_ri = to_ri(real_ri)
    fake_ri = to_ri(fake_ri)

    B = real_ri.size(0)
    eps = torch.rand(B, 1, 1, 1, 1, device=real_ri.device, dtype=real_ri.dtype)  # broadcast to [...,2]
    x_hat = (eps * real_ri + (1.0 - eps) * fake_ri).requires_grad_(True)

    d_hat = D(x_hat)  # [B], real-valued
    grad = autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B,17,32,48,2] real

    grad = grad.reshape(B, -1)           # flatten all real dims (including the 2)
    grad_norm = grad.norm(2, dim=1)      # [B]
    return (grad_norm.sub(1.0).pow(2)).mean()


def d_loss_wgan_gp_real(D: nn.Module, y_real_ri: torch.Tensor, y_fake_ri: torch.Tensor, lambda_gp: float):
    """
    Loss = D(fake).mean() - D(real).mean() + lambda_gp * gp
    All in real space.
    """
    y_real_ri = to_ri(y_real_ri)
    y_fake_ri = to_ri(y_fake_ri)

    d_real = D(y_real_ri)
    d_fake = D(y_fake_ri)
    gp = gradient_penalty_wgan_gp_real(D, y_real_ri, y_fake_ri)
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    return loss, gp


# ---------- minimal usage ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = CVDiscriminatorCrossAttnReal(d=256, heads=8).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    B = 8
    # create real-form inputs directly: [B,17,32,48,2]
    y_real_ri = torch.randn(B, 17, 32, 48, 2, device=device)
    y_fake_ri = torch.randn(B, 17, 32, 48, 2, device=device)

    optD.zero_grad(set_to_none=True)
    lossD, gp = d_loss_wgan_gp_real(D, y_real_ri, y_fake_ri, lambda_gp=10.0)
    lossD.backward()
    optD.step()

    print(f"lossD={lossD.item():.4f}, gp={gp.item():.4f}")
