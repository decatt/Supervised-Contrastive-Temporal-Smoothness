import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


def ensure_ri(x: torch.Tensor) -> torch.Tensor:
    """
    Real-imag packed real tensor.
    Accepts:
      - [B,17,32,48,2] float (..,0)=real, (..,1)=imag
      - complex [B,17,32,48] -> converted to [B,17,32,48,2]
    Returns: [B,17,32,48,2] float
    """
    if torch.is_complex(x):
        return torch.view_as_real(x)  # [...,2]
    if x.dim() == 5 and x.size(-1) == 2 and not torch.is_complex(x):
        return x
    raise ValueError(f"Unsupported input shape/dtype: shape={tuple(x.shape)}, dtype={x.dtype}")


def cv_features_from_ri(x_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input x_ri: [B,17,32,48,2] (real tensor)
    Outputs:
      feat_34ch: [B,34,32,48]   (34 as channels)
      feat_64ch: [B,17,64,48]   (64 on the '32' axis)
    """
    x_ri = ensure_ri(x_ri)                     # [B,17,32,48,2]
    xr = x_ri[..., 0]                          # [B,17,32,48]
    xi = x_ri[..., 1]                          # [B,17,32,48]

    feat_34ch = torch.cat([xr, xi], dim=1)     # [B,34,32,48]
    feat_64ch = torch.cat([xr, xi], dim=2)     # [B,17,64,48]
    return feat_34ch, feat_64ch


class DiscBranch34(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(34, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 32x48 -> 16x24
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), # 16x24 -> 8x12
            nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                         # [B,256,8,12]
        h = h.mean(dim=(2, 3))                  # [B,256]
        return self.proj(h)                     # [B,E]


class DiscBranch17(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(17, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 64x48 -> 32x24
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), # 32x24 -> 16x12
            nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                         # [B,256,16,12]
        h = h.mean(dim=(2, 3))                  # [B,256]
        return self.proj(h)                     # [B,E]


class CVDiscriminatorRI(nn.Module):
    """
    Input: x_ri [B,17,32,48,2] (real tensor, packed real/imag)
    Output: score [B]
    """
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.b34 = DiscBranch34(emb_dim=emb_dim)
        self.b17 = DiscBranch17(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x_ri: torch.Tensor) -> torch.Tensor:
        f34, f64 = cv_features_from_ri(x_ri)     # [B,34,32,48], [B,17,64,48]
        e1 = self.b34(f34)
        e2 = self.b17(f64)
        y = self.head(torch.cat([e1, e2], dim=1))
        return y.squeeze(1)


def gradient_penalty_wgan_gp_real(D: nn.Module, real_ri: torch.Tensor, fake_ri: torch.Tensor) -> torch.Tensor:
    """
    GP = E[(||∇_{x_hat} D(x_hat)||2 - 1)^2]
    All in real domain:
      real_ri, fake_ri: [B,17,32,48,2] float
    """
    real_ri = ensure_ri(real_ri)
    fake_ri = ensure_ri(fake_ri)

    B = real_ri.size(0)
    eps = torch.rand(B, 1, 1, 1, 1, device=real_ri.device, dtype=real_ri.dtype)  # broadcast
    x_hat = eps * real_ri + (1.0 - eps) * fake_ri
    x_hat = x_hat.requires_grad_(True)

    d_hat = D(x_hat)  # [B]
    grad = autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]              # [B,17,32,48,2] real

    grad = grad.reshape(B, -1)
    grad_norm = grad.norm(2, dim=1)             # [B]
    gp = (grad_norm - 1.0).pow(2).mean()
    return gp


def d_loss_wgan_gp_real(D: nn.Module, y_real_ri: torch.Tensor, y_fake_ri: torch.Tensor, lambda_gp: float):
    """
    Loss: D(y_fake).mean() - D(y_real).mean() + lambda * gp
    All inputs are real-packed complex: [B,17,32,48,2]
    """
    d_real = D(y_real_ri)
    d_fake = D(y_fake_ri)
    gp = gradient_penalty_wgan_gp_real(D, y_real_ri, y_fake_ri)
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    return loss, gp


# ------------------------- minimal usage example -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = CVDiscriminatorRI(emb_dim=256).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    B = 8
    # real-packed complex: [B,17,32,48,2]
    y_real_ri = torch.randn(B, 17, 32, 48, 2, device=device)
    y_fake_ri = torch.randn(B, 17, 32, 48, 2, device=device)

    lambda_gp = 10.0
    optD.zero_grad(set_to_none=True)
    lossD, gp = d_loss_wgan_gp_real(D, y_real_ri, y_fake_ri, lambda_gp=lambda_gp)
    lossD.backward()
    optD.step()

    print(f"lossD={lossD.item():.4f}, gp={gp.item():.4f}")
