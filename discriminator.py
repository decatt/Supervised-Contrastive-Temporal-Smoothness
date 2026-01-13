import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


def ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - complex tensor: [B,17,32,48] (dtype complex)
      - real tensor with last dim=2: [B,17,32,48,2]  (..,0)=real, (..,1)=imag
    Returns complex tensor [B,17,32,48]
    """
    if torch.is_complex(x):
        return x
    if x.dim() == 5 and x.size(-1) == 2:
        return torch.complex(x[..., 0], x[..., 1])
    raise ValueError(f"Unsupported input shape/dtype: shape={tuple(x.shape)}, dtype={x.dtype}")


def cv_features(xc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CV feature extraction from complex input xc: [B,17,32,48] (complex)

    Outputs:
      feat_34ch: [B,34,32,48]  (concat real/imag along '17' -> 34 channels)
      feat_64ch: [B,17,64,48]  (concat real/imag along '32' -> 64 channels)
    """
    xc = ensure_complex(xc)                      # [B,17,32,48] complex
    xr = xc.real                                 # [B,17,32,48]
    xi = xc.imag                                 # [B,17,32,48]

    # (1) 34 channels: stack real/imag on the 17-dim -> 34
    feat_34ch = torch.cat([xr, xi], dim=1)       # [B,34,32,48]

    # (2) 64 channels: stack real/imag on the 32-dim -> 64 (still keeping 17 as "channels")
    feat_64ch = torch.cat([xr, xi], dim=2)       # [B,17,64,48]

    return feat_34ch, feat_64ch


class DiscBranch34(nn.Module):
    """Input: [B,34,32,48] -> embedding [B,E]"""
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
        h = self.net(x)                          # [B,256,8,12]
        h = h.mean(dim=(2, 3))                   # GAP -> [B,256]
        return self.proj(h)                      # [B,E]


class DiscBranch17(nn.Module):
    """Input: [B,17,64,48] -> embedding [B,E]"""
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
        h = self.net(x)                          # [B,256,16,12]
        h = h.mean(dim=(2, 3))                   # [B,256]
        return self.proj(h)                      # [B,E]


class CVDiscriminator(nn.Module):
    """
    Discriminator output: realness score per sample, shape [B]
    Input: complex [B,17,32,48]
    Internally uses two CV feature views:
      - [B,34,32,48] with 34 as channels
      - [B,17,64,48] with 17 as channels
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

    def forward(self, xc: torch.Tensor) -> torch.Tensor:
        f34, f64 = cv_features(xc)                # f34:[B,34,32,48], f64:[B,17,64,48]
        e1 = self.b34(f34)                        # [B,E]
        e2 = self.b17(f64)                        # [B,E]
        y = self.head(torch.cat([e1, e2], dim=1)) # [B,1]
        return y.squeeze(1)                       # [B]


@torch.no_grad()
def _rand_eps_like(xc: torch.Tensor) -> torch.Tensor:
    # epsilon is real-valued; broadcast to [B,1,1,1] then multiplies complex fine
    B = xc.shape[0]
    return torch.rand(B, 1, 1, 1, device=xc.device, dtype=xc.real.dtype)


def gradient_penalty_wgan_gp(D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    GP = E[(||∇_{x_hat} D(x_hat)||2 - 1)^2]
    real/fake: complex [B,17,32,48]
    """
    real = ensure_complex(real)
    fake = ensure_complex(fake)

    eps = _rand_eps_like(real)                   # [B,1,1,1] real
    x_hat = eps * real + (1.0 - eps) * fake      # complex
    x_hat = x_hat.requires_grad_(True)

    d_hat = D(x_hat)                             # [B] real
    grad = autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]                                         # complex [B,17,32,48]

    # Convert complex grad -> real/imag and compute per-sample L2 norm over all dims.
    grad_ri = torch.view_as_real(grad)           # [B,17,32,48,2]
    grad_ri = grad_ri.reshape(grad_ri.size(0), -1)
    grad_norm = grad_ri.norm(2, dim=1)           # [B]
    gp = (grad_norm - 1.0).pow(2).mean()
    return gp


def d_loss_wgan_gp(D: nn.Module, y_real: torch.Tensor, y_fake: torch.Tensor, lambda_gp: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loss: D(fake).mean() - D(real).mean() + lambda * gp
    Returns: (loss, gp)
    """
    d_real = D(y_real)
    d_fake = D(y_fake)
    gp = gradient_penalty_wgan_gp(D, y_real, y_fake)
    loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    return loss, gp


# ------------------------- minimal usage example -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = CVDiscriminator(emb_dim=256).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))

    B = 8
    # Example complex inputs
    y_real = torch.randn(B, 17, 32, 48, device=device) + 1j * torch.randn(B, 17, 32, 48, device=device)
    y_fake = torch.randn(B, 17, 32, 48, device=device) + 1j * torch.randn(B, 17, 32, 48, device=device)

    lambda_gp = 10.0
    optD.zero_grad(set_to_none=True)
    lossD, gp = d_loss_wgan_gp(D, y_real, y_fake, lambda_gp=lambda_gp)
    lossD.backward()
    optD.step()

    print(f"lossD={lossD.item():.4f}, gp={gp.item():.4f}")
