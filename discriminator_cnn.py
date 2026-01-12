import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d -> Norm -> SiLU -> (optional) downsample"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, norm: str = "bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            # 8 groups is a reasonable default; ensure divisible
            g = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
            self.norm = nn.GroupNorm(g, out_ch)
        else:
            raise ValueError("norm must be 'bn' or 'gn'")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNEncoder(nn.Module):
    """
    Simple CNN feature extractor:
    [B,C,H,W] -> feature vector [B,feat_dim]
    """
    def __init__(self, in_ch: int, base: int = 32, feat_dim: int = 128, norm: str = "bn"):
        super().__init__()
        # Stem + 2 downsamples (lightweight)
        self.net = nn.Sequential(
            ConvBlock(in_ch, base,     k=3, s=1, p=1, norm=norm),
            ConvBlock(base, base,      k=3, s=2, p=1, norm=norm),   # /2
            ConvBlock(base, base * 2,  k=3, s=1, p=1, norm=norm),
            ConvBlock(base * 2, base * 2, k=3, s=2, p=1, norm=norm),# /2
            ConvBlock(base * 2, base * 4, k=3, s=1, p=1, norm=norm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(base * 4, feat_dim)

    def forward(self, x):
        x = self.net(x)                       # [B,*,h,w]
        x = self.pool(x).flatten(1)           # [B,channels]
        x = self.proj(x)                      # [B,feat_dim]
        return x


class DualViewDiscriminator(nn.Module):
    """
    Input:  x [B,17,32,48]
    BranchA: treat 17 as channels -> encoder17(x)
    BranchB: treat 32 as channels -> encoder32(permute to [B,32,17,48])
    Fuse: concat -> MLP -> 1 logit (binary)
    """
    def __init__(self, feat_dim: int = 128, base: int = 32, norm: str = "bn", dropout: float = 0.1):
        super().__init__()
        self.encoder17 = CNNEncoder(in_ch=17, base=base, feat_dim=feat_dim, norm=norm)
        self.encoder32 = CNNEncoder(in_ch=32, base=base, feat_dim=feat_dim, norm=norm)

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1)  # output logit
        )

    def forward(self, x, return_features: bool = False):
        """
        x: [B,17,32,48]
        returns:
          logit: [B]  (before sigmoid)
          optionally (f17, f32): [B,feat_dim] each
        """
        if x.ndim != 4 or x.shape[1:] != (17, 32, 48):
            raise ValueError(f"Expected x with shape [B,17,32,48], got {tuple(x.shape)}")

        # Branch A: 17 as channels
        f17 = self.encoder17(x)  # [B,feat_dim]

        # Branch B: 32 as channels -> [B,32,17,48]
        x32 = x.permute(0, 2, 1, 3).contiguous()
        f32 = self.encoder32(x32)  # [B,feat_dim]

        fused = torch.cat([f17, f32], dim=1)   # [B,2*feat_dim]
        logit = self.head(fused).squeeze(1)    # [B]

        if return_features:
            return logit, (f17, f32)
        return logit


# -------------------- minimal usage example --------------------
if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 17, 32, 48)

    model = DualViewDiscriminator(feat_dim=128, base=32, norm="bn", dropout=0.1)
    logit = model(x)                # [B]
    prob = torch.sigmoid(logit)     # [B] probability of class=1

    y = torch.randint(0, 2, (B,)).float()     # binary labels in {0,1}
    loss = F.binary_cross_entropy_with_logits(logit, y)
    loss.backward()
    print("ok:", logit.shape, loss.item())
