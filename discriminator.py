import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, norm: str = "bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            g = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
            self.norm = nn.GroupNorm(g, out_ch)
        else:
            raise ValueError("norm must be 'bn' or 'gn'")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNFeatMapEncoder(nn.Module):
    """
    [B,C,H,W] -> feature map [B, out_ch, H', W']
    """
    def __init__(self, in_ch: int, base: int = 32, out_ch: int = 128, norm: str = "bn"):
        super().__init__()
        # two downsamples -> H,W roughly /4
        self.net = nn.Sequential(
            ConvBlock(in_ch, base,        k=3, s=1, p=1, norm=norm),
            ConvBlock(base, base,         k=3, s=2, p=1, norm=norm),      # /2
            ConvBlock(base, base * 2,     k=3, s=1, p=1, norm=norm),
            ConvBlock(base * 2, base * 2, k=3, s=2, p=1, norm=norm),      # /2
            ConvBlock(base * 2, out_ch,   k=3, s=1, p=1, norm=norm),
        )

    def forward(self, x):
        return self.net(x)


class DualViewDiscFeatMap(nn.Module):
    """
    Input:  x [B,17,32,48]
    BranchA: encoder17(x) where C=17
    BranchB: encoder32(x.permute -> [B,32,17,48]) where C=32

    To fuse: spatially align both feature maps to (Hf,Wf) via AdaptiveAvgPool2d,
             concat on channel dim, then conv head -> logit [B].
    """
    def __init__(
        self,
        base: int = 32,
        feat_ch: int = 128,
        fuse_hw: tuple[int, int] = (8, 12),
        norm: str = "bn",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder17 = CNNFeatMapEncoder(in_ch=17, base=base, out_ch=feat_ch, norm=norm)
        self.encoder32 = CNNFeatMapEncoder(in_ch=32, base=base, out_ch=feat_ch, norm=norm)

        self.align = nn.AdaptiveAvgPool2d(fuse_hw)

        # conv head on fused feature map
        self.head = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch) if norm == "bn" else nn.GroupNorm(8 if feat_ch % 8 == 0 else 1, feat_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(feat_ch, 1, kernel_size=1)  # [B,1,Hf,Wf]
        )

        self.pool_logit = nn.AdaptiveAvgPool2d((1, 1))  # -> [B,1,1,1]

    def forward(self, x, return_featmaps: bool = False):
        if x.ndim != 4 or x.shape[1:] != (17, 32, 48):
            raise ValueError(f"Expected x with shape [B,17,32,48], got {tuple(x.shape)}")

        # A: 17-as-channel
        f17 = self.encoder17(x)  # [B,feat_ch,h1,w1]

        # B: 32-as-channel (swap 17 and 32)
        x32 = x.permute(0, 2, 1, 3).contiguous()  # [B,32,17,48]
        f32 = self.encoder32(x32)                 # [B,feat_ch,h2,w2]

        # align spatial size for fusion
        f17a = self.align(f17)  # [B,feat_ch,Hf,Wf]
        f32a = self.align(f32)  # [B,feat_ch,Hf,Wf]

        fused = torch.cat([f17a, f32a], dim=1)  # [B,2*feat_ch,Hf,Wf]
        logit_map = self.head(fused)            # [B,1,Hf,Wf]
        logit = self.pool_logit(logit_map).flatten(1).squeeze(1)  # [B]

        if return_featmaps:
            # return aligned feature maps and logit_map for inspection/visualization
            return logit, {"f17": f17a, "f32": f32a, "fused": fused, "logit_map": logit_map}
        return logit


# -------------------- minimal usage example --------------------
if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 17, 32, 48)

    model = DualViewDiscFeatMap(base=32, feat_ch=128, fuse_hw=(8, 12), norm="bn", dropout=0.1)
    logit, feats = model(x, return_featmaps=True)  # logit [B], feats dict

    y = torch.randint(0, 2, (B,)).float()
    loss = F.binary_cross_entropy_with_logits(logit, y)
    loss.backward()

    print("logit:", logit.shape, "loss:", float(loss))
    for k, v in feats.items():
        print(k, tuple(v.shape))
