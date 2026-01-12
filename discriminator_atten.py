import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm="bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
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
    [B,C,H,W] -> [B, feat_ch, H', W']  (two downsamples)
    """
    def __init__(self, in_ch, base=32, feat_ch=128, norm="bn"):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, base,        3, 1, 1, norm),
            ConvBlock(base, base,         3, 2, 1, norm),   # /2
            ConvBlock(base, base * 2,     3, 1, 1, norm),
            ConvBlock(base * 2, base * 2, 3, 2, 1, norm),   # /2
            ConvBlock(base * 2, feat_ch,  3, 1, 1, norm),
        )

    def forward(self, x):
        return self.net(x)


class BiCrossAttentionFuse(nn.Module):
    """
    Bi-directional cross-attention fusion on aligned feature maps.
    Input:
      a, b: [B, Ca, H, W], [B, Cb, H, W]  (already aligned)
    Output:
      fused_map: [B, d_model, H, W]
      (optionally) a_attn_map, b_attn_map
    """
    def __init__(self, ca, cb, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # 1x1 projections to a common token dim
        self.a_proj = nn.Conv2d(ca, d_model, kernel_size=1, bias=False)
        self.b_proj = nn.Conv2d(cb, d_model, kernel_size=1, bias=False)

        # cross-attn modules (batch_first=True -> [B,N,d])
        self.a_to_b = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.b_to_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

        # fuse the two updated streams back to one map
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _map_to_tokens(x):  # x: [B,C,H,W] -> [B,N,C]
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), h, w  # [B,N,C]

    @staticmethod
    def _tokens_to_map(t, h, w):  # t: [B,N,C] -> [B,C,H,W]
        b, n, c = t.shape
        return t.transpose(1, 2).contiguous().view(b, c, h, w)

    def forward(self, a_map, b_map, return_maps=False):
        # project to common d_model
        a = self.a_proj(a_map)  # [B,d,H,W]
        b = self.b_proj(b_map)  # [B,d,H,W]

        a_tok, h, w = self._map_to_tokens(a)  # [B,N,d]
        b_tok, _, _ = self._map_to_tokens(b)

        # pre-norm style cross attention
        a0 = self.norm_a(a_tok)
        b0 = self.norm_b(b_tok)

        # A queries, B keys/values
        a_attn, _ = self.a_to_b(query=a0, key=b0, value=b0, need_weights=False)
        a_tok2 = a_tok + self.drop(a_attn)

        # B queries, A keys/values
        b_attn, _ = self.b_to_a(query=b0, key=a0, value=a0, need_weights=False)
        b_tok2 = b_tok + self.drop(b_attn)

        # fuse token-wise then reshape back to map
        fused_tok = self.fuse(torch.cat([a_tok2, b_tok2], dim=-1))  # [B,N,d]
        fused_map = self._tokens_to_map(fused_tok, h, w)            # [B,d,H,W]

        if return_maps:
            return fused_map, self._tokens_to_map(a_tok2, h, w), self._tokens_to_map(b_tok2, h, w)
        return fused_map


class DualViewDiscAttnFuse(nn.Module):
    """
    Input: x [B,17,32,48]
      branch17: treat 17 as channels -> [B,feat_ch,h1,w1]
      branch32: treat 32 as channels -> permute to [B,32,17,48] -> [B,feat_ch,h2,w2]
    Align -> bi-cross-attn fuse -> conv head -> logit [B]
    """
    def __init__(
        self,
        base=32,
        feat_ch=128,
        fuse_hw=(8, 12),
        d_model=128,
        nhead=8,
        norm="bn",
        dropout=0.1,
    ):
        super().__init__()
        self.enc17 = CNNFeatMapEncoder(in_ch=17, base=base, feat_ch=feat_ch, norm=norm)
        self.enc32 = CNNFeatMapEncoder(in_ch=32, base=base, feat_ch=feat_ch, norm=norm)

        self.align = nn.AdaptiveAvgPool2d(fuse_hw)

        self.attn_fuse = BiCrossAttentionFuse(
            ca=feat_ch, cb=feat_ch, d_model=d_model, nhead=nhead, dropout=dropout
        )

        # conv head on fused map
        self.head = nn.Sequential(
            ConvBlock(d_model, d_model, k=3, s=1, p=1, norm=norm),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, 1, kernel_size=1),  # [B,1,Hf,Wf]
        )
        self.pool_logit = nn.AdaptiveAvgPool2d((1, 1))  # -> scalar logit

    def forward(self, x, return_debug=False):
        if x.ndim != 4 or x.shape[1:] != (17, 32, 48):
            raise ValueError(f"Expected x with shape [B,17,32,48], got {tuple(x.shape)}")

        f17 = self.enc17(x)  # [B,feat_ch,*,*]
        x32 = x.permute(0, 2, 1, 3).contiguous()  # [B,32,17,48]
        f32 = self.enc32(x32)  # [B,feat_ch,*,*]

        f17a = self.align(f17)  # [B,feat_ch,Hf,Wf]
        f32a = self.align(f32)  # [B,feat_ch,Hf,Wf]

        if return_debug:
            fused, a_upd, b_upd = self.attn_fuse(f17a, f32a, return_maps=True)
        else:
            fused = self.attn_fuse(f17a, f32a, return_maps=False)

        logit_map = self.head(fused)  # [B,1,Hf,Wf]
        logit = self.pool_logit(logit_map).flatten(1).squeeze(1)  # [B]

        if return_debug:
            return logit, {
                "f17": f17a, "f32": f32a,
                "f17_updated": a_upd, "f32_updated": b_upd,
                "fused": fused, "logit_map": logit_map
            }
        return logit


# -------------------- minimal usage example --------------------
if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 17, 32, 48)

    model = DualViewDiscAttnFuse(
        base=32, feat_ch=128, fuse_hw=(8, 12), d_model=128, nhead=8, dropout=0.1
    )
    logit, dbg = model(x, return_debug=True)

    y = torch.randint(0, 2, (B,)).float()
    loss = F.binary_cross_entropy_with_logits(logit, y)
    loss.backward()

    print("logit:", logit.shape, "loss:", float(loss))
    for k, v in dbg.items():
        print(k, tuple(v.shape))
