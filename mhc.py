import torch
import torch.nn as nn


class HyperConnectionsEq3(nn.Module):
    """
    Implements: x_{l+1} = H_res x_l + H_post^T * F(H_pre x_l)

    We store x as (B, L, n, C). Then:
      - H_res: (n, n) mixes lanes -> (B, L, n, C)
      - H_pre: (1, n) aggregates lanes -> (B, L, C)
      - H_post: (1, n) writes back -> (B, L, n, C) via transpose
    """

    def __init__(self, C: int, n: int, F_layer: nn.Module, init_identity: bool = True):
        super().__init__()
        self.C = C
        self.n = n
        self.F = F_layer  # expects (B,L,C) -> (B,L,C)

        # Learnable H matrices
        self.H_res = nn.Parameter(torch.zeros(n, n))
        self.H_pre = nn.Parameter(torch.zeros(1, n))   # row vector
        self.H_post = nn.Parameter(torch.zeros(1, n))  # row vector

        if init_identity:
            # A safe "start close to standard residual" initialization:
            # - H_res = I (each lane keeps itself)
            # - H_pre selects lane 0 as the layer input
            # - H_post writes output back to lane 0
            nn.init.eye_(self.H_res)
            self.H_pre.data.zero_()
            self.H_pre.data[0, 0] = 1.0
            self.H_post.data.zero_()
            self.H_post.data[0, 0] = 1.0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: either (B, L, n*C) or (B, L, n, C)
        returns same shape as input x
        """
        orig_shape_is_flat = (x.dim() == 3)

        if orig_shape_is_flat:
            B, L, NC = x.shape
            assert NC == self.n * self.C, f"Expected last dim {self.n*self.C}, got {NC}"
            x = x.view(B, L, self.n, self.C)  # (B,L,n,C)
        else:
            B, L, n, C = x.shape
            assert n == self.n and C == self.C, f"Expected (n,C)=({self.n},{self.C}), got ({n},{C})"

        # Term 1: H_res x_l  (lane mixing on n dimension)
        # (n,n) @ (B,L,n,C) -> (B,L,n,C)
        x_res = torch.einsum("ij,bljc->blic", self.H_res, x)

        # Compute layer input: H_pre x_l  (aggregate lanes)
        # (1,n) @ (B,L,n,C) -> (B,L,1,C) -> (B,L,C)
        x_in = torch.einsum("an,blnc->blac", self.H_pre, x).squeeze(2)

        # Sublayer: F(H_pre x_l, W_l)
        f_out = self.F(x_in, *args, **kwargs)  # (B,L,C)

        # Term 2: H_post^T * f_out  (write back to all lanes)
        # H_post^T: (n,1). Equivalent to per-lane scaling:
        # (B,L,C) with (n,) -> (B,L,n,C)
        post_weights = self.H_post.squeeze(0)  # (n,)
        x_post = torch.einsum("blc,n->blnc", f_out, post_weights)

        x_next = x_res + x_post  # (B,L,n,C)

        if orig_shape_is_flat:
            return x_next.reshape(B, L, self.n * self.C)
        return x_next


# --------- Example usage: wrap a Transformer sublayer F ----------

class SimpleFFN(nn.Module):
    def __init__(self, C: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, hidden),
            nn.GELU(),
            nn.Linear(hidden, C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    B, L, C, n = 2, 16, 64, 4
    F_layer = SimpleFFN(C=C, hidden=256)

    hc = HyperConnectionsEq3(C=C, n=n, F_layer=F_layer, init_identity=True)

    x0 = torch.randn(B, L, n * C)      # flattened nC stream
    x1 = hc(x0)                        # same shape (B,L,nC)
    print(x1.shape)

    x0_4d = torch.randn(B, L, n, C)    # explicit lanes
    x1_4d = hc(x0_4d)
    print(x1_4d.shape)
