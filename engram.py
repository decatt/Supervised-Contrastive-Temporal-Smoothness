import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

def is_prime(n: int) -> bool:
    """简单判断素数，用于选取哈希表大小。"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i, step = 5, 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += step
        step = 6 - step
    return True

def next_prime(start: int) -> int:
    p = start + 1
    while not is_prime(p):
        p += 1
    return p

class EngramTorch(nn.Module):
    """
    PyTorch 版的简化 Engram 模块。

    参数说明：
      vocab_size: 词表大小（假设输入已是 token ID）
      hidden_size: 每个隐藏状态的维度
      max_ngram_size: 最大 n‑gram 长度（默认 3，即取 2-gram 和 3-gram）
      n_head_per_ngram: 每个 n‑gram 使用的哈希头数量（多头哈希）
      hc_mult: hyper-connection 分支数，每个分支有独立的 W_K，W_V 共享
      kernel_size: 因果卷积核宽度
      dilation: 因果卷积的 dilation
      embed_dim: 每个哈希头的 embedding 维度（记忆向量维度 = (max_ngram_size-1)*n_head*embed_dim）
    """
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_ngram_size: int = 3, n_head_per_ngram: int = 2,
                 hc_mult: int = 2, kernel_size: int = 4, dilation: int = 1,
                 embed_dim: int = 32, seed: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        self.hc_mult = hc_mult
        self.kernel_size = kernel_size
        self.dilation = dilation

        # 生成素数大小的 embedding 表，用于多头哈希:contentReference[oaicite:1]{index=1}。
        torch.manual_seed(seed)
        self.primes: Dict[Tuple[int, int], int] = {}
        base = vocab_size * 2 + 1
        for n in range(2, max_ngram_size + 1):
            cur = base
            for h in range(n_head_per_ngram):
                p = next_prime(cur)
                self.primes[(n, h)] = p
                cur = p + 1

        # 记录总的记忆向量维度
        self.embed_dim = embed_dim
        self.total_embed_dim = (max_ngram_size - 1) * n_head_per_ngram * embed_dim

        # 为每个 (n, head) 创建独立的 embedding table
        self.embeddings = nn.ModuleDict()
        for n in range(2, max_ngram_size + 1):
            for h in range(n_head_per_ngram):
                p = self.primes[(n, h)]
                # embedding_size = embed_dim
                self.embeddings[f"{n}_{h}"] = nn.Embedding(p, embed_dim)

        # 为每个分支创建 key projection；value 投影共享
        self.W_k = nn.ModuleList([
            nn.Linear(self.total_embed_dim, hidden_size) for _ in range(hc_mult)
        ])
        self.W_v = nn.Linear(self.total_embed_dim, hidden_size)

        # 深度可分离卷积：groups = (hc_mult * hidden_size)
        # 使用 Conv1d 方便按 seq 维度做因果卷积:contentReference[oaicite:2]{index=2}。
        self.conv = nn.Conv1d(
            in_channels=hc_mult * hidden_size,
            out_channels=hc_mult * hidden_size,
            kernel_size=kernel_size,
            groups=hc_mult * hidden_size,
            bias=False,
            dilation=dilation
        )
        # 初始化卷积权重
        nn.init.normal_(self.conv.weight, std=0.02)

    def _hash_suffix(self, tokens: torch.LongTensor, n: int) -> torch.LongTensor:
        """
        对单条序列 tokens (L,) 计算每个位置的 n‑gram 哈希索引。
        返回形状：(L, n_head_per_ngram)
        """
        L = tokens.size(0)
        hashes = torch.zeros(L, self.n_head_per_ngram, dtype=torch.long, device=tokens.device)
        # 前缀乘数，用于构造简单的乘法+异或哈希
        multipliers = torch.arange(1, n + 1, device=tokens.device)
        for h in range(self.n_head_per_ngram):
            prime = self.primes[(n, h)]
            for t in range(L):
                # 取 n 个后缀 token，不足补 0
                start = max(0, t - (n - 1))
                window = tokens[start:t + 1]
                if window.size(0) < n:
                    pad = torch.zeros(n - window.size(0), dtype=window.dtype, device=window.device)
                    window = torch.cat([pad, window])
                # 计算异或哈希
                val = 0
                for mul, tok in zip(multipliers, window):
                    val = val ^ (int(tok) * int(mul))
                hashes[t, h] = val % prime
        return hashes

    def _retrieve_memory(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        根据输入 token IDs 检索并拼接 n‑gram embedding，形状：(B, L, total_embed_dim)
        """
        B, L = input_ids.shape
        device = input_ids.device
        num_ngrams = self.max_ngram_size - 1
        total_heads = num_ngrams * self.n_head_per_ngram

        # 初始化输出张量
        memory = input_ids.new_zeros(B, L, total_heads * self.embed_dim, dtype=torch.float32)
        for n_idx, n in enumerate(range(2, self.max_ngram_size + 1)):
            for b in range(B):
                # 每个 batch 单独处理，生成 (L, n_head_per_ngram) 的哈希索引
                hashes = self._hash_suffix(input_ids[b], n)  # (L, heads)
                for h in range(self.n_head_per_ngram):
                    # 查表获得 embedding：(L, embed_dim)
                    emb = self.embeddings[f"{n}_{h}"](hashes[:, h])
                    head_index = n_idx * self.n_head_per_ngram + h
                    start = head_index * self.embed_dim
                    end = start + self.embed_dim
                    memory[b, :, start:end] = emb
        return memory

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        RMSNorm：x / sqrt(mean(x^2) + eps)，不改变形状。
        """
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        hidden_states: (B, L, hc_mult, hidden_size)
        input_ids: (B, L)
        输出形状同 hidden_states
        """
        B, L, G, D = hidden_states.shape
        # 1. 检索并拼接记忆向量:contentReference[oaicite:3]{index=3}
        memory = self._retrieve_memory(input_ids)        # (B, L, total_embed_dim)

        # 2. 投影得到 keys/values:contentReference[oaicite:4]{index=4}
        values = self.W_v(memory)                        # (B, L, hidden_size)
        keys = [wk(memory) for wk in self.W_k]           # 列表，长 hc_mult，每个 (B,L,hidden_size)

        # 3. 融合：计算门控，生成 gated 输出:contentReference[oaicite:5]{index=5}
        # gated_output: (B, L, hc_mult, hidden_size)
        gated_output = hidden_states.new_zeros(B, L, G, D)
        for g in range(G):
            # 当前分支的 key 和 query
            k = keys[g]                    # (B,L,D)
            q = hidden_states[:, :, g, :]  # (B,L,D)
            # RMSNorm
            nk = self._rms_norm(k)
            nq = self._rms_norm(q)
            # 标量门控 alpha = sigma( (nk * nq).sum(dim=-1) / sqrt(D) )
            dot = (nk * nq).sum(dim=-1, keepdim=True)
            alpha = torch.sigmoid(dot / math.sqrt(D))
            # 将 value 投影赋予门控；value 在所有分支共享
            gated_output[:, :, g, :] = alpha * values

        # 4. 因果卷积：对每个 (batch, branch, dimension) 独立卷积:contentReference[oaicite:6]{index=6}
        # 先 reshape 为 (B*G*D, 1, L)
        out = gated_output.permute(0, 2, 3, 1).contiguous()    # (B, G, D, L)
        out = out.view(B * G * D, 1, L)
        # 因为 groups = B*G*D，每个通道用独立卷积核
        weight = self.conv.weight.view(-1, 1, self.kernel_size)   # (B*G*D,1,kernel)
        conv_out = F.conv1d(out,
                            weight=weight,
                            bias=None,
                            dilation=self.dilation,
                            padding=(self.kernel_size - 1) * self.dilation,
                            groups=out.size(0))
        conv_out = conv_out[..., :L]  # 取与输入等长的部分
        conv_out = F.silu(conv_out)   # 非线性激活
        conv_out = conv_out.view(B, G, D, L).permute(0, 3, 1, 2)  # (B,L,G,D)

        # 5. 残差加回：返回 gated_output + conv_out
        return gated_output + conv_out

if __name__ == "__main__":
# 假设词表大小 100，隐藏维度 64
engram = EngramTorch(vocab_size=100, hidden_size=64,
                     max_ngram_size=3, n_head_per_ngram=2,
                     hc_mult=2, kernel_size=3, dilation=1,
                     embed_dim=16)

# input_ids: (batch=2, seq_len=5)
input_ids = torch.tensor([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10]], dtype=torch.long)
# hidden_states: (batch=2, seq_len=5, hc_mult=2, hidden_size=64)
hidden_states = torch.randn(2, 5, 2, 64)

# 前向计算
output = engram(hidden_states, input_ids)
print(output.shape)  # torch.Size([2, 5, 2, 64])

