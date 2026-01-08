import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# -----------------------------
# 1) Dataset（你可以替换成自己的数据读取）
# -----------------------------
class ChannelDataset(Dataset):
    """
    每条样本:
      x:  [34,32,48]  (float32)
      scene_id: int
      time_step: int   (建议用 0..1999 的整数步；间隔5ms你也可以存 ms，但这里用步数更方便)
    """
    def __init__(self, xs: torch.Tensor, scene_ids: torch.Tensor, time_steps: torch.Tensor):
        assert xs.ndim == 4 and xs.shape[1:] == (34, 32, 48)  # [N,34,32,48]
        self.xs = xs
        self.scene_ids = scene_ids.long()
        self.time_steps = time_steps.long()

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.scene_ids[idx], self.time_steps[idx]


# -----------------------------
# 2) 按场景采样的 Batch Sampler（保证 batch 内有多个场景且每场景多个时间点，正/负样本自然充足）
# -----------------------------
class SceneBatchSampler(Sampler[List[int]]):
    """
    每个 batch:
      - 取 num_scenes 个不同场景
      - 每个场景取 samples_per_scene 个样本（随机时间点）
    batch_size = num_scenes * samples_per_scene
    """
    def __init__(
        self,
        scene_to_indices: Dict[int, List[int]],
        num_scenes: int,
        samples_per_scene: int,
        num_batches: int,
        seed: int = 0,
    ):
        self.scene_to_indices = {k: v[:] for k, v in scene_to_indices.items()}
        self.scenes = list(self.scene_to_indices.keys())
        self.num_scenes = num_scenes
        self.samples_per_scene = samples_per_scene
        self.num_batches = num_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_scenes = self.rng.sample(self.scenes, k=self.num_scenes)
            batch = []
            for s in chosen_scenes:
                idxs = self.scene_to_indices[s]
                # 放回采样，避免某些场景样本数不足
                for _ in range(self.samples_per_scene):
                    batch.append(self.rng.choice(idxs))
            self.rng.shuffle(batch)
            yield batch


# -----------------------------
# 3) 模型：子带共享编码器 +（可选）子带 embedding
# -----------------------------
class StaticSubbandEncoder(nn.Module):
    def __init__(self, subbands: int = 34, in_ant: int = 32, in_re: int = 48, emb_dim: int = 8,
                 hidden: int = 256, use_subband_emb: bool = True, subband_emb_dim: int = 8):
        super().__init__()
        self.subbands = subbands
        self.emb_dim = emb_dim
        self.use_subband_emb = use_subband_emb

        in_dim = in_ant * in_re
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

        if use_subband_emb:
            # 子带 embedding（很轻量）：让模型更容易区分“子带差异”和“场景差异”
            # 做法：z = normalize(z + e_k)
            assert subband_emb_dim == emb_dim, "为简单起见这里让子带embedding维度=emb_dim"
            self.subband_emb = nn.Embedding(subbands, emb_dim)
        else:
            self.subband_emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,34,32,48]
        return z: [B,34,8] (L2 normalize)
        """
        B, S, A, R = x.shape
        assert S == self.subbands
        x = x.reshape(B * S, A * R)
        z = self.mlp(x).reshape(B, S, self.emb_dim)

        if self.use_subband_emb:
            device = z.device
            sb_idx = torch.arange(self.subbands, device=device)  # [34]
            e = self.subband_emb(sb_idx)[None, :, :]            # [1,34,8]
            z = z + e

        z = F.normalize(z, dim=-1)  # cosine / InfoNCE 推荐
        return z


# -----------------------------
# 4) 损失：Temporal SupCon / InfoNCE（正：同场景且时间接近；负：其余，尤其不同场景）
# -----------------------------
def temporal_supcon_loss(
    z: torch.Tensor,                 # [B,34,8] normalized
    scene_id: torch.Tensor,          # [B]
    time_step: torch.Tensor,         # [B]
    delta_steps: int = 10,           # 时间接近阈值（10步=50ms，如果步长=5ms）
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    对每个子带 k：
      - sim(i,j)=cos(z_i,k, z_j,k)
      - positives: same scene AND |t_i - t_j| <= delta_steps AND i!=j
      - denominator: all j != i (包含同场景远时间/不同场景)
    """
    B, S, D = z.shape
    device = z.device

    # [B,B] 构造正样本 mask
    same_scene = scene_id[:, None].eq(scene_id[None, :])                 # [B,B]
    close_time = (time_step[:, None] - time_step[None, :]).abs() <= delta_steps
    not_self = ~torch.eye(B, device=device, dtype=torch.bool)
    pos_mask = same_scene & close_time & not_self                        # [B,B]

    # 如果某些 anchor 在 batch 内没有正样本，会导致 NaN；我们对这些 anchor 忽略
    valid_anchor = pos_mask.any(dim=1)                                   # [B]

    loss_all_sb = []
    for k in range(S):
        zk = z[:, k, :]                                                  # [B,D]
        # cosine sim (zk已normalize，点积即可)
        logits = (zk @ zk.t()) / temperature                             # [B,B]
        # 去掉 self 对比：在 logsumexp 前把对角置为 -inf
        logits = logits.masked_fill(~not_self, float("-inf"))

        # log denom: log sum_{j!=i} exp(logits_ij)
        log_denom = torch.logsumexp(logits, dim=1)                        # [B]

        # log numer: log sum_{j in positives} exp(logits_ij)
        # 对没有正样本的 anchor，logsumexp 会是 -inf；后面用 valid_anchor 过滤
        pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
        log_numer = torch.logsumexp(pos_logits, dim=1)                    # [B]

        # loss_i = -(log_numer - log_denom)
        loss_i = -(log_numer - log_denom)                                 #_
