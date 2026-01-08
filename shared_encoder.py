import torch
import torch.nn as nn
import torch.nn.functional as F

class SubbandSharedEncoder(nn.Module):
    def __init__(self, feature_dim=8):
        super(SubbandSharedEncoder, self).__init__()
        
        # 输入: [Batch, 2 (Real/Imag), 32, 48]
        # 使用简单的 ResNet Block 堆叠，不考虑复杂度
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # [16, 24]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # [8, 12]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # [256, 1, 1]
            nn.Flatten()
        )
        
        # 投影头 (Projection Head) - 常用于对比学习，提取特征后再映射
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        # x shape: [B, 34, 32, 48] (Complex) -> 需要拆分实虚部
        # 假设输入已经是 [B, 34, 32, 48] 的复数 tensor 或者 [B, 34, 2, 32, 48]
        
        # 如果输入是复数 Tensor，拆分为 Real/Imag 通道
        if x.is_complex():
            x = torch.stack([x.real, x.imag], dim=2) # [B, 34, 2, 32, 48]
            
        b, s, c, h, w = x.shape
        
        # 1. Merge Batch and Subband dimensions: [B*34, 2, 32, 48]
        x_reshaped = x.view(b * s, c, h, w)
        
        # 2. Encode
        features = self.encoder(x_reshaped) # [B*34, 256]
        embeddings = self.head(features)    # [B*34, 8]
        
        # 3. L2 Normalize (关键：对比学习需要在超球面上进行)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 4. Reshape back: [B, 34, 8]
        return embeddings.view(b, s, -1)
