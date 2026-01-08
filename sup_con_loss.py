class HybridStaticLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_temp=0.5):
        super(HybridStaticLoss, self).__init__()
        self.temperature = temperature
        self.lambda_temp = lambda_temp # 控制时域平稳性损失的权重

    def sup_con_loss(self, features, labels):
        """
        Supervised Contrastive Loss
        features: [N, feature_dim] (这里 N = Batch * 34)
        labels: [N] 场景标签
        """
        device = features.device
        
        # 计算相似度矩阵
        # features 已经做过 normalize，所以 dot product 等于 cosine similarity
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构造 Mask
        # labels_matrix: (i, j) 为 1 表示 i 和 j 是同一场景
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线 (自己与自己的相似度)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # 计算 Logits
        # 为了数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算分母：所有负样本 + 正样本 的 exp
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 计算分子：仅正样本部分
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        loss = - mean_log_prob_pos
        return loss.mean()

    def temporal_smoothness_loss(self, embed_t1, embed_t2):
        """
        时域平稳损失: 惩罚同一子带在极短时间内的特征变化
        Input: [B, 34, 8]
        """
        # 使用 MSE 约束距离
        return F.mse_loss(embed_t1, embed_t2)

    def forward(self, out_t1, out_t2, scene_labels):
        """
        out_t1: [B, 34, 8] 当前时刻特征
        out_t2: [B, 34, 8] 下一时刻特征 (ground truth 属于同一场景)
        scene_labels: [B]
        """
        batch_size, num_subbands, feat_dim = out_t1.shape
        
        # --- 1. Contrastive Loss (场景区分度) ---
        # 将 [B, 34, 8] 展平为 [B*34, 8]，让所有子带参与对比
        features_flat = out_t1.view(-1, feat_dim) 
        
        # 扩展标签：每个子带共享该样本的场景标签
        # [B] -> [B, 34] -> [B*34]
        labels_flat = scene_labels.unsqueeze(1).repeat(1, num_subbands).view(-1)
        
        loss_contrast = self.sup_con_loss(features_flat, labels_flat)
        
        # --- 2. Temporal Loss (静态一致性) ---
        # 直接计算 t1 和 t2 输出的距离
        loss_time = self.temporal_smoothness_loss(out_t1, out_t2)
        
        # 总损失
        total_loss = loss_contrast + self.lambda_temp * loss_time
        return total_loss, loss_contrast, loss_time
