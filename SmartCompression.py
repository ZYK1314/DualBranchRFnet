class SmartCompression(nn.Module):
    def __init__(self, d_model, L):
        super().__init__()
        # 动态压缩门控
        self.gate = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, L*d_model),
            nn.Sigmoid(). 
        )
        
        # 内容感知投影
        self.proj = nn.Linear(d_model, d_model)
        
        # 初始化参数
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, fused, L):
        """
        输入: 
            fused - [B, KL, D]
            L - 目标序列长度
        输出: [B, L, D]
        """
        B, KL, D = fused.shape
        
        # 动态生成压缩门控
        gate = self.gate(fused.mean(dim=1))  # [B, L*D]
        gate = gate.view(B, L, D)  # [B, L, D]
        
        # 内容投影
        content = self.proj(fused)  # [B, KL, D]
        
        # 门控压缩
        compressed = torch.einsum('bld,bkd->blk', gate, content)  # [B, L, KL]
        compressed = torch.softmax(compressed, dim=-1)  # 归一化
        output = torch.einsum('blk,bkd->bld', compressed, content)  # [B, L, D]
        
        return output
