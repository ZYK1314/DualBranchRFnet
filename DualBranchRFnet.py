import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvNeXt
from easy_transformer import easyTrans

# 将 CNN 特征转到 1D
class CNN2Seq(nn.Module):
    def __init__(self, in_channels, out_channels, target_len):
        super().__init__()
        # 空间注意力增强
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 通道压缩
        self.channel_reduce = nn.Conv2d(in_channels, out_channels, 1)

        # 自适应池化压缩长度
        self.pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):  # x: [B,C,H,W]
        # 空间注意力增强
        att_map = self.spatial_att(x)  # [B,1,H,W]
        enhanced = x * att_map
        
        # 通道压缩
        reduced = self.channel_reduce(enhanced)  # [B,64,H,W]
    
        # 空间展平
        reduced = reduced.flatten(2)  # [B, 64, L=H*W]
        # print(reduced.shape)

        # 自适应池化压缩长度
        reduced = self.pool(reduced)  # [B, 64, target_len]
        
        return reduced
        
    
# x = torch.randn(2, 128, 32, 32)
# model = CNN2Seq(128, 64, 128)
# print(model(x).shape)

# 增强transformer特征
class TFEnhancer(nn.Module):
    def __init__(self, seq_len, in_channels, out_channels):
        super().__init__()
        # 位置编码增强
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, 1))
        
        # 时序卷积
        self.temp_conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            # padding=1
        )

    def forward(self, x):  # x: [B, L, D]
        # 添加位置信息
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # 时序特征提取
        x = x.permute(0,2,1)  # [B, D, L]
        return self.temp_conv(x)  # [B,64,3120]
    
# x = torch.randn(2, 3120, 128)
# model = TFEnhancer(3120, 128, 64)
# print(model(x).shape)

class CrossDimAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # CNN→Transformer注意力
        self.cnn2tf = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=4,
            batch_first=True
        )
        
        # Transformer→CNN注意力
        self.tf2cnn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, cnn_seq, tf_seq):
        """
        输入：
        cnn_seq: [B, L1, D] (来自CNN的序列化特征)
        tf_seq: [B, L2, D] (来自Transformer的时序特征)
        """
        # 双向交叉注意力
        # CNN作为Query，TF作为Key/Value
        cnn_enhanced, _ = self.cnn2tf(
            query=cnn_seq,
            key=tf_seq,
            value=tf_seq
        )
        
        # TF作为Query，CNN作为Key/Value
        tf_enhanced, _ = self.tf2cnn(
            query=tf_seq,
            key=cnn_seq,
            value=cnn_seq
        )
        
        return cnn_enhanced, tf_enhanced

class CMF_Fusion(nn.Module):
    def __init__(self, emb_size=256, seq_len=3000, cross_dim=64, cnn_dims=[96, 192, 384, 768], cnn_lens=[1024, 512, 256, 128]):
        super().__init__()
        # 各阶段处理模块
        self.stage_modules = nn.ModuleList([
            nn.ModuleDict({
                'cnn_adapter': CNN2Seq(in_channels=cnn_dims[i], out_channels=cross_dim, target_len=cnn_lens[i]),
                'tf_adapter': TFEnhancer(seq_len=seq_len, in_channels=emb_size, out_channels=cross_dim),
                'cross_attn': CrossDimAttention(dim=cross_dim),
                'fusion': nn.Linear(128, 64)
            }) for i in range(4)
        ])
        
        # 全局聚合
        self.aggregator = nn.Sequential(
            nn.Linear(64*4, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, cnn_features, tf_features):
        all_fusions = []
        for i in range(4):
            # 特征适配
            cnn_seq = self.stage_modules[i]['cnn_adapter'](cnn_features[i])  # [B,C,L1]
            tf_seq = self.stage_modules[i]['tf_adapter'](tf_features[i])     # [B,C,L2]
            
            cnn_seq = cnn_seq.permute(0, 2, 1)  # [B,L1,D]
            tf_seq = tf_seq.permute(0, 2, 1)  # [B,L2,D]

            # 交叉注意力
            cnn_enhanced, tf_enhanced = self.stage_modules[i]['cross_attn'](cnn_seq, tf_seq)
            
            # 拼接融合
            combined = torch.cat([cnn_enhanced.mean(1), tf_enhanced.mean(1)], dim=1)  # [B,128]
            fused = self.stage_modules[i]['fusion'](combined)  # [B,64]
            all_fusions.append(fused)
        
        # 多阶段聚合
        return self.aggregator(torch.cat(all_fusions, dim=1))  # [B,256]
    

##########################
# 完整网络整合
##########################
class DualBranchRFNet(nn.Module):
    def __init__(self, cnn_model, trans_model, emb_size, num_classes):
        super().__init__()
        # === CNN分支 ===
        self.cnn = cnn_model
        self.cnn_blocks = nn.ModuleList()
        for i in range(4):
            block = nn.Sequential(
                self.cnn.downsample_layers[i],
                self.cnn.stages[i]
            )
            self.cnn_blocks.append(block)
        
        # === Transformer分支 ===
        self.transform = trans_model
        self.trans_encoder = self.transform.emb
        self.trans_blocks = self.transform.encoder.layers
        
        # === 特征融合 ===
        self.fusion = CMF_Fusion(emb_size=emb_size, seq_len=8192, cross_dim=64, cnn_dims=[96, 192, 384, 768], cnn_lens=[1024, 512, 256, 128])
        self.dynamic_gate = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 256+128=384
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, spec, iq):
        # === CNN分支前向 ===
        # 降采样 iq
        cnn_feat = spec
        cnn_feats = []
        for block in self.cnn_blocks:
            cnn_feat = block(cnn_feat)
            cnn_feats.append(cnn_feat)
        
        # === Transformer分支前向 ===
        trans_feat = torch.nn.functional.interpolate(iq, scale_factor=1/14, mode='linear', align_corners=False)
        trans_feat = self.trans_encoder(trans_feat)
        trans_feats = []
        for block in  self.trans_blocks:
            trans_feat = block(trans_feat)
            trans_feats.append(trans_feat)
        
        # # === 多尺度特征融合 ===
        # fused = []
        # for i, (cf, fusion_layer) in enumerate(zip(cnn_feats, self.fusion)):
        #     fused.append(fusion_layer(cf, trans_feat))
        
        # # === 动态加权融合 ===
        # weights = self.dynamic_gate(torch.cat([
        #     fused[0].mean(dim=1), 
        #     fused[1].mean(dim=1)
        # ], dim=-1))  # [B, 2]
        
        # final_feat = weights[:,0].unsqueeze(-1)*fused[0] + \
        #              weights[:,1].unsqueeze(-1)*fused[1]

        final_feat = self.fusion(cnn_feats, trans_feats)
        # print('final_feat.shape: ', final_feat.shape)
        
        return self.classifier(final_feat)
    
# cnn feature
# stage0  torch.Size([8, 96, 128, 128]):
# stage1  torch.Size([8, 192, 64, 64]):
# stage2  torch.Size([8, 384, 32, 32]):
# stage3  torch.Size([8, 768, 16, 16]):

# transform feauture
# 4, 256, 3120