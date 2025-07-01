import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from model import DropPath

# class ComplexTemporalEncoder(nn.Module):
#     def __init__(self, embed_dim=128, max_seq_len=40960):
#         super().__init__()
#         # 复数特征提取
#         self.conv_real1 = nn.Conv1d(1, embed_dim//8, kernel_size=7, stride=3, padding=3)
#         self.conv_real2 = nn.Conv1d(embed_dim//8, embed_dim//4, kernel_size=5, stride=2, padding=2)
#         self.conv_imag1 = nn.Conv1d(1, embed_dim//8, kernel_size=7, stride=3, padding=3)
#         self.conv_imag2 = nn.Conv1d(embed_dim//8, embed_dim//4, kernel_size=5, stride=2, padding=2)
#         self.conv_ph1 = nn.Conv1d(1, embed_dim//8, kernel_size=7, stride=3, padding=3)
#         self.conv_ph2 = nn.Conv1d(embed_dim//8, embed_dim//4, kernel_size=5, stride=2, padding=2)
#         self.conv_mag1 = nn.Conv1d(1, embed_dim//8, kernel_size=7, stride=3, padding=3)
#         self.conv_mag2 = nn.Conv1d(embed_dim//8, embed_dim//4, kernel_size=5, stride=2, padding=2)
        
#         self.conv_reduce = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=3)
        
#         # 位置编码
#         self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

#         # 自适应池化
#         self.pool = nn.AdaptiveAvgPool1d(output_size=1024)
        
#         # 归一化与激活
#         self.norm = nn.LayerNorm(embed_dim)
#         self.act = nn.GELU()

#     def forward(self, x):
#         # 分解复数
#         x_real = x[:,0:1,:]  # [B, 1, L]
#         x_imag = x[:,1:2,:]  # [B, 1, L]

#         # 求取相位和幅度
#         x_phase = torch.atan2(x_imag, x_real)  # 相位特征
#         x_magnitude = torch.sqrt(x_real**2 + x_imag**2)  # 幅度特征
#         # print('x_phase.shape: ', x_phase.shape)
#         # print('x_magnitude.shape: ', x_magnitude.shape)

#         # 独立卷积
#         x_real = self.conv_real2(self.conv_real1(x_real)).permute(0, 2, 1)  # [B, L, D/2]
#         x_imag = self.conv_imag2(self.conv_imag1(x_imag)).permute(0, 2, 1)  # [B, L, D/2]
#         x_phase = self.conv_real2(self.conv_real1(x_phase)).permute(0, 2, 1)  # [B, L, D/2]
#         x_magnitude = self.conv_imag2(self.conv_imag1(x_magnitude)).permute(0, 2, 1)  # [B, L, D/2]

#         # 复数融合
#         # x = torch.cat([x_real, x_imag], dim=-1)  # [B, L, D]
#         x = torch.cat([x_real, x_imag, x_phase, x_magnitude], dim=-1)  # [B, L, D]
#         # print('cat x.shape: ', x.shape)
        
#         # 使用1D卷积层降维
#         x = self.conv_reduce(x.permute(0, 2, 1)).permute(0, 2, 1) # [B, L//2, D]
#         # print('conv_reduce x.shape: ', x.shape)
        
#         # 使用线性层降维
#         # x = x.permute(0, 2, 1)  # [B, D, L]
#         # x = self.linear_reduce(x)  # [B, D, L//2]
#         # x = x.permute(0, 2, 1)  # [B, L//2, D]
#         # print('linear_reduce x.shape: ', x.shape)

#         # 使用自适应池化降维
#         # x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 1024, D]
#         # print('pool x.shape: ', x.shape)
        
#         # 添加位置编码
#         x = x + self.pos_embed[:, :x.size(1), :]
        
#         return self.norm(self.act(x))

class ConvBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch//4, kernel_size=7, stride=3, padding=3)
        self.conv2 = nn.Conv1d(out_ch//4, out_ch//2, kernel_size=5, stride=2, padding=2)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class ComplexTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=128, max_seq_len=8192):
        super().__init__()
        
        # 分支定义
        self.real_branch = ConvBranch(1, embed_dim)
        self.imag_branch = ConvBranch(1, embed_dim)
        self.phase_branch = ConvBranch(1, embed_dim)
        self.mag_branch = ConvBranch(1, embed_dim)

        # # 动态下采样计算
        # self.seq_reduce_factor = (3*2)**2  # (stride1 * stride2)^2
        # self.final_seq_len = max_seq_len // self.seq_reduce_factor
        
        # 降维层
        # self.conv_reduce = nn.Sequential(
        #     nn.Conv1d(embed_dim * 4, embed_dim * 2, 3, stride=2, padding=1, dilation=2),
        #     nn.BatchNorm1d(embed_dim * 2),
        #     nn.GELU(),
        #     nn.Conv1d(embed_dim * 2, embed_dim, 3, stride=2, padding=2, dilation=3)
        # )
        self.conv_reduce = nn.Sequential(
            # nn.BatchNorm1d(embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=3)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # 归一化与激活
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 复数解析
        x_complex = torch.view_as_complex(x.permute(0,2,1).contiguous())
        x_mag = torch.abs(x_complex).unsqueeze(1)
        x_phase = torch.angle(x_complex).unsqueeze(1)
        
        # 各分支处理
        real_feat = self.real_branch(x[:,0:1])  # [B, D/4, L']
        imag_feat = self.imag_branch(x[:,1:2])
        # phase_feat = self.phase_branch(x_phase)
        # mag_feat = self.mag_branch(x_mag)
        # print('real_feat.shape: ', real_feat.shape)
        # 特征融合
        # x = torch.cat([real_feat, imag_feat, phase_feat, mag_feat], dim=1)
        x = torch.cat([real_feat, imag_feat], dim=1)
        # print('x.shape: ', x.shape)
        x = self.conv_reduce(x).permute(0,2,1)
        # print('x.shape: ', x.shape)
        # 位置编码
        x = x + self.pos_embed[:, :x.size(1)]

        return self.norm(self.act(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size=256, nhead=4, dim_feedforward=512, drop_rate=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emb_size, nhead, batch_first=True)
        self.linear1 = nn.Linear(emb_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, emb_size)
        # 增加深度可分离卷积
        self.depthwise_conv = nn.Conv1d(
            emb_size, emb_size, 
            kernel_size=3, 
            groups=emb_size, 
            padding=1
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size) 
        self.dropout = nn.Dropout(drop_rate)
        self.activation = nn.GELU()

    def forward(self, src):
        # 自注意力
        shortcut = src
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        # FFN
        ff_output = self.linear2(self.activation(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff_output))
        # src = shortcut + self.norm3(self.depthwise_conv(src.permute(0, 2, 1)).permute(0, 2, 1))
        src = shortcut + self.dropout(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=4, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # 输出形状: [B, seq_len, d_model]

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),  # TODO 这里使用所有嵌入的平均值作为分类输入，而不是使用CLStoken
            nn.LayerNorm(emb_size), 
            nn.ReLU(True),  # TODO 决定是否保留
            nn.Linear(emb_size, n_classes))

class AdaptiveFraming(nn.Module):
    def __init__(self, max_frame_len=4096, min_frame_len=256, overlap_ratio=0.50):
        super().__init__()
        # 可学习帧长参数（动态适应信号特性）
        self.frame_len = nn.Parameter(torch.tensor(max_frame_len, dtype=torch.float))
        self.min_len = min_frame_len
        self.overlap = overlap_ratio
        
    def forward(self, x):
        """
        输入: [B, 2, L]
        输出: [B, N_frames, 2, T] + 有效帧数掩码
        """
        B, C, L = x.shape
        device = x.device
        
        # 动态计算帧长（约束在[min_len, max_len]）
        frame_len = torch.clamp(self.frame_len, self.min_len, L//2).int().item()
        step = int(frame_len * (1 - self.overlap))
        
        # 计算最大可能帧数
        n_frames = (L - frame_len) // step + 1
        if n_frames <= 0:  # 短信号处理
            pad = frame_len - L
            x = F.pad(x, (0, pad))
            n_frames = 1
            step = 1
        
        # 滑动窗口分帧 (PyTorch unfold)
        frames = x.unfold(-1, frame_len, step)  # [B, C, n_frames, frame_len]
        frames = frames.permute(0, 2, 1, 3)    # [B, n_frames, C, frame_len]
        
        # 生成有效帧掩码（应对边缘填充）
        mask = torch.ones(B, n_frames, 1, device=device)
        if L < frame_len:
            mask[:, 0, 0] = 0  # 标记填充帧
        return frames, mask

class HierarchicalFusion(nn.Module):
    def __init__(self, group_sequence=[64,32,16,8], d_model=512, l_feat=1024):
        super().__init__()
        self.group_sequence = group_sequence
        self.l_feat = l_feat

        self.fusion_blocks = nn.ModuleList([
             nn.Sequential(
                nn.Conv1d(in_channels=l_feat * g, out_channels=l_feat, kernel_size=1),
                # nn.Dropout(0.05),  # 添加dropout98
                # nn.LayerNorm(l_feat)
            ) for g in group_sequence
        ])
        
        self.proj = nn.Linear(l_feat, l_feat)
        # 初始化参数
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, features):
        """
        输入: N个[B,L,D]特征组成的列表
        输出: [B,m*L,D] 最终拼接结果
        """
        current_features = features
        
        for g, fusion_block in zip(self.group_sequence, self.fusion_blocks):
            # 动态分组
            groups = self.chunk_list(current_features, chunk_size=g)
            
            # 处理每个分组
            new_features = []
            for group in groups:
                if len(group) == 1:
                    new_features.append(group[0])
                    continue
                
                # 拼接特征 [B,K*L,D]
                stacked = torch.cat(group, dim=-1)
                # print('stacked: ', stacked.shape)
                if stacked.shape[-1] != g * self.l_feat:
                    continue

                # 注意力融合
                B, C, KL = stacked.shape
                # print('stacked: ', stacked.shape)

                # 加权融合  TODO: 优化，可以试用 SmartCompression
                
                fused = fusion_block(stacked.permute(0, 2, 1)).permute(0, 2, 1)  # [B,L,D]
                # print('fused: ', fused.shape)
                fused = self.proj(fused)  # [B,L,D]
                new_features.append(fused)  # 恢复L维度
            
            current_features = new_features
        
        # 最终拼接
        return torch.cat(current_features, dim=-1)

    @staticmethod
    def chunk_list(lst, chunk_size):
        """智能分组：动态调整最后一组大小"""
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


class easyTrans(nn.Module):
    def __init__(self, num_layers, emb_size, nhead, num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.nhead = nhead

        self.framer = AdaptiveFraming(max_frame_len=4096, min_frame_len=256, overlap_ratio=0.50)
        # self.frame_encoder = FrameEncoder()
        # self.temp_encoder = TemporalParallelEncoder()
        self.py_pool = PyramidPool(pool_sizes=[32,16,8])
        self.hier_fusion = HierarchicalFusion(group_sequence=[8, 4, 2, 1], d_model=emb_size, l_feat=4096)
        self.emb = ComplexTemporalEncoder(embed_dim=self.emb_size)
        self.encoder = TransformerEncoder(num_layers=self.num_layers, emb_size=self.emb_size, nhead=self.nhead, dim_feedforward=512)
        self.head = ClassificationHead(emb_size=self.emb_size, n_classes=self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, iq):
        x = iq
        # # 随机剪裁 length 为 10240 的窗口
        # batch_size, channels, length = iq.shape

        # # 生成随机起始点 [B]
        # start = torch.randint(0, length - 102400 + 1, (batch_size,), device=iq.device)

        # # 生成索引矩阵 [B, 10240]
        # indices = start.unsqueeze(1) + torch.arange(102400, device=iq.device)

        # # 使用 gather 进行批量索引
        # iq = torch.gather(
        #     iq, 
        #     dim=2, 
        #     index=indices.unsqueeze(1).expand(-1, channels, -1)
        # )

        # iq = torch.nn.functional.interpolate(iq, scale_factor=0.2, mode='linear', align_corners=False)
        # print('iq.shape: ', iq.shape)
        # frames, mask = self.framer(iq)  # [B,N,2,T]
        # B, N, C, T = frames.shape
        # print('frames.shape: ', frames.shape)
        
        # 沿N维度拆分为N个独立张量
        # frame_splits = frames.unbind(dim=1)  # 生成包含N个 [B,2,T] 张量的元组
        # print('frame_splits.shape: ', len(frame_splits), frame_splits[0].shape)
        # # 初始化结果容器
        # processed_outputs = []

        # # 逐帧处理
        # for frame in frame_splits[:N//4]:
        #     # 输入单帧数据到网络
        #     # x = self.emb(frame)  # [B,L,D]
        #     processed_outputs.append(x)
        
        # 将N个输出张量拼接为 [B,N,D]
        # print('processed_outputs.shape: ', processed_outputs[0].shape, len(processed_outputs))
        # x = self.hier_fusion(frame_splits[:N//3])
        # x = torch.nn.functional.interpolate(x, scale_factor=1/28, mode='linear', align_corners=False)  # 2 MHz Test accuracy: 0.6642335653305054 
        x = torch.nn.functional.interpolate(x, scale_factor=1/14, mode='linear', align_corners=False)  # 4 MHz Test accuracy: 0.6861313581466675 epoch = 100 未收敛
        # 4 MHz Test accuracy: 0.6569343209266663 Test weighted accuracy: 0.35260000824928284 epoch = 200
        # 4 MHz Test accuracy: 0.8759124279022217 Test weighted accuracy: 0.7842025756835938  emb_size=256  使用1D卷积层降维
        # 4 MHz Test accuracy: 0.8394160866737366 Test weighted accuracy: 0.636810302734375   epoch = 200, phase+mag
        print('downsample x.shape: ', x.shape)
        x = self.emb(x)  # [B,L,D]
        print('emb x.shape: ', x.shape)
        x = self.encoder(x)
        print('afer encoder x.shape: ', x.shape)
        x = self.dropout(x)
        x = self.head(x)
        # print('x.shape: ', x.shape)
        return x

# def test():
#     # 验证网络前向传播
#     net = easyTrans(4, 128, 4, 10)
#     spec = torch.randn(2, 224, 224)  # 时频谱图输入
#     iq = torch.view_as_complex(torch.randn(2, 1024, 2))  # 复数IQ信号
#     output = net(spec, iq)
#     print(f"Output shape: {output.shape}")  # 应输出[2,10]

# if __name__ == "__main__":
#     test()

# from torchinfo import summary

# class ComplexWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         # 将实部和虚部合并为复数张量
#         x = torch.view_as_complex(x)
#         return self.model(x)

# model = easyTrans(4, 128, 4, num_classes=7)
# # # wrapped_model = ComplexWrapper(model)
# # # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # # model.to(device)
# summary(model, input_size=[(2, 1024000)], batch_dim=4, device='cpu')

class FrameEncoder(nn.Module):
    def __init__(self, in_ch=2, out_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(  # 使用2D卷积同时处理帧和时序
            in_channels=in_ch,
            out_channels=64,
            kernel_size=(1, 7),  # 帧维kernel_size=1（不跨帧）
            padding=(0, 3)
        )
        self.pool = nn.MaxPool2d((1, 2))
        
    def forward(self, x):
        """
        输入x: [B, N_frames, 2, T]
        输出: [B, N_frames, D]
        """
        B, N, C, T = x.shape
        # 重组维度: [B, N, C, T] → [B, C, N, T]
        x = x.permute(0, 2, 1, 3)
        
        # 2D卷积处理
        x = F.gelu(self.conv(x))  # [B, 64, N, T]
        x = self.pool(x)          # [B, 64, N, T//2]
        # 全局平均池化
        return x.mean(dim=-1)     # [B, 64, N]

class TemporalParallelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 时间卷积核（同时处理所有帧）
        self.temp_conv = nn.Conv1d(
            in_channels=64, 
            out_channels=256,
            kernel_size=3,
            groups=16  # 分组卷积减少参数量
        )
    
    def forward(self, x):
        # x: [B, 64, N]
        return self.temp_conv(x)  # [B, 256, N]

class PyramidPool(nn.Module):
    def __init__(self, pool_sizes):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(s) for s in pool_sizes
        ])
        
    def forward(self, x):
        # x: [B, N*L, D]
        features = []
        for pool in self.pools:
            pooled = pool(x.transpose(1,2)).transpose(1,2)
            features.append(pooled)
        return torch.cat(features, dim=1)  # 自动适配new_L