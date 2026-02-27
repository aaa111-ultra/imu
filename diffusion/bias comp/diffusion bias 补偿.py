import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    """
    为扩散模型的时间步 t 生成正弦位置编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class IMUFeatureEncoder(nn.Module):
    """
    条件提取器：处理输入的 IMU 滑动窗口数据 (Batch, Window_Size, 6)
    将其压缩为一个特征向量 (Batch, Hidden_Dim)
    """
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        # 使用 1D 卷积处理时间序列特征
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # 将时间维度池化，提取全局特征
            nn.Flatten()
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        # x shape: (Batch, Window_Size, Features) -> (Batch, Features, Window_Size)
        x = x.transpose(1, 2)
        feat = self.net(x)
        return self.fc(feat)

class BiasDenoiseModel(nn.Module):
    """
    核心去噪网络：预测噪声
    输入: 
        x_t: 当前带噪声的 Bias (Batch, 6)
        t: 时间步
        condition: IMU 原始数据的特征
    """
    def __init__(self, bias_dim=6, hidden_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 条件编码 (IMU Raw Data)
        self.imu_encoder = IMUFeatureEncoder(input_dim=6, hidden_dim=hidden_dim)
        
        # 输入 Bias 的映射
        self.bias_embed = nn.Linear(bias_dim, hidden_dim)
        
        # 主干网络 (简单的 ResNet Block 思想)
        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), # 拼接后输入
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bias_dim) # 输出预测的噪声，维度与 Bias 一致
        )

    def forward(self, x_t, t, raw_imu_window):
        # 1. 处理时间嵌入
        time_embed = self.time_mlp(t) # (Batch, Hidden)
        
        # 2. 处理条件 (原始 IMU)
        cond_embed = self.imu_encoder(raw_imu_window) # (Batch, Hidden)
        
        # 3. 处理带噪输入
        x_embed = self.bias_embed(x_t) # (Batch, Hidden)
        
        # 4. 特征融合 (这里采用简单的相加或拼接，这里演示拼接)
        # 融合 Condition 和 Time
        global_cond = time_embed + cond_embed 
        
        # 拼接 Bias 特征和全局条件
        combined = torch.cat([x_embed, global_cond], dim=-1)
        
        # 5. 预测噪声
        noise_pred = self.main_net(combined)
        return noise_pred

class IMUDiffusionSystem:
    """
    管理扩散过程（加噪与去噪采样）
    """
    def __init__(self, model, T=1000, device='cuda'):
        self.model = model.to(device)
        self.T = T
        self.device = device
        
        # 定义 Beta Schedule (线性)
        self.betas = torch.linspace(0.0001, 0.02, T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_loss(self, x_0, raw_imu_window):
        """
        训练过程：随机采样 t，加噪，预测噪声，计算 Loss
        x_0: 真实的 Bias (Ground Truth)
        raw_imu_window: 对应的原始 IMU 窗口数据
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        
        # 生成高斯噪声
        noise = torch.randn_like(x_0)
        
        # 计算 x_t (根据公式: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        # 模型预测噪声
        noise_pred = self.model(x_t, t, raw_imu_window)
        
        # 使用 MSE Loss
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, raw_imu_window):
        """
        推理过程：给定 IMU 数据，从纯噪声中恢复出 Bias
        """
        self.model.eval()
        batch_size = raw_imu_window.shape[0]
        bias_dim = 6 # [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        
        # 从标准正态分布开始
        x = torch.randn((batch_size, bias_dim), device=self.device)
        
        # 逐步去噪: T-1 -> 0
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x, t, raw_imu_window)
            
            # 采样步骤 (DDPM update rule)
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            
            mean = sqrt_recip_alpha_t * (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(x)
                std = torch.sqrt(self.posterior_variance[i])
                x = mean + std * noise
            else:
                x = mean
                
        self.model.train()
        return x

# ================= 模拟使用示例 =================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 实例化模型
    # 假设输入是长度为 100 的 IMU 窗口 (Batch, 100, 6)
    model = BiasDenoiseModel(bias_dim=6, hidden_dim=128)
    diffusion = IMUDiffusionSystem(model, T=200, device=device) # T 可以设小一点加快速度
    
    # 2. 模拟训练数据
    batch_size = 16
    window_size = 50
    
    # 模拟输入: 原始 IMU 数据 (acc + gyro)
    dummy_imu_window = torch.randn(batch_size, window_size, 6).to(device)
    # 模拟标签: 对应的真实 Bias (通常需要通过高精度设备或优化算法获得真值)
    dummy_gt_bias = torch.randn(batch_size, 6).to(device) * 0.1 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. 训练循环演示
    print("Start Training...")
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        loss = diffusion.forward_diffusion_loss(dummy_gt_bias, dummy_imu_window)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
    # 4. 推理 (Inference) 演示
    # 给定一段新的 IMU 数据，预测其 Bias
    print("\nStart Inference...")
    predicted_bias = diffusion.sample(dummy_imu_window)
    print("Predicted Bias shape:", predicted_bias.shape)
    print("Sample Prediction:", predicted_bias[0])