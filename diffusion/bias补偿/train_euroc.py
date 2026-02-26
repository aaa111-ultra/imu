import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt

# 导入原始模型定义
import sys
sys.path.append(str(Path(__file__).parent))
from euroc_dataset import get_euroc_dataloaders

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


def train_one_epoch(diffusion, dataloader, optimizer, device):
    """训练一个 epoch"""
    diffusion.model.train()
    total_loss = 0.0
    num_batches = 0
    
    for imu_window, bias_gt in dataloader:
        imu_window = imu_window.to(device)
        bias_gt = bias_gt.to(device)
        
        optimizer.zero_grad()
        loss = diffusion.forward_diffusion_loss(bias_gt, imu_window)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(diffusion, dataloader, device):
    """验证模型"""
    diffusion.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for imu_window, bias_gt in dataloader:
        imu_window = imu_window.to(device)
        bias_gt = bias_gt.to(device)
        
        loss = diffusion.forward_diffusion_loss(bias_gt, imu_window)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(diffusion, dataloader, device):
    """
    评估模型预测性能
    计算预测 Bias 与真实 Bias 的误差
    """
    diffusion.model.eval()
    
    all_pred_bias = []
    all_gt_bias = []
    
    print("Evaluating model (sampling)...")
    for i, (imu_window, bias_gt) in enumerate(dataloader):
        imu_window = imu_window.to(device)
        bias_gt = bias_gt.to(device)
        
        # 通过扩散模型采样预测 Bias
        pred_bias = diffusion.sample(imu_window)
        
        all_pred_bias.append(pred_bias.cpu().numpy())
        all_gt_bias.append(bias_gt.cpu().numpy())
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(dataloader)} batches")
    
    all_pred_bias = np.concatenate(all_pred_bias, axis=0)
    all_gt_bias = np.concatenate(all_gt_bias, axis=0)
    
    # 计算误差指标
    mae = np.mean(np.abs(all_pred_bias - all_gt_bias), axis=0)
    rmse = np.sqrt(np.mean((all_pred_bias - all_gt_bias)**2, axis=0))
    
    print("\n=== Evaluation Results ===")
    print("Mean Absolute Error (MAE) for each bias component:")
    print(f"  Gyro X: {mae[0]:.6f} rad/s")
    print(f"  Gyro Y: {mae[1]:.6f} rad/s")
    print(f"  Gyro Z: {mae[2]:.6f} rad/s")
    print(f"  Accel X: {mae[3]:.6f} m/s²")
    print(f"  Accel Y: {mae[4]:.6f} m/s²")
    print(f"  Accel Z: {mae[5]:.6f} m/s²")
    
    print("\nRoot Mean Square Error (RMSE) for each bias component:")
    print(f"  Gyro X: {rmse[0]:.6f} rad/s")
    print(f"  Gyro Y: {rmse[1]:.6f} rad/s")
    print(f"  Gyro Z: {rmse[2]:.6f} rad/s")
    print(f"  Accel X: {rmse[3]:.6f} m/s²")
    print(f"  Accel Y: {rmse[4]:.6f} m/s²")
    print(f"  Accel Z: {rmse[5]:.6f} m/s²")
    
    return mae, rmse, all_pred_bias, all_gt_bias


def plot_training_history(train_losses, val_losses, save_path):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def main():
    # ==================== 配置参数 ====================
    # 数据集路径（相对路径）
    EUROC_ROOT = Path(__file__).parent.parent / "数据" / "Euroc数据集" / "EuRoC-Dataset"
    
    # 训练参数
    BATCH_SIZE = 32
    WINDOW_SIZE = 50        # IMU 滑动窗口大小
    STRIDE = 10             # 滑动窗口步长
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # 模型参数
    HIDDEN_DIM = 256
    DIFFUSION_STEPS = 200   # 扩散步数（降低以加快训练）
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # ==================== 加载数据 ====================
    print("Loading EuRoC dataset...")
    train_loader, val_loader, test_loader = get_euroc_dataloaders(
        root_dir=EUROC_ROOT,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_workers=0  # Windows 下建议设为 0
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # ==================== 创建模型 ====================
    print("Creating model...")
    model = BiasDenoiseModel(bias_dim=6, hidden_dim=HIDDEN_DIM)
    diffusion = IMUDiffusionSystem(model, T=DIFFUSION_STEPS, device=device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters\n")
    
    # ==================== 训练设置 ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 创建保存目录
    save_dir = Path(__file__).parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    
    # ==================== 训练循环 ====================
    print("=" * 60)
    print("Start Training...")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_one_epoch(diffusion, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(diffusion, val_loader, device)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            print(f"  --> Best model saved (Val Loss: {val_loss:.6f})")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # ==================== 保存训练历史 ====================
    plot_training_history(train_losses, val_losses, save_dir / 'training_history.png')
    
    # ==================== 测试评估 ====================
    print("\n" + "=" * 60)
    print("Loading best model for testing...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} loaded\n")
    
    print("=" * 60)
    print("Testing on test set...")
    print("=" * 60)
    mae, rmse, pred_bias, gt_bias = evaluate(diffusion, test_loader, device)
    
    # 保存测试结果
    np.savez(save_dir / 'test_results.npz', 
             pred_bias=pred_bias, 
             gt_bias=gt_bias,
             mae=mae,
             rmse=rmse)
    print(f"\nTest results saved to {save_dir / 'test_results.npz'}")
    
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
