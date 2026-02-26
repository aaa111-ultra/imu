import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from train_euroc import BiasDenoiseModel, IMUDiffusionSystem
from euroc_dataset import get_euroc_dataloaders

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_bias_compensation():
    """可视化bias补偿前后的对比"""
    
    # 加载数据
    print("Loading test data...")
    EUROC_ROOT = Path(__file__).parent.parent / "数据" / "Euroc数据集" / "EuRoC-Dataset"
    _, _, test_loader = get_euroc_dataloaders(
        root_dir=EUROC_ROOT,
        batch_size=32,
        window_size=50,
        stride=10,
        num_workers=0
    )
    
    # 加载模型
    print("Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasDenoiseModel(bias_dim=6, hidden_dim=256)
    checkpoint = torch.load(Path(__file__).parent / "checkpoints" / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    diffusion = IMUDiffusionSystem(model, T=200, device=device)
    diffusion.model.eval()
    
    # 获取一批数据
    print("Sampling predictions...")
    imu_window, bias_gt = next(iter(test_loader))
    imu_window = imu_window.to(device)
    bias_gt = bias_gt.to(device)
    
    # 模型预测
    with torch.no_grad():
        bias_pred = diffusion.sample(imu_window)
    
    # 转换为numpy
    bias_gt = bias_gt.cpu().numpy()
    bias_pred = bias_pred.cpu().numpy()
    imu_raw = imu_window.cpu().numpy()
    
    # 计算补偿后的IMU数据
    # 对于每个样本，使用窗口中间的IMU数据
    mid_idx = imu_raw.shape[1] // 2
    imu_mid = imu_raw[:, mid_idx, :]  # (batch, 6)
    
    # IMU数据格式: [w_x, w_y, w_z, a_x, a_y, a_z]
    # Bias格式: [b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z]
    imu_compensated_gt = imu_mid - bias_gt  # 使用真实bias补偿
    imu_compensated_pred = imu_mid - bias_pred  # 使用预测bias补偿
    
    # 选择前200个样本进行可视化
    num_samples = min(200, len(bias_gt))
    x = np.arange(num_samples)
    
    # 创建大图
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Bias预测对比（6个子图）
    bias_labels = ['陀螺仪X', '陀螺仪Y', '陀螺仪Z', '加速度X', '加速度Y', '加速度Z']
    units = ['rad/s', 'rad/s', 'rad/s', 'm/s²', 'm/s²', 'm/s²']
    
    for i in range(6):
        ax = plt.subplot(4, 3, i+1)
        ax.plot(x, bias_gt[:num_samples, i], 'b-', label='真实Bias', linewidth=1.5, alpha=0.7)
        ax.plot(x, bias_pred[:num_samples, i], 'r--', label='预测Bias', linewidth=1.5, alpha=0.7)
        ax.set_title(f'{bias_labels[i]} Bias对比', fontsize=12, fontweight='bold')
        ax.set_xlabel('样本序号')
        ax.set_ylabel(f'Bias ({units[i]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. 补偿前后IMU数据对比（以陀螺仪X和加速度X为例）
    # 陀螺仪X
    ax = plt.subplot(4, 3, 7)
    ax.plot(x, imu_mid[:num_samples, 0], 'gray', label='原始IMU', linewidth=1, alpha=0.5)
    ax.plot(x, imu_compensated_gt[:num_samples, 0], 'b-', label='真实Bias补偿', linewidth=1.5, alpha=0.7)
    ax.plot(x, imu_compensated_pred[:num_samples, 0], 'r--', label='预测Bias补偿', linewidth=1.5, alpha=0.7)
    ax.set_title('陀螺仪X - 补偿前后对比', fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('角速度 (rad/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 加速度X
    ax = plt.subplot(4, 3, 8)
    ax.plot(x, imu_mid[:num_samples, 3], 'gray', label='原始IMU', linewidth=1, alpha=0.5)
    ax.plot(x, imu_compensated_gt[:num_samples, 3], 'b-', label='真实Bias补偿', linewidth=1.5, alpha=0.7)
    ax.plot(x, imu_compensated_pred[:num_samples, 3], 'r--', label='预测Bias补偿', linewidth=1.5, alpha=0.7)
    ax.set_title('加速度X - 补偿前后对比', fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('加速度 (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 补偿误差分析
    error_gt = np.abs(imu_compensated_gt - imu_compensated_gt)  # 应该全为0
    error_pred = np.abs(imu_compensated_pred - imu_compensated_gt)
    
    ax = plt.subplot(4, 3, 9)
    ax.plot(x, error_pred[:num_samples, 0], 'r-', label='陀螺仪X误差', linewidth=1.5)
    ax.plot(x, error_pred[:num_samples, 3], 'b-', label='加速度X误差', linewidth=1.5)
    ax.set_title('Bias预测误差', fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('绝对误差')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 波动性对比（标准差分析）
    window = 20  # 滑动窗口计算局部标准差
    
    def rolling_std(data, window):
        """计算滑动标准差"""
        result = np.zeros(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(data[i:i+window])
        return result
    
    # 陀螺仪X波动性
    ax = plt.subplot(4, 3, 10)
    std_raw = rolling_std(imu_mid[:num_samples, 0], window)
    std_comp_gt = rolling_std(imu_compensated_gt[:num_samples, 0], window)
    std_comp_pred = rolling_std(imu_compensated_pred[:num_samples, 0], window)
    x_std = np.arange(len(std_raw))
    
    ax.plot(x_std, std_raw, 'gray', label='原始IMU波动', linewidth=1.5, alpha=0.7)
    ax.plot(x_std, std_comp_gt, 'b-', label='真实补偿后波动', linewidth=1.5, alpha=0.7)
    ax.plot(x_std, std_comp_pred, 'r--', label='预测补偿后波动', linewidth=1.5, alpha=0.7)
    ax.set_title(f'陀螺仪X局部波动性 (窗口={window})', fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('标准差 (rad/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 加速度X波动性
    ax = plt.subplot(4, 3, 11)
    std_raw = rolling_std(imu_mid[:num_samples, 3], window)
    std_comp_gt = rolling_std(imu_compensated_gt[:num_samples, 3], window)
    std_comp_pred = rolling_std(imu_compensated_pred[:num_samples, 3], window)
    
    ax.plot(x_std, std_raw, 'gray', label='原始IMU波动', linewidth=1.5, alpha=0.7)
    ax.plot(x_std, std_comp_gt, 'b-', label='真实补偿后波动', linewidth=1.5, alpha=0.7)
    ax.plot(x_std, std_comp_pred, 'r--', label='预测补偿后波动', linewidth=1.5, alpha=0.7)
    ax.set_title(f'加速度X局部波动性 (窗口={window})', fontsize=12, fontweight='bold')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('标准差 (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 统计对比
    ax = plt.subplot(4, 3, 12)
    ax.axis('off')
    
    # 计算统计指标
    mae_gyro = np.mean(np.abs(bias_pred[:, :3] - bias_gt[:, :3]), axis=0)
    mae_accel = np.mean(np.abs(bias_pred[:, 3:] - bias_gt[:, 3:]), axis=0)
    
    std_raw_gyro = np.std(imu_mid[:num_samples, :3], axis=0)
    std_comp_gyro = np.std(imu_compensated_pred[:num_samples, :3], axis=0)
    
    std_raw_accel = np.std(imu_mid[:num_samples, 3:], axis=0)
    std_comp_accel = np.std(imu_compensated_pred[:num_samples, 3:], axis=0)
    
    stats_text = "=== 统计摘要 ===\n\n"
    stats_text += "Bias预测MAE:\n"
    stats_text += f"  陀螺仪: {mae_gyro.mean():.6f} rad/s\n"
    stats_text += f"  加速度: {mae_accel.mean():.6f} m/s²\n\n"
    stats_text += "数据标准差对比:\n"
    stats_text += f"  原始陀螺仪: {std_raw_gyro.mean():.4f}\n"
    stats_text += f"  补偿陀螺仪: {std_comp_gyro.mean():.4f}\n"
    stats_text += f"  波动降低: {(1-std_comp_gyro.mean()/std_raw_gyro.mean())*100:.1f}%\n\n"
    stats_text += f"  原始加速度: {std_raw_accel.mean():.4f}\n"
    stats_text += f"  补偿加速度: {std_comp_accel.mean():.4f}\n"
    stats_text += f"  波动降低: {(1-std_comp_accel.mean()/std_raw_accel.mean())*100:.1f}%\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(__file__).parent / "checkpoints" / "bias_compensation_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    visualize_bias_compensation()
