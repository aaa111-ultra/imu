import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1. 设置绘图风格 (全局配置)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 24,
    "axes.labelsize": 28,
    "axes.titlesize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 20,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 2.0,
    "lines.linewidth": 3.0,
    "figure.figsize": (12, 10),
    "figure.dpi": 300,
    "mathtext.fontset": "stix"
})

# 定义配色方案
COLORS = {
    'GT': '#DC143C',            # 深红色 (Ground Truth)
    'AirIO': '#1E90FF',         # 道奇蓝
    'AirIO+Denoised': '#FF8C00',# 深橙色
    'AirIO+Bias': '#228B22',    # 森林绿
    'AirIO+Denoised+Bias': '#9370DB' # 中紫色 (Ours)
}

# 自动扫描目录下的序列名称
def discover_sequences(base_dir):
    seq_candidates = set()
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return []
    for f in os.listdir(base_dir):
        if f.endswith("_compare.npz"):
            seq = f[:-len("_compare.npz")]
            seq_candidates.add(seq)
    return sorted(list(seq_candidates))

# ==========================================
def process_sequence(seq_name, base_dir):
    print(f"\n🚀 正在处理序列: {seq_name} ...")
    print(f"发现可用序列: {SEQUENCES}")

    # 自动构建文件路径
    # 兼容 Pegasus (TEST_x_compare.npz/TEST_x_compare+bias.npz) 和 AirIO (xxx_compare_AirIO.npz) 命名
    if os.path.exists(os.path.join(base_dir, f"{seq_name}_compare_AirIO.npz")):
        file_map = {
            'AirIO': f"{seq_name}_compare_AirIO.npz",
            'AirIO+Bias': f"{seq_name}_compare_AirIO+bias.npz",
            'AirIO+Denoised': f"{seq_name}_compare_AirIO+denoised.npz",
            'AirIO+Denoised+Bias': f"{seq_name}_compare_AirIO+avg.npz"
        }
    else:
        file_map = {
            'AirIO': f"{seq_name}_compare.npz",
            'AirIO+Bias': f"{seq_name}_compare+bias.npz"
        }

    # 加载数据
    data_store = {}
    gt_loaded = False
    gt_pos = None
    gt_vel = None

    for name, filename in file_map.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"   ❌ 缺失文件: {filename}")
            continue
            
        try:
            raw_data = np.load(path)
            poses = np.squeeze(raw_data['poses'])
            poses_gt = np.squeeze(raw_data['poses_gt'])
            vel = np.squeeze(raw_data['vel'])
            vel_gt = np.squeeze(raw_data['vel_gt'])
            
            # 对齐起点
            poses_aligned = poses - poses[0, :]
            
            if not gt_loaded:
                gt_pos = poses_gt - poses_gt[0, :]
                gt_vel = vel_gt
                gt_loaded = True
            
            # 计算误差
            min_len_pos = min(len(poses), len(poses_gt) - 1)
            pos_error = poses[:min_len_pos] - poses_gt[1:1+min_len_pos]
            
            min_len_vel = min(len(vel), len(gt_vel))
            vel_error = vel[:min_len_vel] - gt_vel[:min_len_vel]
            
            data_store[name] = {
                'pos': poses_aligned,
                'pos_err': pos_error,
                'vel_err': vel_error,
                'vel_est': vel
            }
        except Exception as e:
            print(f"   ❌ 读取错误 {filename}: {e}")

    if not data_store:
        return

    # 创建输出目录
    output_dir = os.path.join(base_dir, "图")
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 绘图 1: 3D 轨迹 ====================
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # 画 GT
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
             label='Ground Truth', color=COLORS['GT'], 
             linestyle='-', linewidth=3.5, alpha=0.9)
    
    for name, data in data_store.items():
        pos = data['pos']
        lw = 3.0 if name == 'AirIO+Denoised+Bias' else 2.5
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                 label=name, color=COLORS.get(name, 'blue'), 
                 linestyle='-', linewidth=lw, alpha=0.9)

    ax1.set_xlabel("Position X (m)", fontsize=28, labelpad=15)
    ax1.set_ylabel("Position Y (m)", fontsize=28, labelpad=15)
    ax1.set_zlabel("Position Z (m)", fontsize=28, labelpad=15)
    ax1.set_title(f"3D Trajectory Estimate ({seq_name})", fontsize=32, pad=25)
    ax1.tick_params(axis='both', labelsize=24)
    # ax1.legend(loc='best', frameon=True, edgecolor='black', fancybox=False, fontsize=14)
    ax1.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_name}_Trajectory_3D.png'), bbox_inches='tight', dpi=300)
    plt.close(fig1)

    # ==================== 绘图 2: 位置误差 ====================
    fig2, axes_pos = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    components = ['X', 'Y', 'Z']
    
    for i, ax in enumerate(axes_pos):
        ax.axhline(0, color=COLORS['GT'], linewidth=1.5, linestyle='-', alpha=0.7)
        
        for name, data in data_store.items():
            pos_err = data['pos_err']
            x_axis = np.arange(len(pos_err))
            
            # Ours 加粗 (3.5), 其他 (2.5)
            lw = 3.5 if name == 'AirIO+Denoised+Bias' else 2.5
            ax.plot(x_axis, pos_err[:, i], label=name, color=COLORS[name], 
                    linestyle='-', linewidth=lw, alpha=0.9)

        ax.set_ylabel(f"Error {components[i]} (m)", fontsize=28)
        ax.tick_params(direction='in', right=True, top=True, labelsize=24)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # if i == 0:
        #     ax.legend(loc='upper right', ncol=4, frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.0, 1.3))

    axes_pos[-1].set_xlabel("Time Step (frame)", fontsize=28)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(output_dir, f'{seq_name}_Position_Error.png'), bbox_inches='tight', dpi=300)
    plt.close(fig2)

    # ==================== 绘图 3: 速度对比 ====================
    fig3, axes_vel = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i, ax in enumerate(axes_vel):
        x_axis_gt = np.arange(len(gt_vel))
        ax.plot(x_axis_gt, gt_vel[:, i], label='Ground Truth', color=COLORS['GT'],
                linestyle='-', linewidth=2.5, alpha=0.9)
        
        for name, data in data_store.items():
            vel_est = data['vel_est']
            x_axis = np.arange(len(vel_est))
            ax.plot(x_axis, vel_est[:, i], label=name, color=COLORS.get(name, 'blue'), 
                    linestyle='-', linewidth=2.0, alpha=0.8)

        ax.set_ylabel(f"Velocity {components[i]} (m/s)", fontsize=28)
        ax.tick_params(direction='in', right=True, top=True, labelsize=24)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # if i == 0:
        #     ax.legend(loc='upper right', ncol=3, frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.0, 1.3))

    axes_vel[-1].set_xlabel("Time Step (frame)", fontsize=28)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(output_dir, f'{seq_name}_Velocity_Compare.png'), bbox_inches='tight', dpi=300)
    plt.close(fig3)

    # ==================== 绘图 4: 速度误差 ====================
    fig4, axes_err = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i, ax in enumerate(axes_err):
        ax.axhline(0, color=COLORS['GT'], linewidth=1.5, linestyle='-', alpha=0.7)
        
        for name, data in data_store.items():
            vel_err = data['vel_err']
            x_axis = np.arange(len(vel_err))
            
            # Ours 加粗 (2.5), 其他 (1.5)
            lw = 2.5 if name == 'AirIO+Denoised+Bias' else 1.5
            ax.plot(x_axis, vel_err[:, i], label=name, color=COLORS[name], 
                    linestyle='-', linewidth=lw, alpha=0.9)

        ax.set_ylabel(f"Error {components[i]} (m/s)", fontsize=28)
        ax.tick_params(direction='in', right=True, top=True, labelsize=24)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # if i == 0:
        #     ax.legend(loc='upper right', ncol=4, frameon=True, edgecolor='black', fontsize=12, bbox_to_anchor=(1.0, 1.3))

    axes_err[-1].set_xlabel("Time Step (frame)", fontsize=28)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(output_dir, f'{seq_name}_Velocity_Error.png'), bbox_inches='tight', dpi=300)
    plt.close(fig4)
    
    print(f"   ✅ 完成！图片已保存到 图 文件夹")

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 数据根目录
    BASE_DIR = r"/home/mengxu/lec/deffision/bias/Air-IO-main/result/Pegasus/绘图数据"
    
    # 自动扫描数据集下的所有序列
    SEQUENCES = discover_sequences(BASE_DIR)
    
    print("========================================")
    print("   开始绘制风格对比图")
    print("========================================")
    
    for seq in SEQUENCES:
        process_sequence(seq, BASE_DIR)
        
    print("\n🎉 所有序列处理完毕！")