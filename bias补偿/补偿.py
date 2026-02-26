import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from train_euroc import BiasDenoiseModel, IMUDiffusionSystem


def load_model_and_diffusion(model_path, device, hidden_dim=256, T=200):
    """
    加载训练好的模型与扩散系统
    """
    model = BiasDenoiseModel(bias_dim=6, hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion = IMUDiffusionSystem(model, T=T, device=device)
    diffusion.model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Model loaded from epoch {epoch}\n")
    return diffusion


def predict_bias_series(imu_values, diffusion, window_size=50, device='cuda', batch_size=32):
    """
    使用滑动窗口预测每个时刻的 bias

    Args:
        imu_values: (N, 6) numpy array, 顺序为 [gyro_xyz, acc_xyz]
    Returns:
        predicted_bias: (N, 6) numpy array
    """
    num_samples = len(imu_values)
    predicted_bias = np.zeros_like(imu_values)

    if num_samples == 0:
        return predicted_bias

    if num_samples < window_size:
        window = torch.from_numpy(imu_values.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            bias = diffusion.sample(window).cpu().numpy()[0]
        predicted_bias[:] = bias
        return predicted_bias

    half_window = window_size // 2

    # 前半个窗口使用第一个窗口的预测
    first_window = torch.from_numpy(imu_values[:window_size].astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        first_bias = diffusion.sample(first_window).cpu().numpy()[0]

    for i in range(half_window):
        predicted_bias[i] = first_bias

    # 中间部分：批量预测
    windows = []
    indices = []
    for i in range(half_window, num_samples - half_window):
        start_idx = i - half_window
        end_idx = i + half_window
        if end_idx <= num_samples:
            windows.append(imu_values[start_idx:end_idx])
            indices.append(i)

        if len(windows) == batch_size or i == num_samples - half_window - 1:
            if windows:
                batch_windows = torch.from_numpy(np.array(windows).astype(np.float32)).to(device)
                with torch.no_grad():
                    batch_bias = diffusion.sample(batch_windows).cpu().numpy()

                for j, idx in enumerate(indices):
                    predicted_bias[idx] = batch_bias[j]

                windows = []
                indices = []

    # 后半个窗口使用最后一个窗口的预测
    last_window = torch.from_numpy(imu_values[-window_size:].astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        last_bias = diffusion.sample(last_window).cpu().numpy()[0]

    for i in range(num_samples - half_window, num_samples):
        predicted_bias[i] = last_bias

    return predicted_bias

def compensate_imu_sequence(seq_path, model, diffusion, window_size=50, device='cuda'):
    """
    对单个序列的IMU数据进行bias补偿
    
    Args:
        seq_path: 序列路径 (包含mav0文件夹)
        model: 训练好的bias预测模型
        diffusion: 扩散系统
        window_size: 滑动窗口大小
        device: 计算设备
    
    Returns:
        compensated_imu_data: 补偿后的IMU数据DataFrame
    """
    # 读取IMU数据
    imu_file = seq_path / 'mav0' / 'imu0' / 'data.csv'
    imu_data = pd.read_csv(imu_file)
    
    # 提取列名（去除单位）
    imu_data.columns = [col.split(' [')[0].strip() for col in imu_data.columns]
    
    # 提取IMU原始值
    imu_values = imu_data[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z', 
                           'a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].values
    timestamps = imu_data['#timestamp'].values
    
    # 预测 bias 并补偿
    print(f"  Processing {len(imu_values)} IMU samples...")
    predicted_bias = predict_bias_series(
        imu_values, diffusion, window_size=window_size, device=device, batch_size=32
    )
    compensated_values = imu_values - predicted_bias
    
    # 创建补偿后的DataFrame
    compensated_df = imu_data.copy()
    compensated_df['w_RS_S_x'] = compensated_values[:, 0]
    compensated_df['w_RS_S_y'] = compensated_values[:, 1]
    compensated_df['w_RS_S_z'] = compensated_values[:, 2]
    compensated_df['a_RS_S_x'] = compensated_values[:, 3]
    compensated_df['a_RS_S_y'] = compensated_values[:, 4]
    compensated_df['a_RS_S_z'] = compensated_values[:, 5]
    
    # 添加预测的bias信息（可选，用于分析）
    bias_df = pd.DataFrame({
        '#timestamp': timestamps,
        'b_w_x_pred': predicted_bias[:, 0],
        'b_w_y_pred': predicted_bias[:, 1],
        'b_w_z_pred': predicted_bias[:, 2],
        'b_a_x_pred': predicted_bias[:, 3],
        'b_a_y_pred': predicted_bias[:, 4],
        'b_a_z_pred': predicted_bias[:, 5]
    })
    
    return compensated_df, bias_df


def copy_directory_structure(src, dst, exclude_files=None):
    """
    递归复制目录结构，排除特定文件
    
    Args:
        src: 源目录
        dst: 目标目录
        exclude_files: 要排除的文件列表
    """
    if exclude_files is None:
        exclude_files = []
    
    dst.mkdir(parents=True, exist_ok=True)
    
    for item in src.iterdir():
        src_path = src / item.name
        dst_path = dst / item.name
        
        # 跳过排除的文件
        if any(exclude in str(src_path) for exclude in exclude_files):
            continue
        
        if src_path.is_dir():
            copy_directory_structure(src_path, dst_path, exclude_files)
        else:
            shutil.copy2(src_path, dst_path)


def generate_euroc_compensated_dataset(root_dir, output_dir, model_path, device='cuda'):
    """
    生成整个EuRoC数据集的bias补偿版本（保持完整文件夹结构）
    
    Args:
        root_dir: EuRoC原始数据集根目录
        output_dir: 输出目录
        model_path: 训练好的模型路径
        device: 计算设备
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 首先复制数据集划分文件
    print("Copying dataset split files...")
    for split_file in ['train_list.txt', 'val_list.txt', 'test_list.txt']:
        src_file = root_dir / split_file
        dst_file = output_dir / split_file
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  ✓ Copied {split_file}")
    
    # 加载模型
    print("\nLoading trained model...")
    diffusion = load_model_and_diffusion(model_path, device, hidden_dim=256, T=200)
    
    # 获取所有序列
    all_sequences = []
    for split in ['train', 'val', 'test']:
        split_file = root_dir / f'{split}_list.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
                all_sequences.extend(sequences)
    
    # 去重
    all_sequences = list(set(all_sequences))
    print(f"Found {len(all_sequences)} sequences to process\n")
    
    # 处理每个序列
    for seq_name in tqdm(all_sequences, desc="Processing sequences"):
        print(f"\nProcessing: {seq_name}")
        seq_path = root_dir / seq_name
        
        if not seq_path.exists():
            print(f"  Skipping: path not found")
            continue
        
        try:
            # 步骤1: 复制整个序列文件夹（排除IMU data.csv）
            output_seq_path = output_dir / seq_name
            print(f"  Copying directory structure...")
            copy_directory_structure(
                seq_path, 
                output_seq_path, 
                exclude_files=['imu0/data.csv']  # 排除IMU数据文件
            )
            
            # 步骤2: 补偿IMU数据
            print(f"  Compensating IMU data...")
            compensated_df, bias_df = compensate_imu_sequence(
                seq_path, diffusion.model, diffusion, window_size=50, device=device
            )
            
            # 步骤3: 保存补偿后的IMU数据（保持原始CSV格式）
            imu_output_dir = output_seq_path / 'mav0' / 'imu0'
            imu_output_dir.mkdir(parents=True, exist_ok=True)
            
            compensated_file = imu_output_dir / 'data.csv'
            
            # 添加回列名的单位信息
            header_mapping = {
                '#timestamp': '#timestamp [ns]',
                'w_RS_S_x': 'w_RS_S_x [rad s^-1]',
                'w_RS_S_y': 'w_RS_S_y [rad s^-1]',
                'w_RS_S_z': 'w_RS_S_z [rad s^-1]',
                'a_RS_S_x': 'a_RS_S_x [m s^-2]',
                'a_RS_S_y': 'a_RS_S_y [m s^-2]',
                'a_RS_S_z': 'a_RS_S_z [m s^-2]'
            }
            compensated_df_renamed = compensated_df.rename(columns=header_mapping)
            compensated_df_renamed.to_csv(compensated_file, index=False)
            
            # 步骤4: 保存预测的bias信息（额外添加，用于分析）
            bias_file = imu_output_dir / 'predicted_bias.csv'
            bias_df.to_csv(bias_file, index=False)
            
            print(f"  ✓ Complete: {seq_name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"All sequences processed!")
    print(f"Compensated dataset saved to: {output_dir}")
    print(f"{'='*60}")


def generate_pegasus_compensated_dataset(root_dir, output_dir, model_path, device='cuda'):
    """
    生成 Pegasus 数据集的 bias 补偿版本
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained model...")
    diffusion = load_model_and_diffusion(model_path, device, hidden_dim=256, T=200)

    seq_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    print(f"Found {len(seq_dirs)} sequences to process\n")

    for seq_path in tqdm(seq_dirs, desc="Processing Pegasus"):
        try:
            output_seq_path = output_dir / seq_path.name
            copy_directory_structure(seq_path, output_seq_path)

            imu_file = seq_path / 'imu_data.csv'
            if not imu_file.exists():
                print(f"  Skipping {seq_path.name}: imu_data.csv not found")
                continue

            imu_df = pd.read_csv(imu_file)
            required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_cols = [c for c in required_cols if c not in imu_df.columns]
            if missing_cols:
                print(f"  Skipping {seq_path.name}: missing columns {missing_cols}")
                continue

            gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
            acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
            imu_values = np.concatenate([gyro, acc], axis=1).astype(np.float32)

            predicted_bias = predict_bias_series(imu_values, diffusion, window_size=50, device=device)
            compensated = imu_values - predicted_bias

            imu_df[['gyro_x', 'gyro_y', 'gyro_z']] = compensated[:, 0:3]
            imu_df[['acc_x', 'acc_y', 'acc_z']] = compensated[:, 3:6]

            out_file = output_seq_path / 'imu_data.csv'
            imu_df.to_csv(out_file, index=False)

            bias_df = pd.DataFrame({
                'timestamp': imu_df['timestamp'].values,
                'b_w_x_pred': predicted_bias[:, 0],
                'b_w_y_pred': predicted_bias[:, 1],
                'b_w_z_pred': predicted_bias[:, 2],
                'b_a_x_pred': predicted_bias[:, 3],
                'b_a_y_pred': predicted_bias[:, 4],
                'b_a_z_pred': predicted_bias[:, 5]
            })
            bias_df.to_csv(output_seq_path / 'predicted_bias.csv', index=False)

            print(f"  ✓ Complete: {seq_path.name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Pegasus compensated dataset saved to: {output_dir}")
    print(f"{'='*60}")


def read_blackbird_imu(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    col_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    if first_line.startswith('#'):
        df = pd.read_csv(file_path, comment='#', header=None, names=col_names)
        has_comment = True
    else:
        df = pd.read_csv(file_path)
        has_comment = False
    return df, has_comment


def write_blackbird_imu(df, file_path, include_comment):
    if include_comment:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('# timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z\n')
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def generate_blackbird_compensated_dataset(root_dir, output_dir, model_path, device='cuda'):
    """
    生成 Blackbird 数据集的 bias 补偿版本
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained model...")
    diffusion = load_model_and_diffusion(model_path, device, hidden_dim=256, T=200)

    imu_files = []
    for root, _, files in os.walk(root_dir):
        if 'imu_data.csv' in files:
            imu_files.append(Path(root) / 'imu_data.csv')

    print(f"Found {len(imu_files)} imu_data.csv files to process\n")

    for imu_file in tqdm(imu_files, desc="Processing Blackbird"):
        try:
            rel_dir = imu_file.parent.relative_to(root_dir)
            output_dir_path = output_dir / rel_dir
            output_dir_path.mkdir(parents=True, exist_ok=True)

            df, has_comment = read_blackbird_imu(imu_file)
            required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"  Skipping {imu_file}: missing columns {missing_cols}")
                continue

            gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
            acc = df[['acc_x', 'acc_y', 'acc_z']].values
            imu_values = np.concatenate([gyro, acc], axis=1).astype(np.float32)

            predicted_bias = predict_bias_series(imu_values, diffusion, window_size=50, device=device)
            compensated = imu_values - predicted_bias

            df[['gyro_x', 'gyro_y', 'gyro_z']] = compensated[:, 0:3]
            df[['acc_x', 'acc_y', 'acc_z']] = compensated[:, 3:6]

            out_file = output_dir_path / 'imu_data.csv'
            write_blackbird_imu(df, out_file, has_comment)

            bias_df = pd.DataFrame({
                'timestamp': df['timestamp'].values,
                'b_w_x_pred': predicted_bias[:, 0],
                'b_w_y_pred': predicted_bias[:, 1],
                'b_w_z_pred': predicted_bias[:, 2],
                'b_a_x_pred': predicted_bias[:, 3],
                'b_a_y_pred': predicted_bias[:, 4],
                'b_a_z_pred': predicted_bias[:, 5]
            })
            bias_df.to_csv(output_dir_path / 'predicted_bias.csv', index=False)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Blackbird compensated dataset saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    # ==================== 配置参数 ====================
    # 原始数据集路径
    EUROC_ROOT = Path(__file__).parent.parent / "数据" / "Euroc数据集" / "EuRoC-Dataset"
    PEGASUS_ROOT = Path(__file__).parent.parent / "数据" / "Pegasus数据集" / "PegasusDataset"
    BLACKBIRD_ROOT = Path(__file__).parent.parent / "数据" / "blackbird数据集" / "Blackbird" / "Blackbird"

    # 输出路径
    EUROC_OUTPUT = Path(__file__).parent.parent / "数据" / "Euroc数据集" / "EuRoC-Dataset-DiffusionCompensated"
    PEGASUS_OUTPUT = Path(__file__).parent.parent / "数据" / "Pegasus数据集" / "PegasusDataset-Denoised"
    BLACKBIRD_OUTPUT = Path(__file__).parent.parent / "数据" / "blackbird数据集" / "Blackbird" / "Blackbird_denoised"

    # 模型路径
    MODEL_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 检查模型
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 生成补偿后的数据集
    if EUROC_ROOT.exists():
        generate_euroc_compensated_dataset(
            root_dir=EUROC_ROOT,
            output_dir=EUROC_OUTPUT,
            model_path=MODEL_PATH,
            device=device
        )
    else:
        print(f"Warning: EuRoC dataset not found at {EUROC_ROOT}")

    if PEGASUS_ROOT.exists():
        generate_pegasus_compensated_dataset(
            root_dir=PEGASUS_ROOT,
            output_dir=PEGASUS_OUTPUT,
            model_path=MODEL_PATH,
            device=device
        )
    else:
        print(f"Warning: Pegasus dataset not found at {PEGASUS_ROOT}")

    if BLACKBIRD_ROOT.exists():
        generate_blackbird_compensated_dataset(
            root_dir=BLACKBIRD_ROOT,
            output_dir=BLACKBIRD_OUTPUT,
            model_path=MODEL_PATH,
            device=device
        )
    else:
        print(f"Warning: Blackbird dataset not found at {BLACKBIRD_ROOT}")

    print("\n✓ Dataset generation complete!")
    print("\n你现在可以使用补偿后的数据集进行无人机定位:")
    print(f"  EuRoC: {EUROC_OUTPUT}")
    print(f"  Pegasus: {PEGASUS_OUTPUT}")
    print(f"  Blackbird: {BLACKBIRD_OUTPUT}")
    print("  额外文件: predicted_bias.csv (包含每个时刻预测的bias)")


if __name__ == "__main__":
    main()
