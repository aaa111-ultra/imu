import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os

class EuRoCBiasDataset(Dataset):
    """
    EuRoC 数据集加载器
    用于加载 IMU 数据和对应的 Bias 真值
    """
    def __init__(self, root_dir, split='train', window_size=50, stride=10):
        """
        Args:
            root_dir: EuRoC 数据集根目录
            split: 'train', 'test', 'val'
            window_size: IMU 滑动窗口大小
            stride: 滑动窗口步长
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.stride = stride
        
        # 读取数据集划分列表
        split_file = self.root_dir / f'{split}_list.txt'
        with open(split_file, 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]
        
        print(f"Loading {split} set with {len(self.sequences)} sequences: {self.sequences}")
        
        # 加载所有序列的数据
        self.data_samples = []
        self._load_all_sequences()
        
        print(f"Total {len(self.data_samples)} samples in {split} set")
    
    def _load_all_sequences(self):
        """加载所有序列并生成滑动窗口样本"""
        for seq_name in self.sequences:
            try:
                seq_path = self.root_dir / seq_name / 'mav0'
                
                # 读取 IMU 数据
                imu_file = seq_path / 'imu0' / 'data.csv'
                imu_data = pd.read_csv(imu_file)
                
                # 读取真值数据（包含 bias）
                gt_file = seq_path / 'state_groundtruth_estimate0' / 'data.csv'
                gt_data = pd.read_csv(gt_file)
                
                # 提取列名（去除空格和单位）
                imu_data.columns = [col.split(' [')[0].strip() for col in imu_data.columns]
                gt_data.columns = [col.split(' [')[0].strip() for col in gt_data.columns]
                
                # 提取 IMU 数据：角速度 (w_x, w_y, w_z) + 加速度 (a_x, a_y, a_z)
                imu_values = imu_data[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z', 
                                         'a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].values
                imu_timestamps = imu_data['#timestamp'].values
                
                # 兼容两种 GT 列名格式
                # 格式1: b_w_RS_S_x, b_a_RS_S_x (完整版)
                # 格式2: bwx, bax (简写版)
                if 'b_w_RS_S_x' in gt_data.columns:
                    bias_cols = ['b_w_RS_S_x', 'b_w_RS_S_y', 'b_w_RS_S_z',
                                 'b_a_RS_S_x', 'b_a_RS_S_y', 'b_a_RS_S_z']
                elif 'bwx' in gt_data.columns:
                    bias_cols = ['bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz']
                else:
                    raise KeyError(f"Unknown bias column format in {seq_name}")
                
                gt_bias = gt_data[bias_cols].values
                
                # 时间戳列名也有两种格式
                timestamp_col = '#timestamp' if '#timestamp' in gt_data.columns else '#time(ns)'
                gt_timestamps = gt_data[timestamp_col].values
            except KeyError as e:
                print(f"Warning: Skipping sequence {seq_name} due to missing columns: {e}")
                print(f"Available GT columns: {list(gt_data.columns)}")
                continue
            
            # 时间对齐：将 GT bias 插值到 IMU 时间戳
            aligned_bias = self._align_timestamps(imu_timestamps, gt_timestamps, gt_bias)
            
            # 生成滑动窗口样本
            num_samples = (len(imu_values) - self.window_size) // self.stride + 1
            for i in range(num_samples):
                start_idx = i * self.stride
                end_idx = start_idx + self.window_size
                
                # IMU 窗口数据
                imu_window = imu_values[start_idx:end_idx]
                
                # 对应时刻的 Bias（取窗口中间时刻）
                mid_idx = (start_idx + end_idx) // 2
                bias_gt = aligned_bias[mid_idx]
                
                self.data_samples.append({
                    'imu_window': imu_window.astype(np.float32),
                    'bias_gt': bias_gt.astype(np.float32),
                    'sequence': seq_name
                })
    
    def _align_timestamps(self, imu_ts, gt_ts, gt_values):
        """
        将 GT 数据插值到 IMU 时间戳
        """
        aligned = np.zeros((len(imu_ts), gt_values.shape[1]))
        for i in range(gt_values.shape[1]):
            aligned[:, i] = np.interp(imu_ts, gt_ts, gt_values[:, i])
        return aligned
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        imu_window = torch.from_numpy(sample['imu_window'])
        bias_gt = torch.from_numpy(sample['bias_gt'])
        
        return imu_window, bias_gt


def get_euroc_dataloaders(root_dir, batch_size=32, window_size=50, stride=10, num_workers=0):
    """
    创建 EuRoC 数据集的 DataLoader
    
    Args:
        root_dir: EuRoC 数据集根目录
        batch_size: 批次大小
        window_size: IMU 滑动窗口大小
        stride: 滑动窗口步长
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EuRoCBiasDataset(root_dir, split='train', 
                                      window_size=window_size, stride=stride)
    val_dataset = EuRoCBiasDataset(root_dir, split='val', 
                                    window_size=window_size, stride=stride)
    test_dataset = EuRoCBiasDataset(root_dir, split='test', 
                                     window_size=window_size, stride=stride)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 设置数据集路径（相对路径）
    root_dir = Path(__file__).parent.parent / "数据" / "Euroc数据集" / "EuRoC-Dataset"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_euroc_dataloaders(
        root_dir=root_dir,
        batch_size=16,
        window_size=50,
        stride=10
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试一个 batch
    for imu_window, bias_gt in train_loader:
        print(f"\nIMU window shape: {imu_window.shape}")  # (Batch, Window, 6)
        print(f"Bias GT shape: {bias_gt.shape}")          # (Batch, 6)
        print(f"\nSample IMU data (first in batch):\n{imu_window[0][:5]}")
        print(f"\nSample Bias GT (first in batch):\n{bias_gt[0]}")
        break
