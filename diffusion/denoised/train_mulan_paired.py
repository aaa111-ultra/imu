import os
import shutil
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from flax import serialization

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(BASE_DIR, '..', '数据', 'Euroc数据集'))

CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'window_size': 128,    # 窗口大小
    'stride': 64,
    'learning_rate': 5e-5, # 有监督学习可以用稍小的 LR
    'epochs': 100,
    'channels': 3,         # 加速度计 3轴
    # EuRoC 数据集路径 (按你的目录结构)
    'noisy_root': os.path.join(DATA_ROOT, 'EuRoC-Dataset'),
    'gt_root': os.path.join(DATA_ROOT, 'EuRoC-Dataset'),
    'train_list': os.path.join(DATA_ROOT, 'EuRoC-Dataset', 'train_list.txt'),
    'val_list': os.path.join(DATA_ROOT, 'EuRoC-Dataset', 'val_list.txt'),
    'test_list': os.path.join(DATA_ROOT, 'EuRoC-Dataset', 'test_list.txt'),
    'output_root': os.path.join(DATA_ROOT, 'EuRoC-Dataset-Denoised'),
    'export_batch_size': 64,
    'export_after_train': True,
    'export_pegasus': True,
    'export_blackbird': True,
    'grad_clip_norm': 1.0,
    'log_var_min': -4.0,
    'log_var_max': 6.0,
    'use_huber': True,
    'huber_delta': 1.0,
    'log_var_weight': 0.1,
    'eval_every': 5,
    'use_lr_schedule': True,
    'warmup_epochs': 10,
    'augment_data': True,
    'augment_std': 1e-3, # 噪声增强标准差
    'weight_decay': 1e-4,
    'use_ema': True,
    'ema_decay': 0.995,
    'skip_train': False,
    'load_checkpoint': True,
    'save_checkpoint': True,
    'checkpoint_path': os.path.join(BASE_DIR, 'checkpoints', 'mulan_params.msgpack'),
    # Pegasus / Blackbird 数据集路径
    'pegasus_root': os.path.normpath(os.path.join(DATA_ROOT, '..', 'Pegasus数据集', 'PegasusDataset')),
    'pegasus_output_root': os.path.normpath(os.path.join(DATA_ROOT, '..', 'Pegasus数据集', 'PegasusDataset-Denoised')),
    'blackbird_root': os.path.normpath(os.path.join(DATA_ROOT, '..', 'blackbird数据集', 'Blackbird', 'Blackbird')),
    'blackbird_output_root': os.path.normpath(os.path.join(DATA_ROOT, '..', 'blackbird数据集', 'Blackbird', 'Blackbird_denoised')),
}

GYRO_COLS = ['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']
ACC_COLS = ['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']
B_GYRO_COLS = ['b_w_RS_S_x [rad s^-1]', 'b_w_RS_S_y [rad s^-1]', 'b_w_RS_S_z [rad s^-1]']
B_ACC_COLS = ['b_a_RS_S_x [m s^-2]', 'b_a_RS_S_y [m s^-2]', 'b_a_RS_S_z [m s^-2]']

BLACKBIRD_ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
BLACKBIRD_GYRO_COLS = ['gyro_x', 'gyro_y', 'gyro_z']
EUROC_IMU_COLS = [
    'timestamp [ns]',
    'w_RS_S_x [rad s^-1]',
    'w_RS_S_y [rad s^-1]',
    'w_RS_S_z [rad s^-1]',
    'a_RS_S_x [m s^-2]',
    'a_RS_S_y [m s^-2]',
    'a_RS_S_z [m s^-2]',
]


def load_seq_list(list_file):
    with open(list_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_seq_lists(list_files):
    seqs = []
    for list_file in list_files:
        if list_file and os.path.exists(list_file):
            seqs.extend(load_seq_list(list_file))
    return seqs

# ==========================================
# 2. 成对数据加载器 (Paired Data Loader)
# ==========================================
class EuRoCPairedDataset:
    def __init__(self, noisy_root, gt_root, list_file, window_size=128, stride=64):
        self.window_size = window_size
        self.stride = stride

        seq_list = load_seq_list(list_file)

        print(f"正在加载 EuRoC 数据...\nNoisy root: {noisy_root}\nGT root: {gt_root}")
        print(f"训练序列数量: {len(seq_list)}")

        noisy_segments = []
        clean_segments = []

        for seq in seq_list:
            noisy_path = os.path.join(noisy_root, seq, 'mav0', 'imu0', 'data.csv')
            gt_path = os.path.join(gt_root, seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')

            if not os.path.exists(noisy_path) or not os.path.exists(gt_path):
                print(f"跳过序列 {seq} (文件不存在)")
                continue

            df_noisy = pd.read_csv(noisy_path)
            df_gt = pd.read_csv(gt_path)

            df_noisy.columns = [c.strip() for c in df_noisy.columns]
            df_gt.columns = [c.strip() for c in df_gt.columns]

            time_col_noisy = [c for c in df_noisy.columns if 'timestamp' in c.lower()][0]
            time_col_gt = [c for c in df_gt.columns if 'timestamp' in c.lower()][0]

            df_noisy = df_noisy.rename(columns={time_col_noisy: 'timestamp'}).sort_values('timestamp')
            df_gt = df_gt.rename(columns={time_col_gt: 'timestamp'}).sort_values('timestamp')

            # Align imu with nearest groundtruth to get bias, then form clean = noisy - bias
            df_merged = pd.merge_asof(df_noisy, df_gt, on='timestamp', direction='nearest', tolerance=1_000_000)

            df_merged = df_merged.dropna(subset=B_GYRO_COLS + B_ACC_COLS)

            acc = df_merged[ACC_COLS].values.astype(np.float32)
            b_acc = df_merged[B_ACC_COLS].values.astype(np.float32)

            data_noisy = acc
            data_clean = acc - b_acc

            print(f"{seq}: 对齐后数据量 {len(data_noisy)}")
            noisy_segments.append(data_noisy)
            clean_segments.append(data_clean)

        if not noisy_segments:
            raise ValueError("未加载到任何有效序列，请检查数据路径与 list 文件")

        data_noisy = np.concatenate(noisy_segments, axis=0)
        data_clean = np.concatenate(clean_segments, axis=0)

        self.mean = np.mean(data_noisy, axis=0)
        self.std = np.std(data_noisy, axis=0) + 1e-6

        self.data_noisy = (data_noisy - self.mean) / self.std
        self.data_clean = (data_clean - self.mean) / self.std

        self.windows_noisy = []
        self.windows_clean = []

        n_samples = len(self.data_noisy)
        for i in range(0, n_samples - window_size, stride):
            self.windows_noisy.append(self.data_noisy[i : i + window_size])
            self.windows_clean.append(self.data_clean[i : i + window_size])

        self.windows_noisy = np.array(self.windows_noisy)
        self.windows_clean = np.array(self.windows_clean)

        print(f"样本生成完毕: {self.windows_noisy.shape}")

    def get_batch(self, batch_size, key):
        """随机采样成对的 Batch"""
        indices = jax.random.randint(key, (batch_size,), 0, len(self.windows_noisy))
        noisy_batch = jnp.array(self.windows_noisy[indices])
        clean_batch = jnp.array(self.windows_clean[indices])

        if CONFIG['augment_data']:
            key, aug_key = jax.random.split(key)
            noise = jax.random.normal(aug_key, noisy_batch.shape) * CONFIG['augment_std']
            noisy_batch = noisy_batch + noise

        return noisy_batch, clean_batch

# ==========================================
# 3. MuLAN 1D 模型 (保持不变，核心架构)
# ==========================================
class ResidualBlock1D(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3,))(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.features, kernel_size=(3,))(x)
        x = nn.GroupNorm(num_groups=8)(x)
        if residual.shape[-1] != self.features:
            residual = nn.Conv(features=self.features, kernel_size=(1,))(residual)
        return x + residual

class MuLAN_UNet1D(nn.Module):
    out_channels: int = 6
    base_features: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 注意：有监督模式下，我们可以不需要 Time Embedding，或者将其置为0
        # 这里为了简化，我们去掉 Time Embedding，变成纯粹的 Image-to-Image UNet
        
        # Encoder
        h1 = nn.Conv(self.base_features, kernel_size=(3,))(x)
        h1 = ResidualBlock1D(self.base_features)(h1, train)
        
        h2 = nn.max_pool(h1, window_shape=(2,), strides=(2,))
        h2 = ResidualBlock1D(self.base_features * 2)(h2, train)
        
        h3 = nn.max_pool(h2, window_shape=(2,), strides=(2,))
        h3 = ResidualBlock1D(self.base_features * 4)(h3, train)
        
        # Bottleneck
        h_mid = ResidualBlock1D(self.base_features * 4)(h3, train)
        h_mid = ResidualBlock1D(self.base_features * 4)(h_mid, train)
        
        # Decoder
        h4 = jax.image.resize(h_mid, shape=(h_mid.shape[0], h_mid.shape[1]*2, h_mid.shape[2]), method='nearest')
        h4 = jnp.concatenate([h4, h2], axis=-1)
        h4 = ResidualBlock1D(self.base_features * 2)(h4, train)
        
        h5 = jax.image.resize(h4, shape=(h4.shape[0], h4.shape[1]*2, h4.shape[2]), method='nearest')
        h5 = jnp.concatenate([h5, h1], axis=-1)
        h5 = ResidualBlock1D(self.base_features)(h5, train)
        
        # Output Heads
        # 1. 预测 Clean Data (去噪结果)
        pred_clean = nn.Conv(self.out_channels, kernel_size=(3,), use_bias=True)(h5)
        
        # 2. 预测不确定性 (Aleatoric Uncertainty / Sigma)
        pred_log_var = nn.Conv(self.out_channels, kernel_size=(3,), use_bias=True)(h5)
        
        return pred_clean, pred_log_var

# ==========================================
# 4. MuLAN 核心 Loss (有监督版)
# ==========================================
@jax.jit
def train_step(state, batch_noisy, batch_clean):
    """
    MuLAN 的核心思想：自适应 Loss
    Loss = (Target - Pred)^2 / (2 * Sigma^2) + 0.5 * log(Sigma^2)
    这会自动降低 '难以预测/噪声极大' 区域的权重，防止模型强行拟合噪声。
    """
    def loss_fn(params):
        pred_clean, pred_log_var = state.apply_fn({'params': params}, batch_noisy, train=True)
        pred_log_var = jnp.clip(pred_log_var, CONFIG['log_var_min'], CONFIG['log_var_max'])
        
        # 1. 重构误差 (Reconstruction Error)
        # 使用预测的方差来加权 MSE
        precision = jnp.exp(-pred_log_var)
        residual = batch_clean - pred_clean
        if CONFIG['use_huber']:
            base_loss = optax.huber_loss(residual, delta=CONFIG['huber_delta'])
        else:
            base_loss = residual ** 2
        weighted_mse = jnp.mean(precision * base_loss)
        
        # 2. 正则化项 (Regularization)
        # 惩罚过大的不确定性，防止模型通过无限增大 Sigma 来作弊
        log_var_penalty = jnp.mean(pred_log_var)
        total_loss = weighted_mse + CONFIG['log_var_weight'] * log_var_penalty
        return total_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, params, batch_noisy, batch_clean):
    pred_clean, pred_log_var = state.apply_fn({'params': params}, batch_noisy, train=False)
    pred_log_var = jnp.clip(pred_log_var, CONFIG['log_var_min'], CONFIG['log_var_max'])

    precision = jnp.exp(-pred_log_var)
    residual = batch_clean - pred_clean
    if CONFIG['use_huber']:
        base_loss = optax.huber_loss(residual, delta=CONFIG['huber_delta'])
    else:
        base_loss = residual ** 2
    weighted_mse = jnp.mean(precision * base_loss)
    log_var_penalty = jnp.mean(pred_log_var)
    total_loss = weighted_mse + CONFIG['log_var_weight'] * log_var_penalty
    return total_loss

# ==========================================
# 5. 训练主流程
# ==========================================
def run_training():
    if not os.path.exists(CONFIG['noisy_root']):
        print("错误：请检查 noisy_root 路径")
        return None
    if not os.path.exists(CONFIG['gt_root']):
        print("错误：请检查 gt_root 路径")
        return None
    if not os.path.exists(CONFIG['train_list']):
        print("错误：找不到 train_list.txt")
        return None

    # 初始化
    key = jax.random.PRNGKey(CONFIG['seed'])
    dataset = EuRoCPairedDataset(
        CONFIG['noisy_root'],
        CONFIG['gt_root'],
        CONFIG['train_list'],
        window_size=CONFIG['window_size'],
        stride=CONFIG['stride']
    )
    val_dataset = None
    if os.path.exists(CONFIG['val_list']):
        val_dataset = EuRoCPairedDataset(
            CONFIG['noisy_root'],
            CONFIG['gt_root'],
            CONFIG['val_list'],
            window_size=CONFIG['window_size'],
            stride=CONFIG['stride']
        )
    
    model = MuLAN_UNet1D(out_channels=CONFIG['channels'])
    
    # Init params
    key, init_key = jax.random.split(key)
    dummy_x = jnp.ones((1, CONFIG['window_size'], CONFIG['channels']))
    params = model.init(init_key, dummy_x)['params']
    
    if CONFIG['use_lr_schedule']:
        schedule = optax.cosine_decay_schedule(
            init_value=CONFIG['learning_rate'],
            decay_steps=CONFIG['epochs'] - CONFIG['warmup_epochs'],
            alpha=0.01 # 最小学习率比例
        )
        # 预热
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=CONFIG['learning_rate'],
            transition_steps=CONFIG['warmup_epochs']
        )
        # 合并调度器
        lr_schedule = optax.join_schedules(
            schedules=[warmup_fn, schedule],
            boundaries=[CONFIG['warmup_epochs']]
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = optax.adamw(CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        
    tx = optax.chain(
        optax.clip_by_global_norm(CONFIG['grad_clip_norm']),
        optimizer
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if CONFIG['load_checkpoint'] and os.path.exists(CONFIG['checkpoint_path']):
        with open(CONFIG['checkpoint_path'], 'rb') as f:
            ckpt_bytes = f.read()
        loaded_params = serialization.from_bytes(state.params, ckpt_bytes)
        state = state.replace(params=loaded_params)
        ema_params = loaded_params if CONFIG['use_ema'] else None
        print(f"已加载模型参数: {CONFIG['checkpoint_path']}")
    else:
        ema_params = state.params if CONFIG['use_ema'] else None

    if CONFIG['skip_train']:
        if not os.path.exists(CONFIG['checkpoint_path']):
            print("错误：skip_train=True 但未找到 checkpoint，请先训练并保存模型")
            return None
        print("跳过训练，直接导出")
        final_params = ema_params if CONFIG['use_ema'] else state.params
        if CONFIG['export_after_train']:
            export_denoised_dataset(state, final_params, dataset)
        if CONFIG['export_pegasus']:
            export_pegasus_dataset(state, final_params, dataset)
        if CONFIG['export_blackbird']:
            export_blackbird_dataset(state, final_params, dataset)
        return state
    
    print("\n=== 开始训练 (有监督 MuLAN 模式) ===")
    
    loss_history = []
    val_loss_history = []
    
    steps_per_epoch = max(1, len(dataset.windows_noisy) // CONFIG['batch_size'])

    for epoch in range(CONFIG['epochs']):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            key, batch_key = jax.random.split(key)
            batch_noisy, batch_clean = dataset.get_batch(CONFIG['batch_size'], batch_key)
            
            state, loss = train_step(state, batch_noisy, batch_clean)

            if CONFIG['use_ema']:
                ema_params = optax.incremental_update(
                    state.params,
                    ema_params,
                    step_size=1.0 - CONFIG['ema_decay']
                )

            epoch_losses.append(float(loss))

        epoch_loss = float(np.mean(epoch_losses))
        loss_history.append(epoch_loss)
        if epoch % 10 == 0:
            if val_dataset is not None and epoch % CONFIG['eval_every'] == 0:
                key, val_key = jax.random.split(key)
                val_noisy, val_clean = val_dataset.get_batch(CONFIG['batch_size'], val_key)
                eval_params = ema_params if CONFIG['use_ema'] else state.params
                val_loss = eval_step(state, eval_params, val_noisy, val_clean)
                val_loss_history.append(float(val_loss))
                print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Val: {float(val_loss):.4f}")
            else:
                val_loss_history.append(None)
                print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")
        else:
            val_loss_history.append(None)
            
    # 保存 loss 曲线与去噪效果图
    plot_loss_curve(loss_history, val_loss_history)
    final_params = ema_params if CONFIG['use_ema'] else state.params
    plot_results(state, final_params, dataset)

    if CONFIG['save_checkpoint']:
        os.makedirs(os.path.dirname(CONFIG['checkpoint_path']), exist_ok=True)
        with open(CONFIG['checkpoint_path'], 'wb') as f:
            f.write(serialization.to_bytes(final_params))
        print(f"已保存模型参数: {CONFIG['checkpoint_path']}")

    if CONFIG['export_after_train']:
        export_denoised_dataset(state, final_params, dataset)
    if CONFIG['export_pegasus']:
        export_pegasus_dataset(state, final_params, dataset)
    if CONFIG['export_blackbird']:
        export_blackbird_dataset(state, final_params, dataset)
    return state


def build_window_starts(n_samples, window_size, stride):
    if n_samples < window_size:
        raise ValueError("序列长度小于 window_size，无法切窗")
    starts = list(range(0, n_samples - window_size + 1, stride))
    last_start = n_samples - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def denoise_sequence(state, params, dataset, data_noisy):
    data_norm = (data_noisy - dataset.mean) / dataset.std
    n_samples = len(data_norm)
    starts = build_window_starts(n_samples, dataset.window_size, CONFIG['stride'])

    accum = np.zeros((n_samples, CONFIG['channels']), dtype=np.float32)
    counts = np.zeros((n_samples, 1), dtype=np.float32)

    batch_size = CONFIG['export_batch_size']
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i:i + batch_size]
        batch = np.stack([data_norm[s:s + dataset.window_size] for s in batch_starts], axis=0)
        pred_clean, _ = state.apply_fn({'params': params}, jnp.array(batch), train=False)
        pred_clean = np.array(pred_clean)

        for j, s in enumerate(batch_starts):
            accum[s:s + dataset.window_size] += pred_clean[j]
            counts[s:s + dataset.window_size] += 1.0

    counts[counts == 0] = 1.0
    denoised_norm = accum / counts
    denoised = denoised_norm * dataset.std + dataset.mean
    return denoised


def export_denoised_dataset(state, params, dataset):
    list_files = [CONFIG['train_list'], CONFIG['val_list'], CONFIG['test_list']]
    seq_list = load_seq_lists(list_files)
    if not seq_list:
        print("未找到任何序列，导出取消")
        return

    os.makedirs(CONFIG['output_root'], exist_ok=True)

    for seq in seq_list:
        src_seq_dir = os.path.join(CONFIG['noisy_root'], seq)
        dst_seq_dir = os.path.join(CONFIG['output_root'], seq)
        if not os.path.exists(src_seq_dir):
            print(f"跳过序列 {seq} (源目录不存在)")
            continue

        shutil.copytree(src_seq_dir, dst_seq_dir, dirs_exist_ok=True)

        imu_src = os.path.join(src_seq_dir, 'mav0', 'imu0', 'data.csv')
        imu_dst = os.path.join(dst_seq_dir, 'mav0', 'imu0', 'data.csv')
        if not os.path.exists(imu_src):
            print(f"跳过序列 {seq} (imu0/data.csv 不存在)")
            continue

        df_noisy = pd.read_csv(imu_src)
        missing_cols = [c for c in ACC_COLS if c not in df_noisy.columns]
        if missing_cols:
            print(f"跳过序列 {seq} (缺少列: {missing_cols})")
            continue

        data_noisy = df_noisy[ACC_COLS].values.astype(np.float32)
        denoised = denoise_sequence(state, params, dataset, data_noisy)

        df_noisy[ACC_COLS] = denoised
        df_noisy.to_csv(imu_dst, index=False)
        print(f"已导出: {imu_dst}")


def export_pegasus_dataset(state, params, dataset):
    src_root = CONFIG['pegasus_root']
    dst_root = CONFIG['pegasus_output_root']
    if not os.path.exists(src_root):
        print(f"Pegasus 根目录不存在: {src_root}")
        return

    os.makedirs(dst_root, exist_ok=True)

    seq_dirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    for seq in seq_dirs:
        src_seq_dir = os.path.join(src_root, seq)
        dst_seq_dir = os.path.join(dst_root, seq)
        shutil.copytree(src_seq_dir, dst_seq_dir, dirs_exist_ok=True)

        imu_src = os.path.join(src_seq_dir, 'imu_data.csv')
        imu_dst = os.path.join(dst_seq_dir, 'imu_data.csv')
        if not os.path.exists(imu_src):
            print(f"跳过序列 {seq} (imu_data.csv 不存在)")
            continue

        df_noisy = pd.read_csv(imu_src)
        missing_cols = [c for c in ['acc_x', 'acc_y', 'acc_z'] if c not in df_noisy.columns]
        if missing_cols:
            print(f"跳过序列 {seq} (缺少列: {missing_cols})")
            continue

        data_noisy = df_noisy[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float32)
        denoised = denoise_sequence(state, params, dataset, data_noisy)

        df_noisy[['acc_x', 'acc_y', 'acc_z']] = denoised
        df_noisy.to_csv(imu_dst, index=False)
        print(f"已导出: {imu_dst}")


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
    df.columns = [c.strip() for c in df.columns]
    time_cols = [c for c in df.columns if 'timestamp' in c.lower()]
    if time_cols and time_cols[0] != 'timestamp':
        df = df.rename(columns={time_cols[0]: 'timestamp'})
    if 'timestamp' not in df.columns:
        df = pd.read_csv(file_path, header=None, names=col_names)
        df.columns = [c.strip() for c in df.columns]
        has_comment = False
    return df, has_comment


def write_blackbird_imu_euroc(df, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('#' + ','.join(EUROC_IMU_COLS) + '\n')
    df.to_csv(file_path, mode='a', header=False, index=False)


def export_blackbird_dataset(state, params, dataset):
    src_root = CONFIG['blackbird_root']
    dst_root = CONFIG['blackbird_output_root']
    if not os.path.exists(src_root):
        print(f"Blackbird 根目录不存在: {src_root}")
        return

    os.makedirs(dst_root, exist_ok=True)

    for root, _, files in os.walk(src_root):
        if 'imu_data.csv' not in files:
            continue

        rel_dir = os.path.relpath(root, src_root)
        src_imu = os.path.join(root, 'imu_data.csv')
        dst_dir = os.path.join(dst_root, rel_dir)
        dst_imu = os.path.join(dst_dir, 'imu_data.csv')

        os.makedirs(dst_dir, exist_ok=True)
        df_noisy, _ = read_blackbird_imu(src_imu)

        if 'timestamp' not in df_noisy.columns:
            print(f"跳过: {src_imu} (缺少 timestamp 列)")
            continue

        missing_cols = [c for c in BLACKBIRD_ACC_COLS if c not in df_noisy.columns]
        if missing_cols:
            print(f"跳过: {src_imu} (缺少列: {missing_cols})")
            continue

        data_noisy = df_noisy[BLACKBIRD_ACC_COLS].values.astype(np.float32)
        denoised = denoise_sequence(state, params, dataset, data_noisy)
        df_noisy[BLACKBIRD_ACC_COLS] = denoised

        missing_gyro = [c for c in BLACKBIRD_GYRO_COLS if c not in df_noisy.columns]
        if missing_gyro:
            print(f"跳过: {src_imu} (缺少列: {missing_gyro})")
            continue

        df_out = pd.DataFrame({
            EUROC_IMU_COLS[0]: df_noisy['timestamp'].values,
            EUROC_IMU_COLS[1]: df_noisy[BLACKBIRD_GYRO_COLS[0]].values,
            EUROC_IMU_COLS[2]: df_noisy[BLACKBIRD_GYRO_COLS[1]].values,
            EUROC_IMU_COLS[3]: df_noisy[BLACKBIRD_GYRO_COLS[2]].values,
            EUROC_IMU_COLS[4]: df_noisy[BLACKBIRD_ACC_COLS[0]].values,
            EUROC_IMU_COLS[5]: df_noisy[BLACKBIRD_ACC_COLS[1]].values,
            EUROC_IMU_COLS[6]: df_noisy[BLACKBIRD_ACC_COLS[2]].values,
        })

        write_blackbird_imu_euroc(df_out, dst_imu)
        print(f"已导出: {dst_imu}")

def plot_results(state, params, dataset):
    # 取一个测试样本
    idx = 0
    noisy_sample = dataset.windows_noisy[idx:idx+1]
    clean_sample = dataset.windows_clean[idx:idx+1]
    
    # 推理
    pred_clean, pred_log_var = state.apply_fn({'params': params}, noisy_sample, train=False)
    sigma = jnp.exp(0.5 * pred_log_var)
    
    # 反归一化
    noisy_real = noisy_sample * dataset.std + dataset.mean
    clean_real = clean_sample * dataset.std + dataset.mean
    pred_real = pred_clean * dataset.std + dataset.mean

    # 绘制 3 个加速度通道的去噪结果和不确定性
    num_channels = CONFIG['channels']
    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 3 * num_channels))
    
    channel_names = ['Acc X', 'Acc Y', 'Acc Z']

    for i in range(num_channels):
        # 去噪结果
        ax_denoise = axes[i, 0] if num_channels > 1 else axes[0]
        ax_denoise.set_title(f"Channel {i}: {channel_names[i]} Denoising")
        ax_denoise.plot(noisy_real[0, :, i], 'r', alpha=0.3, label='Noisy Input (Raw)')
        ax_denoise.plot(clean_real[0, :, i], 'g', label='Ground Truth (Clean)')
        ax_denoise.plot(pred_real[0, :, i], 'b', linestyle='--', label='MuLAN Output')
        ax_denoise.legend()
        ax_denoise.grid(True)
        
        # 不确定性
        ax_uncertainty = axes[i, 1] if num_channels > 1 else axes[1]
        ax_uncertainty.set_title(f"Channel {i}: {channel_names[i]} Learned Uncertainty (Sigma)")
        ax_uncertainty.plot(sigma[0, :, i], 'k', label='Predicted Uncertainty')
        ax_uncertainty.fill_between(range(dataset.window_size), 0, sigma[0, :, i], color='gray', alpha=0.3)
        ax_uncertainty.legend()
        ax_uncertainty.grid(True)
    
    plt.tight_layout()
    plt.savefig("paired_training_result.png")
    print("结果图已保存: paired_training_result.png")


def plot_loss_curve(loss_history, val_loss_history=None):
    epochs = np.arange(1, len(loss_history) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss_history, label='Train Loss')

    if val_loss_history:
        val_epochs = [i + 1 for i, v in enumerate(val_loss_history) if v is not None]
        val_values = [v for v in val_loss_history if v is not None]
        if val_epochs:
            plt.plot(val_epochs, val_values, label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss_curve.png")
    print("Loss 曲线已保存: training_loss_curve.png")

if __name__ == "__main__":
    run_training()