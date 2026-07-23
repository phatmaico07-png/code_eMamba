#!/usr/bin/env python3
"""
train.py — Integer-exact QAT + Min-Max + Residual Head
Đã sửa lỗi crash print_summary, thêm bộ kẹp biên trọng số chống tràn số RTL và bộ lập lịch LR.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Tự động đồng bộ các tham số tỷ lệ SCALE từ cấu hình phần cứng của mô hình
from model.mamba_model import build_model
from model.ssm import SCALE, A_SCALE, D2_BIAS_SCALE

# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class MambaConfig:
    input_height: int = 8
    input_width: int = 8
    in_channels: int = 5
    d_model: int = 20
    expand: int = 2
    d_state: int = 8
    d_conv: int = 4
    n_mamba_blocks: int = 2
    patch_size: int = 2
    dt_rank: int = 3
    norm_type: str = "range_norm"
    softplus_type: str = "relu"
    silu_type: str = "silu"
    output_dim: int = 57
    batch_size: int = 32
    epochs: int = 200
    lr: float = 5e-5          # Giảm từ 8e-5 → 5e-5: tránh dao động sau warmup (best ở ep 11 với 8e-5)
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    grad_clip: float = 0.5    # Giảm từ 1.0 → 0.5: chặn gradient lớn trước optimizer step

    @property
    def d_inner(self): return self.expand * self.d_model
    @property
    def seq_len(self): 
        return (self.input_height // self.patch_size) * (self.input_width // self.patch_size)
    @property
    def patch_dim(self):
        return self.patch_size * self.patch_size * self.in_channels

    def print_summary(self):
        print("="*75)
        print("eMamba Training Config (Integer-exact QAT + Residual Head + Min-Max)")
        print("="*75)
        print(f" D={self.d_model} | E={self.expand} | D_INNER={self.d_inner}")
        print(f" N_STATE={self.d_state} | KERNEL={self.d_conv} | BLOCKS={self.n_mamba_blocks}")
        print(f" SEQ_LEN={self.seq_len} | PATCH_DIM={self.patch_dim}")
        print(f" HEAD: Residual MLP (20→64→20→57) | QUANT: INT8 (SCALE={SCALE})")
        print(f" Epochs={self.epochs} | LR={self.lr} | Warmup={self.warmup_epochs} | Batch={self.batch_size}")
        print("="*75)


# =====================================================================
# DATASET & PREPROCESS
# =====================================================================
class MARSDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def load_norm_range(norm_path="/kaggle/working/data/processed/mars/norm_range.npz"):
    nr = np.load(norm_path)
    return nr['pj_mins'], nr['pj_maxs']


def preprocess():
    """
    Load data đã được preprocess.py chuẩn hóa sẵn.
    KHÔNG normalize lại — preprocess.py đã xử lý:
      - X: percentile [1%, 99%] per-channel → [-1, +1]
      - Y: min-max per-joint per-axis → [-1, +1], shape (N, 19, 3)
    Chỉ reshape Y từ (N, 19, 3) → (N, 57) để khớp output_dim=57.
    """
    base = "/kaggle/working/data/processed/mars"
    train_data = np.load(f"{base}/train.npz")
    test_data  = np.load(f"{base}/test.npz")

    X_tr = train_data['X'].astype(np.float32)  # (N, H, W, C) đã chuẩn hóa
    Y_tr = train_data['Y'].astype(np.float32)  # (N, 19, 3)   đã chuẩn hóa
    X_te = test_data['X'].astype(np.float32)
    Y_te = test_data['Y'].astype(np.float32)

    # Flatten Y: (N, 19, 3) → (N, 57) cho MSELoss
    Y_tr_flat = Y_tr.reshape(-1, 57)
    Y_te_flat = Y_te.reshape(-1, 57)

    pj_mins, pj_maxs = load_norm_range(f"{base}/norm_range.npz")

    return X_tr, Y_tr_flat, X_te, Y_te_flat, pj_mins, pj_maxs


# =====================================================================
# EVALUATION (Chính xác theo Min-Max nghịch đảo)
# =====================================================================
def evaluate(model, loader, device, pj_mins, pj_maxs):
    """
    Denormalize nhất quán với preprocess.py:
      pred/gt ∈ [-1,+1]  →  (val+1)/2 * (maxs-mins) + mins  →  mét  →  *100 = cm
    pj_mins/pj_maxs shape: (19, 3)
    """
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            gts.append(y)

    pred_norm = torch.cat(preds).numpy().reshape(-1, 19, 3)  # (N,57)→(N,19,3)
    gt_norm   = torch.cat(gts).numpy().reshape(-1, 19, 3)

    rng = pj_maxs - pj_mins  # (19, 3)
    pred_m = (pred_norm + 1) / 2 * rng + pj_mins  # mét
    gt_m   = (gt_norm   + 1) / 2 * rng + pj_mins

    pred_cm = pred_m * 100
    gt_cm   = gt_m   * 100

    mae    = np.mean(np.abs(pred_cm - gt_cm))
    rmse   = np.sqrt(np.mean((pred_cm - gt_cm) ** 2))
    mpjpe  = np.mean(np.linalg.norm(pred_cm - gt_cm, axis=-1))

    return {"mae_cm": mae, "rmse_cm": rmse, "mpjpe_cm": mpjpe}


# =====================================================================
# HÀM BẮT BUỘC CHO QAT: Đảm bảo trọng số khớp dải INT8/INT16 của RTL
# =====================================================================
def clamp_weights(model):
    """Ép biên chống tràn số khi chuyển đổi mô hình từ số thực sang ma trận .mem cứng trên kit FPGA"""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'A_log' in name:
                p.clamp_(max=math.log(128.0 / float(A_SCALE)))
            elif 'proj_delta_2.bias' in name:
                p.clamp_(-128.0 / float(D2_BIAS_SCALE), 127.0 / float(D2_BIAS_SCALE))
            else:
                p.clamp_(-128.0 / float(SCALE), 127.0 / float(SCALE))


def get_lr_scheduler(epoch, cfg):
    """Bộ lập lịch tốc độ học tích hợp Warmup + Cosine Decay mịn đồ thị hàm răng cưa"""
    warmup_epochs = cfg.warmup_epochs
    cosine_epochs = 40
    decay_start   = cfg.epochs - cosine_epochs

    if epoch < warmup_epochs:
        return cfg.lr * (epoch + 1) / warmup_epochs
    elif epoch < decay_start:
        return cfg.lr
    else:
        t = (epoch - decay_start) / max(1, cosine_epochs)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))


# =====================================================================
# MAIN
# =====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cfg = MambaConfig()
    cfg.print_summary() # Đã hoạt động bình thường, không còn lỗi crash thuộc tính

    X_tr, Y_tr, X_te, Y_te, pj_mins, pj_maxs = preprocess()

    kw = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(MARSDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=True, **kw)
    test_loader = DataLoader(MARSDataset(X_te, Y_te), batch_size=cfg.batch_size, shuffle=False, **kw)

    from model.mamba_model import build_model
    model = build_model(cfg).to(device)
    model.summary()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_mpjpe = float('inf')

    for epoch in range(cfg.epochs):
        # Thiết lập tốc độ học thích ứng theo từng epoch học tập
        lr = get_lr_scheduler(epoch, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        # Clamp 1 lần sau mỗi epoch — tránh vòng lặp update→clamp→update gây dao động biên
        clamp_weights(model)

        metrics = evaluate(model, test_loader, device, pj_mins, pj_maxs)
        epoch_time = time.time() - start_time
        
        print(f"Ep {epoch+1:3d}/{cfg.epochs} | lr={lr:.1e} | Loss={total_loss:.4f} | "
              f"MAE={metrics['mae_cm']:.3f} cm | RMSE={metrics['rmse_cm']:.3f} cm | "
              f"MPJPE={metrics['mpjpe_cm']:.3f} cm | Time={epoch_time:.1f}s")

        if metrics['mpjpe_cm'] < best_mpjpe:
            best_mpjpe = metrics['mpjpe_cm']

            # Tự động chọn thư mục có quyền ghi thích hợp (Ưu tiên /kaggle/working nếu có)
            output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(output_dir, "best.pt")

            # Thực hiện lưu vào đường dẫn tuyệt đối an toàn
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  → Đạt kỷ lục mới! Đã lưu mô hình tối ưu tại: {save_path} (MPJPE: {best_mpjpe:.3f} cm)")

    print("\n🎉 Quá trình huấn luyện QAT đã hoàn thành thành công!")


if __name__ == "__main__":
    main()