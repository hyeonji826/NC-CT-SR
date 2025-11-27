# E:\LD-CT SR\_scripts_4_wavelet\train_n2n.py
# 2.5D Neighboring-Slice Supervised Training (NS-N2N 스타일)
#
# - Dataset: NCCTDenoiseDataset (dataset_n2n.py)
#   -> input : [3, H, W] (z-1, z, z+1)
#   -> target: [1, H, W] (z)
# - Model : SwinIR (in_chans=3, img_range=1.0)
# - Loss  : SupervisedWaveletLoss (MSE + λ * Wavelet)
#
# 기존 Neighbor2Neighbor + adaptive wavelet + edge loss는 사용하지 않는다.

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# SwinIR path
sys.path.insert(0, r"E:\LD-CT SR\_externals\SwinIR")
from models.network_swinir import SwinIR  # type: ignore

from dataset_n2n import NCCTDenoiseDataset
from losses_n2n import SupervisedWaveletLoss, WaveletSparsityPrior
from utils import (
    EarlyStopping,
    WarmupScheduler,
    cleanup_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
    save_sample_images,
)


# -------------------------------------------------------------------------
# HN/LN 대표 슬라이스 선택 (전체 CT 중 noise 상대 극단) – 기존 함수 그대로 사용
# -------------------------------------------------------------------------
def load_fixed_full_slices(nc_ct_dir, hu_window, seed=None):
    from scipy import ndimage
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nc_ct_dir = Path(nc_ct_dir)
    files = sorted(list(nc_ct_dir.glob("*.nii.gz")))
    candidates = []

    print(f"\n🔍 Analyzing slices (RELATIVE noise-based selection, seed={seed})...")
    max_files = min(50, len(files))

    for file_idx, file_path in enumerate(files[:max_files]):
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            D = volume.shape[2]

            slice_indices = [D // 5, D // 3, D // 2, 2 * D // 3, 4 * D // 5]
            for slice_idx in slice_indices:
                if slice_idx >= D:
                    continue

                slice_2d = volume[:, :, slice_idx]
                h, w = slice_2d.shape

                center_ratio_w = 0.55
                center_ratio_h = 0.60
                margin_h = int(h * (1 - center_ratio_h) / 2)
                margin_w = int(w * (1 - center_ratio_w) / 2)

                center_slice = slice_2d[margin_h : h - margin_h, margin_w : w - margin_w]

                tissue_mask = (center_slice > -100) & (center_slice < 100)
                if tissue_mask.sum() < 1000:
                    continue

                noise_std = center_slice[tissue_mask].std()

                edge_x = ndimage.sobel(center_slice, axis=0)
                edge_y = ndimage.sobel(center_slice, axis=1)
                edge_mag = np.sqrt(edge_x**2 + edge_y**2)
                edge_score = edge_mag[tissue_mask].mean()

                candidates.append(
                    {
                        "file_path": file_path,
                        "slice_idx": slice_idx,
                        "noise_std": noise_std,
                        "edge_score": edge_score,
                        "slice_2d": center_slice,
                    }
                )
        except Exception:
            continue

        if file_idx >= 49:
            break

    if len(candidates) < 2:
        raise ValueError(f"Not enough valid slices found (only {len(candidates)})")

    noise_values = [c["noise_std"] for c in candidates]
    p10 = np.percentile(noise_values, 10)
    p50 = np.percentile(noise_values, 50)
    p90 = np.percentile(noise_values, 90)

    print(f"  Scanned {max_files}/{len(files)} files...")
    print(f"   Noise distribution: 10th={p10:.1f}, median={p50:.1f}, 90th={p90:.1f} HU")

    hn_candidates = [c for c in candidates if c["noise_std"] >= p90]
    if len(hn_candidates) == 0:
        hn_candidates = sorted(candidates, key=lambda x: x["noise_std"], reverse=True)[:10]
    high_noise = max(hn_candidates, key=lambda x: x["noise_std"])

    ln_candidates = [c for c in candidates if c["noise_std"] <= p10]
    if len(ln_candidates) < 5:
        ln_candidates = sorted(candidates, key=lambda x: x["noise_std"])[: max(5, len(candidates) // 7)]
    low_noise = max(ln_candidates, key=lambda x: x["edge_score"])

    hn_pct = (
        sum(1 for c in candidates if c["noise_std"] > high_noise["noise_std"]) / len(candidates) * 100
    )
    ln_pct = (
        sum(1 for c in candidates if c["noise_std"] > low_noise["noise_std"]) / len(candidates) * 100
    )

    print("✅ Selected 2 representative slices (RELATIVE extremes):")
    print(
        f"   [HN] {high_noise['file_path'].name} slice {high_noise['slice_idx']}  "
        f"Noise: {high_noise['noise_std']:.1f} HU (top {hn_pct:.1f}%)  "
        f"Edge: {high_noise['edge_score']:.1f}"
    )
    print(
        f"   [LN] {low_noise['file_path'].name} slice {low_noise['slice_idx']}  "
        f"Noise: {low_noise['noise_std']:.1f} HU (bottom {ln_pct:.1f}%)  "
        f"Edge: {low_noise['edge_score']:.1f}"
    )
    print(
        f"   Noise ratio (HN/LN): "
        f"{high_noise['noise_std'] / max(low_noise['noise_std'], 1e-6):.2f}x"
    )

    slices = []
    slice_info = []

    for label, cand in [("HN", high_noise), ("LN", low_noise)]:
        slice_2d = cand["slice_2d"]
        # hu_window 적용 + [0,1] 정규화
        slice_2d = np.clip(slice_2d, hu_window[0], hu_window[1])
        slice_2d = (slice_2d - hu_window[0]) / (hu_window[1] - hu_window[0])
        slice_2d = slice_2d.astype(np.float32)

        slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        slices.append(slice_tensor)

        slice_info.append(
            {
                "label": label,
                "noise_std_hu": float(cand["noise_std"]),
                "edge_score": float(cand.get("edge_score", 0.0)),
                "file": cand["file_path"].name,
                "slice_idx": int(cand["slice_idx"]),
            }
        )

    return slices, slice_info

def compute_roi_noise_hu(x_01: torch.Tensor,
                         hu_window,
                         body_hu_range=(-100, 100),
                         roi_h=0.7,
                         roi_w=0.5) -> float:
    """
    x_01 : [1,1,H,W]  0~1 정규화된 CT 슬라이스
    hu_window : (hu_min, hu_max)  ex) (-160, 240)
    body_hu_range : 조직 마스크 범위 (기본 -100~100 HU)
    roi_h, roi_w : ROI 크기 비율 (70% x 50%)
    반환값 : ROI 내 노이즈 표준편차 (HU 단위)
    """
    hu_min, hu_max = hu_window
    arr = x_01[0, 0].detach().cpu().numpy()
    H, W = arr.shape

    # 중심 ROI (위/아래 15%, 좌/우 25% 잘라내기 → 70% x 50%)
    top = int(H * (1 - roi_h) / 2)
    bottom = H - top
    left = int(W * (1 - roi_w) / 2)
    right = W - left
    roi = arr[top:bottom, left:right]

    # 0~1 → HU
    hu = roi * (hu_max - hu_min) + hu_min

    # 조직 마스크
    mask = (hu > body_hu_range[0]) & (hu < body_hu_range[1])
    vals = hu[mask]
    # 조직 픽셀이 너무 적으면 ROI 전체 사용 (fallback)
    if vals.size < 500:
        vals = hu

    return float(vals.std())


# -------------------------------------------------------------------------
# Config / arg parsing
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="_scripts_4_wavelet/config/config_n2n.yaml",
    )
    parser.add_argument("--exp", type=str, default="debug")
    return parser.parse_args()


def load_config(path):
    from utils import load_yaml_config

    return load_yaml_config(path)


def setup_experiment(config, exp_name):
    base_dir = Path(config["data"]["output_dir"])
    exp_dir = base_dir / exp_name
    ckpt_dir = exp_dir / "ckpts"
    log_dir = exp_dir / "logs"
    sample_dir = exp_dir / "samples"

    for d in [exp_dir, ckpt_dir, log_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # config 저장
    import yaml

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard: tensorboard --logdir={log_dir}")
    return exp_dir, ckpt_dir, sample_dir, writer


# -------------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------------
def train_n2n():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------------- Dataset ----------------
    print("\n📂 Loading NC-CT dataset (2.5D NS-N2N)...")
    full_dataset = NCCTDenoiseDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=tuple(config["preprocessing"]["hu_window"]),
        patch_size=config["preprocessing"]["patch_size"],
        mode="train",
    )

    val_size = int(len(full_dataset) * config["training"]["val_split"])
    test_size = int(len(full_dataset) * config["training"].get("test_split", 0.1))
    train_size = len(full_dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    print(f"   Train/Val/Test sizes: {train_size}/{val_size}/{test_size}")

    # ---------------- Experiment dirs ----------------
    exp_dir, ckpt_dir, sample_dir, writer = setup_experiment(config, args.exp)

    # ---------------- Model ----------------
    print("\n🧱 Building SwinIR model...")
    # NS-N2N: 입력 채널을 3으로 강제
    config["model"]["in_chans"] = 3

    model = SwinIR(
        upscale=config["model"]["upscale"],
        in_chans=config["model"]["in_chans"],
        img_size=config["model"]["img_size"],
        window_size=config["model"]["window_size"],
        img_range=config["model"]["img_range"],
        depths=config["model"]["depths"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        upsampler=config["model"]["upsampler"],
        resi_connection=config["model"]["resi_connection"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters   : {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Pretrained weights (기존과 동일)
    pretrained_path = config["model"].get("pretrained_path", None)
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if pretrained_path.exists():
            state_dict = torch.load(pretrained_path, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
            model.load_state_dict(state_dict, strict=False)
            print(f"   Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"   ⚠️ Pretrained path not found: {pretrained_path}")

    # ---------------- Loss / Optim / Scheduler ----------------
    criterion = SupervisedWaveletLoss(
        wavelet_weight=config["training"]["wavelet_weight"],
        wavelet_threshold=config["training"]["wavelet_threshold"],
        wavelet_levels=config["training"]["wavelet_levels"],
        hu_window=tuple(config["preprocessing"]["hu_window"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"],
    )

    warmup = WarmupScheduler(
        optimizer,
        warmup_epochs=config["training"].get("warmup_epochs", 0),
        warmup_lr=config["training"].get("warmup_lr", 1e-6),
        base_lr=config["training"]["learning_rate"],
    )

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"].get("early_stopping_delta", 0.0),
    )

    scaler = GradScaler(enabled=config["training"].get("use_amp", False))
    use_amp = config["training"].get("use_amp", False)

    start_epoch = 1
    best_val_loss = float("inf")

    if config["training"].get("resume"):
        resume_path = Path(config["training"]["resume"])
        if resume_path.exists():
            start_epoch, loaded_val = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1
            best_val_loss = loaded_val
            print(f"\nResumed from epoch {start_epoch-1}, best val loss {best_val_loss:.4f}")
        else:
            print(f"\nWARNING: resume path not found: {resume_path}")

    print("\n==================== Start Training (NS-N2N) ====================\n")

    global_step = 0

    for epoch in range(start_epoch, config["training"]["num_epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        train_losses = []
        train_loss_details = {k: [] for k in ["base", "wavelet_weighted", "total", "estimated_noise"]}

        if warmup.is_warmup():
            warmup.step()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)    # [B,3,H,W]
            targets = targets.to(device, non_blocking=True)  # [B,1,H,W]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss, loss_dict = criterion(outputs, targets)
                scaler.scale(loss).backward()
                if config["training"]["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["training"]["gradient_clip"],
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss, loss_dict = criterion(outputs, targets)
                loss.backward()
                if config["training"]["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["training"]["gradient_clip"],
                    )
                optimizer.step()

            train_losses.append(loss.item())
            for k in train_loss_details.keys():
                if k in loss_dict:
                    train_loss_details[k].append(loss_dict[k])

            global_step += 1
            writer.add_scalar("Train/total_loss", loss.item(), global_step)
            if "base" in loss_dict:
                writer.add_scalar("Train/base", loss_dict["base"], global_step)
            if "wavelet_weighted" in loss_dict:
                writer.add_scalar("Train/wavelet", loss_dict["wavelet_weighted"], global_step)
            if "estimated_noise" in loss_dict:
                writer.add_scalar("Train/estimated_noise", loss_dict["estimated_noise"], global_step)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        avg_train_loss = float(np.mean(train_losses))
        avg_train_details = {
            k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in train_loss_details.items()
        }

        # ---------------- Validation ----------------
        model.eval()
        val_losses = []
        val_details = {k: [] for k in ["base", "wavelet_weighted", "total", "estimated_noise"]}

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss, loss_dict = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss, loss_dict = criterion(outputs, targets)

                val_losses.append(loss.item())
                for k in val_details.keys():
                    if k in loss_dict:
                        val_details[k].append(loss_dict[k])

        avg_val_loss = float(np.mean(val_losses))
        avg_val_details = {
            k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in val_details.items()
        }

        scheduler.step()
        writer.add_scalar("Epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/val_loss", avg_val_loss, epoch)
        writer.add_scalar("Epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # ---------------- Console Summary ----------------
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"Train loss : {avg_train_loss:.4f}")
        print(f"Val loss   : {avg_val_loss:.4f}")
        if avg_train_details["base"] != 0:
            print(f"  Base     : {avg_train_details['base']:.4f}")
        if avg_train_details["wavelet_weighted"] != 0:
            print(f"  Wavelet  : {avg_train_details['wavelet_weighted']:.4f}")
        if avg_train_details["estimated_noise"] > 0:
            print(f"  Noise    : {avg_train_details['estimated_noise']*400:.1f} HU")
        print(f"LR         : {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}\n")

        # ---------------- Checkpoint ----------------
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if epoch % config["training"]["save_interval"] == 0 or is_best:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.pth"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                avg_val_loss,
                ckpt_path,
                is_best=is_best,
            )
            cleanup_old_checkpoints(ckpt_dir, keep_last_n=config["training"]["keep_last_n"])

        # ---------------- 샘플 HN/LN 이미지 저장 ----------------
        if config["validation"].get("save_samples", True) and (
            epoch % config["training"]["sample_interval"] == 0 or epoch == 1
        ):
            sample_seed = epoch // 10
            fixed_samples, slice_info = load_fixed_full_slices(
                config["data"]["nc_ct_dir"],
                config["preprocessing"]["hu_window"],
                seed=sample_seed,
            )

            hu_window = tuple(config["preprocessing"]["hu_window"])

            model.eval()
            with torch.no_grad():
                denoised_list = []
                metrics_list = []

                for i, single_sample in enumerate(fixed_samples):
                    # single_sample: [1,1,H,W] (noisy, 0~1 정규화 상태)
                    single_noisy = single_sample.to(device)

                    # 2.5D 입력 스택: 중앙 슬라이스를 3번 복제 (시각화용)
                    center_2d = single_noisy[0, 0]  # [H,W]
                    stack = torch.stack([center_2d, center_2d, center_2d], dim=0).unsqueeze(0)  # [1,3,H,W]

                    if use_amp:
                        with autocast():
                            single_denoised = model(stack)
                    else:
                        single_denoised = model(stack)

                    single_denoised = torch.clamp(single_denoised, 0.0, 1.0)

                    # wavelet/MSE는 중앙 채널 하나만 쓰지만,
                    # 시각화/노이즈 계산도 1채널만 사용
                    den_1ch = single_denoised[:, 1:2, ...] if single_denoised.shape[1] > 1 else single_denoised

                    denoised_list.append(den_1ch)

                    # === ROI 기반 노이즈 (HU) 계산 ===
                    #   - noisy: 학습 전 LD
                    #   - denoised: 모델 출력
                    noisy_hu_std = compute_roi_noise_hu(single_noisy, hu_window)
                    den_hu_std = compute_roi_noise_hu(den_1ch, hu_window)

                    m = {
                        "label": slice_info[i].get("label", f"Sample {i+1}"),
                        "file": slice_info[i].get("file", "unknown"),
                        "original_noise_hu": float(noisy_hu_std),   # 재계산 (slice_info 값 대신)
                        "estimated_noise_hu": float(den_hu_std),    # 모델 적용 후 ROI 노이즈
                    }
                    metrics_list.append(m)

                noisy_batch = torch.cat(fixed_samples, dim=0).to(device)  # [2,1,H,W]
                denoised_batch = torch.cat(denoised_list, dim=0)          # [2,1,H,W]

                if len(metrics_list) >= 2:
                    print(f"\n📊 Sample Metrics (Epoch {epoch}):")
                    hn, ln = metrics_list[0], metrics_list[1]
                    print(
                        f"   HN: noise={hn['estimated_noise_hu']:.1f} HU "
                        f"(orig={hn['original_noise_hu']:.1f} HU)"
                    )
                    print(
                        f"   LN: noise={ln['estimated_noise_hu']:.1f} HU "
                        f"(orig={ln['original_noise_hu']:.1f} HU)"
                    )

                save_sample_images(
                    noisy_batch,
                    denoised_batch,
                    sample_dir / f"epoch_{epoch}.png",
                    epoch,
                    metrics=metrics_list if metrics_list else None,
                )
            model.train()

        # ---------------- Early stopping ----------------
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}, best val loss {best_val_loss:.4f}")
            break   

    writer.close()
    print("\n==================== Training Done ====================")
    print(f"Experiment dir: {exp_dir}")
    print(f"Best val loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    train_n2n()
