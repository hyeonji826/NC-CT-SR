# E:\LD-CT SR\_scripts_4_wavelet\train_n2n.py
# NS-N2N training with 3D UNet + Transformer hybrid
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import random
import sys
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
from torch.cuda.amp import GradScaler, autocast
from dataset_n2n import NSN2NDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import 3D UNet + Transformer model
from model_3d_unet_trans import UNet3DTransformer
from losses_n2n import HighQualityNSN2NLoss
from utils import (
    EarlyStopping,
    WarmupScheduler,
    cleanup_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
    save_sample_images,
    load_config,
)

# ---------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------
def adjust_loss_weights(criterion, hn_reduction, ln_reduction, config: dict):
    """HN/LN 노이즈 감소율 기반으로 lambda 자동 조정 (config-driven)"""
    if not config["adaptive_weights"]["enabled"]:
        return
        
    avg_red = (hn_reduction + ln_reduction) / 2.0
    
    aw = config["adaptive_weights"]
    under_thresh = aw["under_denoise_threshold"]
    over_thresh = aw["over_smooth_threshold"]

    def clamp(v, lo_hi):
        return max(lo_hi[0], min(lo_hi[1], v))

    if avg_red < under_thresh:
        factor_noise = aw["factor_noise_increase"]
        factor_rc_ic = aw["factor_rc_ic_increase"]
        factor_edge_hf = aw["factor_edge_hf_decrease"]
    elif avg_red > over_thresh:
        factor_noise = aw["factor_noise_decrease"]
        factor_rc_ic = aw["factor_rc_ic_decrease"]
        factor_edge_hf = aw["factor_edge_hf_increase"]
    else:
        return

    criterion.lambda_noise = clamp(criterion.lambda_noise * factor_noise, aw["clamp_noise"])
    criterion.lambda_rc    = clamp(criterion.lambda_rc * factor_rc_ic,  aw["clamp_rc"])
    criterion.lambda_ic    = clamp(criterion.lambda_ic * factor_rc_ic,  aw["clamp_ic"])
    criterion.lambda_edge  = clamp(criterion.lambda_edge * factor_edge_hf, aw["clamp_edge"])
    criterion.lambda_hf    = clamp(criterion.lambda_hf * factor_edge_hf, aw["clamp_hf"])

    print(
        f"[ADJUST] avg_noise_reduction={avg_red:.1f}% → "
        f"λ_rc={criterion.lambda_rc:.3f}, λ_ic={criterion.lambda_ic:.3f}, "
        f"λ_noise={criterion.lambda_noise:.3f}, λ_edge={criterion.lambda_edge:.3f}, "
        f"λ_hf={criterion.lambda_hf:.3f}"
    )

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_roi_noise_hu(
    x_01: torch.Tensor,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
    roi_h: float,
    roi_w: float,
) -> float:
    """ROI 중앙 영역에서 노이즈 레벨 계산 (단일 이미지)"""
    hu_min, hu_max = hu_window
    arr = x_01[0, 0].detach().cpu().numpy()
    H, W = arr.shape

    h_margin = int(H * (1.0 - roi_h) / 2.0)
    w_margin = int(W * (1.0 - roi_w) / 2.0)
    roi = arr[h_margin : H - h_margin, w_margin : W - w_margin]

    roi_hu = roi * (hu_max - hu_min) + hu_min

    body_mask = (roi_hu >= body_hu_range[0]) & (roi_hu <= body_hu_range[1])
    if body_mask.sum() < 100:
        return 0.0

    noise_std_hu = float(roi_hu[body_mask].std())
    return noise_std_hu


def select_hn_ln_slices(
    nc_ct_dir: str,
    hu_window: Tuple[float, float],
    config: dict,
    max_files: int = 50,
    seed: int | None = 0,
) -> List[dict]:
    """Noise가 가장 큰 slice / 가장 작은 slice 하나씩 고정 샘플로 선택."""
    if seed is not None:
        set_seed(seed)

    nc_ct_dir = Path(nc_ct_dir)
    files = sorted(list(nc_ct_dir.glob("*.nii.gz")) + list(nc_ct_dir.glob("*.nii")))
    files = files[:max_files]

    candidates = []

    for path in tqdm(files, desc="Analyzing volumes for HN/LN samples"):
        nii = nib.load(str(path))
        vol = nii.get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        H, W, D = vol.shape
        for z in range(D):
            s = vol[:, :, z]
            s = np.clip(s, hu_window[0], hu_window[1])
            s01 = (s - hu_window[0]) / (hu_window[1] - hu_window[0] + 1e-8)
            t = torch.from_numpy(s01[None, None, ...])

            noise_std_hu = compute_roi_noise_hu(
                t, hu_window, 
                body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                roi_h=config["noise_analysis"]["roi_h"],
                roi_w=config["noise_analysis"]["roi_w"]
            )
            if noise_std_hu <= 0:
                continue

            candidates.append(
                {"file": path, "z": z, "noise_std_hu": noise_std_hu}
            )

    if len(candidates) < 2:
        raise RuntimeError("Not enough valid slices to select HN/LN samples.")

    noises = np.array([c["noise_std_hu"] for c in candidates], dtype=np.float32)
    p10 = float(np.percentile(noises, 10))
    p90 = float(np.percentile(noises, 90))

    hn_candidates = [c for c in candidates if c["noise_std_hu"] >= p90]
    ln_candidates = [c for c in candidates if c["noise_std_hu"] <= p10]
    if not hn_candidates:
        hn_candidates = sorted(candidates, key=lambda x: -x["noise_std_hu"])[:10]
    if not ln_candidates:
        ln_candidates = sorted(candidates, key=lambda x: x["noise_std_hu"])[:10]

    high_noise = max(hn_candidates, key=lambda x: x["noise_std_hu"])
    low_noise = min(ln_candidates, key=lambda x: x["noise_std_hu"])

    print(
        f"Selected HN slice: {high_noise['file'].name} "
        f"(z={high_noise['z']}, noise={high_noise['noise_std_hu']:.1f} HU)"
    )
    print(
        f"Selected LN slice: {low_noise['file'].name} "
        f"(z={low_noise['z']}, noise={low_noise['noise_std_hu']:.1f} HU)"
    )

    return [
        {
            "label": "HN",
            "file": str(high_noise["file"]),
            "z": int(high_noise["z"]),
            "original_noise_hu": float(high_noise["noise_std_hu"]),
        },
        {
            "label": "LN",
            "file": str(low_noise["file"]),
            "z": int(low_noise["z"]),
            "original_noise_hu": float(low_noise["noise_std_hu"]),
        },
    ]


def load_fixed_full_slice(path: str, z: int, hu_window: Tuple[float, float]) -> torch.Tensor:
    """Load 5 consecutive slices as 3D volume for 3D UNet"""
    nii = nib.load(str(path))
    vol = nii.get_fdata().astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    D = vol.shape[2]
    
    # Get 5 adjacent slices (±2 from center)
    slices = []
    for offset in [-2, -1, 0, 1, 2]:
        z_idx = np.clip(z + offset, 0, D - 1)
        slices.append(vol[:, :, z_idx])
    
    # Window and normalize
    def window_norm(s):
        s = np.clip(s, hu_window[0], hu_window[1])
        s01 = (s - hu_window[0]) / (hu_window[1] - hu_window[0] + 1e-8)
        return s01.astype(np.float32)
    
    x_5slices = np.stack([window_norm(s) for s in slices], axis=0)  # (5, H, W)
    t = torch.from_numpy(x_5slices).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 5, H, W)
    return t


def prepare_experiment(config: dict, exp_name: str) -> tuple[Path, Path, Path, SummaryWriter]:
    output_root = Path(config["data"]["output_dir"])
    exp_dir = output_root / exp_name
    ckpt_dir = exp_dir / "ckpts"
    log_dir = exp_dir / "logs"
    sample_dir = exp_dir / "samples"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(exp_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: tensorboard --logdir \"{log_dir}\"")
    return exp_dir, ckpt_dir, sample_dir, writer


# ---------------------------------------------------------------------
# 메인 학습 루프
# ---------------------------------------------------------------------


def train_n2n(config_path: str = "config_n2n.yaml") -> None:
    config = load_config(config_path)
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("[INFO] Initializing dataset...")

    # ----- Load config values -----
    hu_window = tuple(config["preprocessing"]["hu_window"])
    patch_size = int(config["preprocessing"]["patch_size"])
    min_body_fraction = float(config["preprocessing"]["min_body_fraction"])
    match_threshold = float(config["preprocessing"]["match_threshold"])

    # dataset.* 섹션
    lpf_sigma = float(config["dataset"]["lpf_sigma"])
    lpf_median_size = int(config["dataset"]["lpf_median_size"])
    noise_aug_ratio = float(config["dataset"]["noise_aug_ratio"])

    body_hu_range = tuple(config["dataset"]["body_hu_range"])
    noise_roi_margin_ratio = float(config["dataset"]["noise_roi_margin_ratio"])
    noise_tissue_range = tuple(config["dataset"]["noise_tissue_range"])
    noise_default_std = float(config["dataset"]["noise_default_std"])

    # ----- Train dataset (full) -----
    full_dataset = NSN2NDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=hu_window,
        patch_size=patch_size,
        min_body_fraction=min_body_fraction,
        lpf_sigma=lpf_sigma,
        lpf_median_size=lpf_median_size,
        match_threshold=match_threshold,
        noise_aug_ratio=noise_aug_ratio,
        body_hu_range=body_hu_range,
        noise_roi_margin_ratio=noise_roi_margin_ratio,
        noise_tissue_range=noise_tissue_range,
        noise_default_std=noise_default_std,
        mode="train",
    )

    # ----- Train / Val / Test split -----
    val_split = float(config["training"]["val_split"])
    test_split = float(config["training"].get("test_split", 0.1))  # optional

    val_size = int(len(full_dataset) * val_split)
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ----- But validation mode must not apply training augmentation -----
    # Create a proper validation dataset WITHOUT noise aug.
    val_dataset = NSN2NDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=hu_window,
        patch_size=patch_size,
        min_body_fraction=min_body_fraction,
        lpf_sigma=lpf_sigma,
        lpf_median_size=lpf_median_size,
        match_threshold=match_threshold,
        noise_aug_ratio=noise_aug_ratio,  # ignored in val mode
        body_hu_range=body_hu_range,
        noise_roi_margin_ratio=noise_roi_margin_ratio,
        noise_tissue_range=noise_tissue_range,
        noise_default_std=noise_default_std,
        mode="val",
    )

    # test dataset도 동일 방식
    test_dataset = NSN2NDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=hu_window,
        patch_size=patch_size,
        min_body_fraction=min_body_fraction,
        lpf_sigma=lpf_sigma,
        lpf_median_size=lpf_median_size,
        match_threshold=match_threshold,
        noise_aug_ratio=noise_aug_ratio,
        body_hu_range=body_hu_range,
        noise_roi_margin_ratio=noise_roi_margin_ratio,
        noise_tissue_range=noise_tissue_range,
        noise_default_std=noise_default_std,
        mode="val",   # test에서도 noise aug 금지
    )

    # ============ DataLoader 생성 ============

    def make_loader(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(config["training"]["num_workers"]),
            pin_memory=True,
            drop_last=shuffle,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset, shuffle=False)
    test_loader  = make_loader(test_dataset, shuffle=False)

    print(f"[INFO] Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # ----- 3D UNet + Transformer Model -----
    print("[INFO] Initializing 3D UNet + Transformer model...")
    
    model = UNet3DTransformer(
        in_channels=1,
        base_channels=config["model"]["base_channels"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {num_params:,}")
    print(f"[INFO] Input: (B, 1, 5, H, W) → Output: (B, 1, 1, H, W)")
    print(f"[INFO] 3D convolution + 3D attention for superior z-axis context")


    # ----- Loss -----
    criterion = HighQualityNSN2NLoss(
        lambda_rc=config["loss"]["lambda_rc"],
        lambda_ic=config["loss"]["lambda_ic"],
        lambda_noise=config["loss"]["lambda_noise"],
        lambda_edge=config["loss"]["lambda_edge"],
        lambda_hf=config["loss"]["lambda_hf"],
        target_noise_ratio=config["loss"]["target_noise_ratio"],
        min_body_pixels=config["loss"]["min_body_pixels"],
        artifact_grad_factor=config["loss"]["artifact_grad_factor"],
        flat_threshold=config["loss"]["flat_threshold"],
    )


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

    # ----- WarmupScheduler: 타입 안전 캐스팅 -----
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    warmup_lr_cfg = config["training"].get("warmup_lr", 1e-6)
    base_lr_cfg = config["training"]["learning_rate"]

    warmup_lr = float(warmup_lr_cfg)
    base_lr = float(base_lr_cfg)

    print(
        f"[INFO] WarmupScheduler: warmup_epochs={warmup_epochs}, "
        f"warmup_lr={warmup_lr}, base_lr={base_lr}"
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

    use_amp = bool(config["training"].get("use_amp", False))
    scaler = GradScaler(enabled=use_amp)


    # ----- 실험 디렉토리 -----
    exp_name = config["training"].get("exp_name", "ns_n2n_main")
    exp_dir, ckpt_dir, sample_dir, writer = prepare_experiment(config, exp_name)

    # ----- Resume -----
    start_epoch = 1
    best_val_loss = float("inf")

    if config["training"].get("resume"):
        resume_path = Path(config["training"]["resume"])
        if resume_path.exists():
            start_epoch, loaded_val = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1
            best_val_loss = loaded_val
            print(f"Resumed from epoch {start_epoch-1}, best val loss {best_val_loss:.4f}")
        else:
            print(f"WARNING: resume path not found: {resume_path}")

    # ----- HN/LN 샘플 고정 -----
    sample_slices: List[dict] | None = None
    if config.get("validation", {}).get("save_samples", True):
        sample_slices = select_hn_ln_slices(
            nc_ct_dir=config["data"]["nc_ct_dir"],
            hu_window=hu_window,
            config=config,
            max_files=config["validation"].get("sample_scan_files", 50),
            seed=seed,
        )

    print("\n==================== Start Training (NS-N2N) ====================\n")

    global_step = 0
    for epoch in range(start_epoch, config["training"]["num_epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        train_losses: List[float] = []
        train_detail = {"flat": [], "rc": [], "ic": [], "noise": [], "edge": [], "hf": [], 
                        "flat_ratio": [], "edge_ratio": []}

        train_W_sum = 0.0
        train_W_count = 0

        if warmup.is_warmup():
            warmup.step()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")

        for batch in pbar:
            x_i = batch["x_i"].to(device, non_blocking=True)
            x_i_aug = batch["x_i_aug"].to(device, non_blocking=True)
            x_ip1 = batch["x_ip1"].to(device, non_blocking=True)
            x_mid = batch["x_mid"].to(device, non_blocking=True)
            W = batch["W"].to(device, non_blocking=True)
            noise_synthetic = batch["noise_synthetic"].to(device, non_blocking=True)

            batch_W_mean = float(W.mean().item())
            train_W_sum += batch_W_mean
            train_W_count += 1

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                # 3D UNet forward: (B, 1, 5, H, W) → (B, 1, 1, H, W)
                output = model(x_i_aug)
                
                # Squeeze depth dimension: (B, 1, 1, H, W) → (B, 1, H, W)
                noise_pred = output.squeeze(2)
                            
                loss, loss_dict = criterion(
                    noise_pred=noise_pred,
                    x_i=x_i,
                    x_i_aug=x_i_aug,
                    x_ip1=x_ip1,
                    x_mid=x_mid,
                    W=W,
                    noise_synthetic=noise_synthetic,
                )


            if use_amp:
                scaler.scale(loss).backward()
                if config["training"].get("gradient_clip", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config["training"].get("gradient_clip", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip"]
                    )
                optimizer.step()

            train_losses.append(float(loss.item()))
            for k in train_detail.keys():
                train_detail[k].append(loss_dict.get(k, 0.0))

            global_step += 1
            writer.add_scalar("Train/total_loss", loss.item(), global_step)
            for k in ("flat", "rc", "ic", "noise", "edge", "hf", "flat_ratio", "edge_ratio"):
                if k in loss_dict:
                    writer.add_scalar(f"Train/{k}", loss_dict[k], global_step)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                noise=f"{loss_dict.get('noise', 0):.4f}",
            )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        avg_train_detail = {k: (float(np.mean(v)) if v else 0.0)
                            for k, v in train_detail.items()}
        avg_train_W = train_W_sum / train_W_count if train_W_count > 0 else 0.0
        
        # ---------------- Validation ----------------
        model.eval()
        val_losses: List[float] = []

        val_W_sum = 0.0
        val_W_count = 0

        with torch.no_grad():
            for batch in val_loader:
                x_i = batch["x_i"].to(device, non_blocking=True)            
                x_i_aug = batch["x_i_aug"].to(device, non_blocking=True)
                x_ip1 = batch["x_ip1"].to(device, non_blocking=True)
                x_mid = batch["x_mid"].to(device, non_blocking=True)
                W = batch["W"].to(device, non_blocking=True)
                noise_synthetic = batch["noise_synthetic"].to(device, non_blocking=True)

                batch_W_mean = float(W.mean().item())
                val_W_sum += batch_W_mean
                val_W_count += 1

                with autocast(enabled=use_amp):
                    # 3D UNet forward
                    output = model(x_i_aug)
                    denoised_pred = output.squeeze(2)  # (B, 1, 1, H, W) → (B, 1, H, W)
                    
                    val_loss, _ = criterion(
                        noise_pred=denoised_pred,
                        x_i=x_i,
                        x_i_aug=x_i_aug,
                        x_ip1=x_ip1,
                        x_mid=x_mid,
                        W=W,
                        noise_synthetic=noise_synthetic,
                    )


                val_losses.append(float(val_loss.item()))

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_val_W = val_W_sum / val_W_count if val_W_count > 0 else 0.0

        scheduler.step()
        writer.add_scalar("Epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/val_loss", avg_val_loss, epoch)
        writer.add_scalar("Epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 🔹 W coverage 로그
        writer.add_scalar("Epoch/train_W_mean", avg_train_W, epoch)
        writer.add_scalar("Epoch/val_W_mean", avg_val_W, epoch)

        # ---------------- 로그 ----------------
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"Train loss : {avg_train_loss:.4f}")
        print(f"  Flat     : {avg_train_detail['flat']:.4f} (flat_ratio: {avg_train_detail.get('flat_ratio', 0):.3f})")
        print(f"  RC       : {avg_train_detail['rc']:.4f}")
        print(f"  IC       : {avg_train_detail['ic']:.4f}")
        print(f"  Noise    : {avg_train_detail['noise']:.4f}")
        print(f"  Edge     : {avg_train_detail['edge']:.4f} (edge_ratio: {avg_train_detail.get('edge_ratio', 0):.3f})")
        print(f"  HF       : {avg_train_detail['hf']:.4f}")
        print(f"Val loss   : {avg_val_loss:.4f}")
        print(f"Train W    : {avg_train_W:.4f}")
        print(f"Val W      : {avg_val_W:.4f}")
        print(f"LR         : {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}\n")

        # ---------------- Checkpoint ----------------
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if is_best or (epoch % config["training"]["save_interval"] == 0):
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.pth"
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, avg_val_loss, ckpt_path, is_best=is_best
            )
            cleanup_old_checkpoints(
                ckpt_dir, keep_last_n=config["training"]["keep_last_n"]
            )
            
        # ---------------- 샘플 PNG (고정 HN/LN) ----------------
        if sample_slices is not None and \
           epoch % config["training"]["sample_interval"] == 0:
            fixed_samples: List[torch.Tensor] = []
            denoised_list: List[torch.Tensor] = []
            metrics_list: List[dict] = []

            for info in sample_slices:
                noisy_5slice = load_fixed_full_slice(info["file"], info["z"], hu_window).to(device)
                with torch.no_grad():
                    denoised = model(noisy_5slice)  # (1, 1, 1, H, W)
                    denoised = denoised.squeeze(2)  # (1, 1, H, W)
                
                # Extract center slice from input for visualization
                noisy_center = noisy_5slice[:, :, 2, :, :]  # (1, 1, H, W) - center of 5 slices
                
                fixed_samples.append(noisy_center.cpu())
                denoised_list.append(denoised.cpu())

                noisy_hu_std = compute_roi_noise_hu(
                    noisy_center.cpu(), hu_window,
                    body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                    roi_h=config["noise_analysis"]["roi_h"],
                    roi_w=config["noise_analysis"]["roi_w"]
                )
                den_hu_std = compute_roi_noise_hu(
                    denoised.cpu(), hu_window,
                    body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                    roi_h=config["noise_analysis"]["roi_h"],
                    roi_w=config["noise_analysis"]["roi_w"]
                )
                metrics_list.append(
                    {
                        "label": info["label"],
                        "file": info["file"],
                        "original_noise_hu": float(noisy_hu_std),
                        "estimated_noise_hu": float(den_hu_std),
                    }
                )

            noisy_batch = torch.cat(fixed_samples, dim=0)
            denoised_batch = torch.cat(denoised_list, dim=0)
            png_path = sample_dir / f"epoch_{epoch:03d}.png"
            save_sample_images(noisy_batch, denoised_batch, png_path, epoch, metrics=metrics_list)
            print(f"Saved sample: {png_path.name}")

            # ---- HN/LN 노이즈 감소율 계산해서 loss weight 조정 ----
            hn_red = ln_red = None
            for m in metrics_list:
                lbl = m.get("label", "")
                orig = m.get("original_noise_hu", 0.0)
                deno = m.get("estimated_noise_hu", 0.0)
                if orig > 0:
                    red = (orig - deno) / orig * 100.0
                else:
                    red = 0.0
                if lbl == "HN":
                    hn_red = red
                elif lbl == "LN":
                    ln_red = red

            if hn_red is not None and ln_red is not None:
                adjust_loss_weights(criterion, hn_red, ln_red, config)

            # 텐서보드 로그는 기존 것 그대로
            if metrics_list:
                for m in metrics_list:
                    lbl = m.get("label", "unknown")
                    orig = m.get("original_noise_hu", 0.0)
                    deno = m.get("estimated_noise_hu", 0.0)
                    writer.add_scalar(f"Sample/{lbl}_noise_before", orig, epoch)
                    writer.add_scalar(f"Sample/{lbl}_noise_after", deno, epoch)
                    writer.add_scalar(f"Sample/{lbl}_noise_reduction", orig - deno, epoch)


        # ---------------- Early stopping ----------------
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch}, best val loss {best_val_loss:.4f}")
            break

    writer.close()
    print("\n==================== Training Done ====================")
    print(f"Experiment dir: {exp_dir}")
    print(f"Best val loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NS-N2N CT denoising (SwinIR)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_n2n.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    train_n2n(args.config)