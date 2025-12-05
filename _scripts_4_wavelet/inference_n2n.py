# inference_n2n.py
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset_n2n import NSN2NDataset
from model_3d_unet_trans import UNet3DTransformer
from utils import compute_noise_hu, load_checkpoint, apply_window
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_inference_image(origin_img, den_img, origin_hu, den_hu, hu_window, out_path):
    """
    origin_img, den_img : (1,1,H,W) tensor, 0~1 정규화
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    origin = origin_img[0,0].cpu().numpy()
    den    = den_img[0,0].cpu().numpy()

    # 1) 0~1 → HU 변환
    hu_min, hu_max = hu_window
    origin_hu_img = origin * (hu_max - hu_min) + hu_min
    den_hu_img    = den    * (hu_max - hu_min) + hu_min

    # 2) clinical window (학습 때랑 동일하게)
    origin_win = apply_window(origin_hu_img, wl=40, ww=400)
    den_win    = apply_window(den_hu_img,    wl=40, ww=400)

    # 3) 회전도 학습쪽과 맞추고 싶으면
    origin_win = np.rot90(origin_win, k=1)
    den_win    = np.rot90(den_win,    k=1)

    # 4) 같은 스케일로 표시 (vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(origin_win, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Origin - {origin_hu:.1f} HU")
    axes[0].axis("off")

    axes[1].imshow(den_win, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Denoised - {den_hu:.1f} HU")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def build_model(config, device):
    model = UNet3DTransformer(
        in_channels=1,
        base_channels=config["model"]["base_channels"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    return model


def create_inference_loader(config):
    """원본 NC-CT에서 synthetic noise/flip 없이 쓰는 dataloader"""
    dataset = NSN2NDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=tuple(config["preprocessing"]["hu_window"]),
        patch_size=0,  # ★ 풀 슬라이스 그대로
        min_body_fraction=config["preprocessing"]["min_body_fraction"],
        lpf_sigma=config["dataset"]["lpf_sigma"],
        lpf_median_size=config["dataset"]["lpf_median_size"],
        match_threshold=config["preprocessing"]["match_threshold"],
        noise_aug_ratio=config["dataset"]["noise_aug_ratio"],  # val 모드라 실제론 안 씀
        body_hu_range=tuple(config["dataset"]["body_hu_range"]),
        noise_roi_margin_ratio=config["dataset"]["noise_roi_margin_ratio"],
        noise_tissue_range=tuple(config["dataset"]["noise_tissue_range"]),
        noise_default_std=config["dataset"]["noise_default_std"],
        mode="val",  # ★ 여기 중요: synthetic noise / flip OFF
    )

    print(f"[INFER] Loaded {len(dataset)} slice pairs for inference")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,        # noise metric 비교 때문에 1장씩 보는 게 안전
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader


def run_inference(config_path, ckpt_path, max_samples=10):
    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    config_path = Path(config_path)
    ckpt_path = Path(ckpt_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] Using device: {device}")

    # -------------------------------------------------
    # Model + checkpoint
    # -------------------------------------------------
    model = build_model(config, device)
    load_checkpoint(ckpt_path, model)  # utils.load_checkpoint
    model.to(device)
    model.eval()
    print(f"[INFER] Loaded checkpoint from: {ckpt_path}")

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    loader = create_inference_loader(config)

    hu_window = tuple(config["preprocessing"]["hu_window"])
    body_hu_range = tuple(config["dataset"]["body_hu_range"])

    # -------------------------------------------------
    # Run through a few samples
    # -------------------------------------------------
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_samples:
                break

            x_i = batch["x_i"].to(device)          # (1,1,H,W) 원본
            x_i_aug = batch["x_i_aug"].to(device)  # (1,1,5,H,W) 5-slice 원본 volume

            # 모델 통과
            denoised, noise_pred = model(x_i_aug)  # (1,1,1,H,W)

            # compute_noise_hu는 (1,1,H,W) 기준이니까 reshape
            denoised_2d = denoised[:, :, 0, :, :]  # (1,1,H,W)

            origin_hu = compute_noise_hu(
                x_i, hu_window, body_hu_range, use_highpass=True, debug=False
            )
            denoised_hu = compute_noise_hu(
                denoised_2d, hu_window, body_hu_range, use_highpass=True, debug=False
            )

            reduction = (
                100.0 * (origin_hu - denoised_hu) / origin_hu
                if origin_hu > 1e-6
                else 0.0
            )

            print(
                f"[{idx:03d}] Origin: {origin_hu:.1f} HU → "
                f"Denoised: {denoised_hu:.1f} HU "
                f"({reduction:+.1f}% )"
            )

            results.append((origin_hu, denoised_hu, reduction))

    if not results:
        print("[INFER] No samples processed.")
        return

    # 전체 평균도 찍어주기
    import numpy as np

    origins, denoised_vals, reductions = zip(*results)
    print("\n[INFER] Summary (first %d samples):" % len(results))
    print(
        f"  Origin avg noise:   {np.mean(origins):.2f} HU\n"
        f"  Denoised avg noise: {np.mean(denoised_vals):.2f} HU\n"
        f"  Avg reduction:      {np.mean(reductions):.1f}%"
    )

    save_dir = Path("inference_results")
    save_path = save_dir / f"sample_{idx:03d}.png"
    save_inference_image(
    x_i,                 # (1,1,H,W)
    denoised_2d,         # (1,1,H,W)
    origin_hu,
    denoised_hu,
    hu_window,
    out_path=save_dir / f"sample_{idx:03d}.png",
    )
    print(f"[SAVE] Saved sample to {save_path}")


if __name__ == "__main__":
    # 예시 실행:
    # python inference_n2n.py --config _scripts_4_wavelet/config/config_n2n.yaml
    #                         --ckpt  E:/LD-CT SR/Outputs/ns_n2n_experiments/residual_denoising_v1/checkpoints/best_model.pth
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_n2n.yaml",
        help="Path to config_n2n.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to best_model.pth",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Number of slice pairs to evaluate",
    )

    args = parser.parse_args()
    run_inference(args.config, args.ckpt, max_samples=args.max_samples)
