"""
utils.py - Swin-N2NLDM 보조 도구 및 추론 엔진
Float32 체크포인트, 메트릭, 역정규화, CT 윈도잉, 추론 파이프라인
"""

import os
import logging
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from dataset import HU_MIN, HU_MAX
from skimage.transform import iradon


# ============================================================
# 로거 설정
# ============================================================

def setup_logger(log_dir: str, name: str = 'SwinN2NLDM') -> logging.Logger:
    """학습 로그를 파일과 콘솔에 동시 출력하는 로거."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 파일 핸들러
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ============================================================
# Float32 Checkpoint 저장/로드
# ============================================================

def save_checkpoint(state: dict, filepath: str) -> None:
    """
    Float32 정밀도를 유지하며 체크포인트를 저장한다.
    학습 재개 및 추론에 필요한 모든 상태를 보존.
    """
    # float32 정밀도 보장
    for key, val in state.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    val[k] = v.float()

    torch.save(state, filepath)


def load_checkpoint(filepath: str, device: torch.device = torch.device('cpu')
                    ) -> dict:
    """Float32 체크포인트를 로드한다."""
    state = torch.load(filepath, map_location=device, weights_only=False)
    return state


def load_denoiser_checkpoint(model, ckpt_path: str,
                              device: torch.device = torch.device('cpu'),
                              strict: bool = False):
    """
    Pretrain 체크포인트를 디노이저 모델에 로드.
    다양한 체크포인트 형식을 자동 감지 (model_state_dict, model, state_dict, raw).

    Returns:
        (missing_keys, unexpected_keys, epoch)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = None
        for key in ['model_state_dict', 'model', 'state_dict']:
            if key in ckpt:
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    epoch = ckpt.get('epoch', None) if isinstance(ckpt, dict) else None
    return missing, unexpected, epoch


# ============================================================
# 영상 품질 메트릭: PSNR, SSIM
# ============================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 2.0) -> float:
    """
    Peak Signal-to-Noise Ratio 계산.
    [-1, 1] 범위 기준 data_range=2.0.
    """
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return float('inf')
    psnr = 10.0 * np.log10(data_range ** 2 / mse)
    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    """
    Structural Similarity Index 계산.
    float32 정밀도 유지를 위해 순수 텐서 연산으로 구현.
    """
    device = pred.device

    # 가우시안 윈도우
    coords = torch.arange(window_size, dtype=torch.float32,
                           device=device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad)
    mu_y = F.conv2d(target, window, padding=pad)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred ** 2, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(target ** 2, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean().item()


# ============================================================
# Edge Loss (Sobel 기반)
# ============================================================

def get_sobel_kernel(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sobel 커널 반환 (x, y 방향)."""
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def compute_edge_map(tensor: torch.Tensor) -> torch.Tensor:
    """
    Sobel 필터로 엣지 맵 계산.
    입력: (B, 1, H, W) 텐서
    출력: (B, 1, H, W) 엣지 강도 맵
    """
    sobel_x, sobel_y = get_sobel_kernel(tensor.device)

    # Sobel 필터 적용
    edge_x = F.conv2d(tensor, sobel_x, padding=1)
    edge_y = F.conv2d(tensor, sobel_y, padding=1)

    # 엣지 강도 (magnitude)
    edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    return edge_mag


def compute_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Edge Loss: 출력과 타겟의 엣지 맵 차이 (L1).
    장벽, 경계 등 고주파 성분 보존을 강제.

    Args:
        pred: 모델 출력 (B, 1, H, W)
        target: 타겟 이미지 (B, 1, H, W)

    Returns:
        edge_loss: 스칼라 텐서
    """
    pred_edge = compute_edge_map(pred)
    target_edge = compute_edge_map(target)

    edge_loss = F.l1_loss(pred_edge, target_edge)

    return edge_loss


def compute_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor,
                          sigma: float = 3.0) -> torch.Tensor:
    """
    Low-frequency Preservation Loss: 저주파 성분 보존.

    저주파 = 구조, 균일 영역 (위액, 장기 밀도 등)
    고주파 = 노이즈, 엣지

    저주파 성분이 보존되면 저음영 영역(위액 등)이 보존됨.

    Args:
        pred: 모델 출력 (B, 1, H, W)
        target: 타겟 이미지 (B, 1, H, W)
        sigma: Gaussian blur sigma (클수록 더 저주파만 추출)

    Returns:
        lowfreq_loss: 스칼라 텐서
    """
    # Gaussian kernel 생성
    kernel_size = int(sigma * 4) | 1  # 홀수로 만듦
    if kernel_size < 3:
        kernel_size = 3

    # 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=pred.device)
    x = x - kernel_size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 2D Gaussian kernel (separable)
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    pad = kernel_size // 2

    # 저주파 추출 (Gaussian blur)
    pred_lowfreq = F.conv2d(pred, gauss_2d, padding=pad)
    target_lowfreq = F.conv2d(target, gauss_2d, padding=pad)

    # L1 loss on low-frequency components
    lowfreq_loss = F.l1_loss(pred_lowfreq, target_lowfreq)

    return lowfreq_loss


# ============================================================
# 역정규화 (Denormalization)
# ============================================================

def denormalize_to_hu(tensor: torch.Tensor,
                      hu_min: float = HU_MIN,
                      hu_max: float = HU_MAX) -> torch.Tensor:
    """
    학습을 위해 [-1, 1]로 압축했던 데이터를
    다시 DICOM 수준의 HU 정밀도로 복원한다.
    float32 정밀도를 유지한 역변환.
    """
    rescaled = (tensor.float() + 1.0) / 2.0           # [0, 1]
    hu = rescaled * (hu_max - hu_min) + hu_min         # HU scale
    return hu


def denormalize_to_dicom_array(tensor: torch.Tensor,
                               hu_min: float = HU_MIN,
                               hu_max: float = HU_MAX,
                               bit_depth: int = 16) -> np.ndarray:
    """
    텐서를 DICOM 저장 가능한 정수형 배열로 변환한다.
    float32 → HU → 정수형 매핑.

    Args:
        tensor: (1, 1, H, W) or (1, H, W) or (H, W) 텐서
        bit_depth: 출력 비트 심도 (12 또는 16)
    """
    hu = denormalize_to_hu(tensor, hu_min, hu_max)
    hu_np = hu.detach().cpu().numpy().squeeze()

    if bit_depth == 16:
        # 16-bit 범위로 매핑 (DICOM 표준)
        pixel_array = hu_np.astype(np.float32)
        # RescaleSlope=1, RescaleIntercept=0 기준
        return pixel_array
    elif bit_depth == 12:
        # 12-bit 범위 (0 ~ 4095)
        normalized = (hu_np - hu_min) / (hu_max - hu_min)
        pixel_array = (normalized * 4095).clip(0, 4095).astype(np.uint16)
        return pixel_array
    else:
        return hu_np


# ============================================================
# CT 윈도잉 (Window Level / Width)
# ============================================================

def apply_ct_window(normalized: np.ndarray,
                    window_level: float = 40.0,
                    window_width: float = 400.0,
                    hu_min: float = HU_MIN,
                    hu_max: float = HU_MAX) -> np.ndarray:
    """
    정규화된 [-1,1] 영상에 CT 윈도잉을 적용하여 8-bit 디스플레이 영상으로 변환.

    Args:
        normalized: [-1, 1] 범위의 float32 영상
        window_level: 윈도우 중심 HU (복부=40)
        window_width: 윈도우 폭 HU (복부=400)
    Returns:
        uint8 ndarray [0, 255]
    """
    # [-1, 1] → HU
    hu = (normalized + 1.0) / 2.0 * (hu_max - hu_min) + hu_min

    # 윈도잉 클리핑
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    windowed = np.clip(hu, lower, upper)

    # [0, 255] uint8
    display = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return display


# ============================================================
# 시노그램 ↔ CT 이미지 변환
# ============================================================

def sinogram_to_ct(sinogram: np.ndarray, n_angles: int = 360) -> np.ndarray:
    """
    시노그램을 FBP (Filtered Back Projection)로 CT 이미지로 역변환.

    Args:
        sinogram: (H, W) 형태의 시노그램
        n_angles: 시노그램 생성 시 사용한 각도 수

    Returns:
        ct_image: (size, size) 형태의 CT 이미지 (HU)
    """
    # 저장된 시노그램 형태: (angles, detectors)
    # iradon이 기대하는 형태: (detectors, angles)
    # 따라서 transpose 필요
    sinogram = sinogram.T  # (detectors, angles)

    # theta 배열을 실제 angles 수에 맞춤
    actual_angles = sinogram.shape[1]
    theta = np.linspace(0., 180., actual_angles, endpoint=False)

    # FBP 역변환
    ct_attenuation = iradon(sinogram, theta=theta, circle=True)

    # 감쇠계수 → HU 변환
    mu_water = 0.02
    ct_hu = (ct_attenuation / mu_water - 1.0) * 1000.0

    return ct_hu.astype(np.float32)


def sinogram_to_ct_tensor(sinogram_tensor: torch.Tensor, n_angles: int = 360,
                          normalized_input: bool = True) -> torch.Tensor:
    """
    배치 시노그램 텐서를 CT 이미지 텐서로 변환 (시각화용).

    Args:
        sinogram_tensor: (B, 1, H, W) 시노그램 텐서
        n_angles: 각도 수
        normalized_input: True면 입력이 [-1, 1] 정규화된 상태

    Returns:
        ct_tensor: (B, 1, H', W') CT 이미지 텐서 (정규화된 [-1, 1] 범위)

    Note:
        시각화 목적이므로 정확한 HU 값 대신 상대적 구조 보존에 초점.
        [-1,1] 시노그램 → [0,1] → FBP → min-max 정규화 → [-1,1]
    """
    batch_size = sinogram_tensor.shape[0]
    ct_list = []

    for i in range(batch_size):
        sino = sinogram_tensor[i, 0].detach().cpu().numpy()

        # [-1, 1] → [0, 1] (원본 시노그램은 비음수)
        if normalized_input:
            sino = (sino + 1.0) / 2.0

        # Transpose: (angles, detectors) → (detectors, angles) for iradon
        sino_t = sino.T
        actual_angles = sino_t.shape[1]
        theta = np.linspace(0., 180., actual_angles, endpoint=False)

        # FBP 역변환
        ct_recon = iradon(sino_t, theta=theta, circle=True)

        # 시각화용 min-max 정규화 → [-1, 1]
        ct_min, ct_max = ct_recon.min(), ct_recon.max()
        if ct_max > ct_min:
            ct_normalized = (ct_recon - ct_min) / (ct_max - ct_min) * 2.0 - 1.0
        else:
            ct_normalized = np.zeros_like(ct_recon)

        ct_list.append(ct_normalized.astype(np.float32))

    ct_array = np.stack(ct_list, axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(ct_array).float()


# ============================================================
# 샘플 시각화 저장
# ============================================================

def save_ct_slice(tensor: torch.Tensor,
                  save_path: str,
                  window_level: float = 40.0,
                  window_width: float = 400.0) -> None:
    """
    단일 CT 슬라이스를 그대로 저장 (회전/반전 없음).
    CT 윈도잉 적용하여 8-bit grayscale PNG로 저장.

    Args:
        tensor: (1, 1, H, W) or (B, 1, H, W) normalized [-1, 1] tensor
        save_path: 저장 경로
        window_level: CT 윈도우 레벨 (HU)
        window_width: CT 윈도우 폭 (HU)
    """
    try:
        from skimage.io import imsave
    except ImportError:
        return

    img = tensor.detach().cpu().numpy().squeeze()  # (H, W)
    display = apply_ct_window(img, window_level, window_width)  # uint8
    imsave(save_path, display)


# ============================================================
# 추론 엔진 (Inference Engine)
# ============================================================

class InferenceEngine:
    """
    테스트 데이터셋을 사용하여 추론 샘플을 생성하는 엔진.

    파이프라인:
    1. Swin U-Net 디노이저로 초기 복원
    2. (선택) VAE 인코딩 → 확산 정제 → VAE 디코딩
    3. 역정규화하여 HU 스케일 복원
    """

    def __init__(self, model, device: torch.device,
                 use_diffusion: bool = True,
                 refine_steps: int = 20):
        self.model = model
        self.device = device
        self.use_diffusion = use_diffusion
        self.refine_steps = refine_steps

    @torch.no_grad()
    def run_single(self, input_tensor: torch.Tensor
                   ) -> Dict[str, torch.Tensor]:
        """단일 입력에 대한 추론 실행."""
        self.model.eval()
        x = input_tensor.to(self.device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        output = self.model.inference(
            x, use_diffusion=self.use_diffusion,
            refine_steps=self.refine_steps
        )

        return {
            'input': x.cpu(),
            'output': output.cpu(),
            'output_hu': denormalize_to_hu(output).cpu()
        }

    @torch.no_grad()
    def run_batch(self, dataloader,
                  max_samples: Optional[int] = None
                  ) -> list:
        """데이터로더 전체 또는 제한된 수의 샘플에 대해 추론 실행."""
        self.model.eval()
        results = []
        count = 0

        for batch in dataloader:
            x = batch['input'].to(self.device)
            output = self.model.inference(
                x, use_diffusion=self.use_diffusion,
                refine_steps=self.refine_steps
            )

            for i in range(x.shape[0]):
                results.append({
                    'input': x[i:i+1].cpu(),
                    'output': output[i:i+1].cpu(),
                    'output_hu': denormalize_to_hu(output[i:i+1]).cpu()
                })
                count += 1
                if max_samples is not None and count >= max_samples:
                    return results

        return results


# ============================================================
# 학습 유틸리티
# ============================================================

class EarlyStopping:
    """검증 손실 기반 조기 종료."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss: float) -> bool:
        """True를 반환하면 학습을 중단해야 함."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class LRSchedulerWrapper:
    """Warmup + Cosine Annealing 학습률 스케줄러 래퍼."""

    def __init__(self, optimizer, total_epochs: int,
                 warmup_epochs: int = 5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
