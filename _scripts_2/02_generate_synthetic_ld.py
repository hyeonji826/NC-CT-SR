#!/usr/bin/env python3
"""
02_generate_synthetic_ld.py
NC (표준선량) → Synthetic LD (저선량) 변환
물리적 노이즈 모델: Poisson + Gaussian
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


class CTNoiseSimulator:
    """
    CT 저선량 노이즈 시뮬레이터
    
    물리적 근거:
    1. Poisson noise: X-ray photon counting 과정에서 발생
    2. Gaussian noise: 전자 노이즈 (detector, readout)
    
    피폭량 감소 → photon count 감소 → 노이즈 증가
    """
    
    def __init__(self, dose_reduction_factor=12, seed=42):
        """
        Args:
            dose_reduction_factor: 피폭량 감소 비율 (12 = 1/12배)
            seed: 랜덤 시드
        """
        self.dose_factor = dose_reduction_factor
        self.seed = seed
        np.random.seed(seed)
        
        # 노이즈 파라미터 (경험적 값)
        self.poisson_scale = 0.1  # Poisson 강도
        self.gaussian_std = 0.02  # Gaussian 표준편차
        
    def add_poisson_noise(self, image):
        """
        Poisson 노이즈 추가
        
        저선량: photon count ↓ → signal-dependent noise ↑
        """
        # 음수 방지
        image_shifted = image + 1.0  # [0, 1] → [1, 2]
        
        # Dose reduction 시뮬레이션
        # 낮은 선량 = 낮은 photon count = 높은 noise
        photon_count = image_shifted / self.dose_factor
        
        # Poisson sampling
        noisy = np.random.poisson(photon_count * 1000) / 1000.0
        
        # 원래 범위로 복원
        noisy = noisy * self.dose_factor - 1.0
        
        return noisy
    
    def add_gaussian_noise(self, image):
        """
        Gaussian 노이즈 추가
        
        전자 노이즈는 signal-independent
        """
        noise = np.random.normal(0, self.gaussian_std, image.shape)
        return image + noise
    
    def simulate_low_dose(self, image):
        """
        전체 저선량 시뮬레이션
        
        Args:
            image: 표준선량 CT (normalized [0, 1])
        
        Returns:
            noisy_image: 저선량 CT
        """
        # Step 1: Poisson noise (photon counting)
        noisy = self.add_poisson_noise(image)
        
        # Step 2: Gaussian noise (electronic)
        noisy = self.add_gaussian_noise(noisy)
        
        # Clip to valid range
        noisy = np.clip(noisy, 0, 1)
        
        return noisy


def process_patient(nc_path, output_path, simulator, sample_dir=None, visualize=False):
    """
    환자 1명의 NC → Synthetic LD 변환
    
    Args:
        nc_path: NC_norm.nii.gz 경로
        output_path: 출력 경로
        simulator: CTNoiseSimulator 인스턴스
        sample_dir: 샘플 저장 경로
        visualize: 시각화 여부
    """
    # Load NC
    img = sitk.ReadImage(str(nc_path))
    arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
    
    # Simulate LD
    noisy_arr = np.zeros_like(arr)
    for z in range(arr.shape[0]):
        noisy_arr[z] = simulator.simulate_low_dose(arr[z])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_img = sitk.GetImageFromArray(noisy_arr)
    noisy_img.CopyInformation(img)
    sitk.WriteImage(noisy_img, str(output_path))
    
    # Visualize (optional)
    if visualize and sample_dir is not None:
        sample_dir = Path(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        patient_id = output_path.parent.name
        num_slices = arr.shape[0]
        
        # 여러 슬라이스 비교 (4개)
        slice_indices = [
            num_slices // 4,
            num_slices // 2 - 5,
            num_slices // 2 + 5,
            3 * num_slices // 4
        ]
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(f'Patient {patient_id} - Synthetic LD Generation\n'
                    f'Dose Reduction: 1/{simulator.dose_factor}x', 
                    fontsize=16, fontweight='bold')
        
        for row, slice_idx in enumerate(slice_indices):
            # Original
            axes[row, 0].imshow(arr[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_title(f'Slice {slice_idx}: Original NC (Standard Dose)')
            axes[row, 0].axis('off')
            
            # Synthetic LD
            axes[row, 1].imshow(noisy_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title(f'Synthetic LD (1/{simulator.dose_factor}x dose)')
            axes[row, 1].axis('off')
            
            # Noise Map
            diff = np.abs(arr[slice_idx] - noisy_arr[slice_idx])
            im = axes[row, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
            axes[row, 2].set_title(f'Noise Map (mean: {diff.mean():.4f})')
            axes[row, 2].axis('off')
        
        plt.colorbar(im, ax=axes[:, 2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = sample_dir / f'{patient_id}_comparison.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return arr, noisy_arr
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic Low-Dose CT')
    
    parser.add_argument('--input-dir', type=str, 
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC',
                       help='NC 데이터 경로')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\synthetic_ld\NC',
                       help='Synthetic LD 출력 경로')
    parser.add_argument('--sample-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\synthetic_ld',
                       help='샘플 비교 이미지 저장 경로')
    parser.add_argument('--dose-reduction', type=int, default=12,
                       help='피폭량 감소 비율 (12 = 1/12배)')
    parser.add_argument('--visualize-samples', type=int, default=10,
                       help='시각화할 샘플 수')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir)
    
    print("="*80)
    print("Synthetic Low-Dose CT 생성")
    print("="*80)
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print(f"피폭량 감소: 1/{args.dose_reduction}x")
    print("="*80)
    
    # 노이즈 시뮬레이터 초기화
    simulator = CTNoiseSimulator(
        dose_reduction_factor=args.dose_reduction,
        seed=args.seed
    )
    
    # 환자 목록
    patient_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print(f"\n총 환자 수: {len(patient_dirs)}")
    
    # 처리
    visualize_count = 0
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        patient_id = patient_dir.name
        nc_path = patient_dir / 'NC_norm.nii.gz'
        
        if not nc_path.exists():
            print(f"Missing: {nc_path}")
            continue
        
        # 출력 경로
        output_path = output_dir / patient_id / 'NC_synthetic_ld.nii.gz'
        
        # 시각화 여부
        visualize = visualize_count < args.visualize_samples
        
        # 처리
        clean, noisy = process_patient(nc_path, output_path, simulator, sample_dir, visualize)
        
        if visualize:
            visualize_count += 1
            
            # 노이즈 통계
            if clean is not None:
                noise = noisy - clean
                print(f"\n환자 {patient_id} 노이즈 통계:")
                print(f"  Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")
                print(f"  Noisy mean: {noisy.mean():.4f}, std: {noisy.std():.4f}")
                print(f"  Noise mean: {noise.mean():.4f}, std: {noise.std():.4f}")
    
    print("\n" + "="*80)
    print("완료!")
    print(f"출력 경로: {output_dir}")
    print("="*80)
    
    # 요약 통계
    print("\n[통계 요약]")
    print(f"처리된 환자 수: {len(list(output_dir.glob('*/NC_synthetic_ld.nii.gz')))}")


if __name__ == '__main__':
    main()