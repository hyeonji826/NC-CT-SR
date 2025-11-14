"""
02_generate_synthetic_ld.py
NC (표준선량) → Synthetic LD (저선량) 변환
개선된 물리적 CT 노이즈 모델
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter


class ImprovedCTNoiseSimulator:
    """
    개선된 CT 저선량 노이즈 시뮬레이터
    
    실제 CT 물리학 기반:
    1. Quantum noise (Poisson) - photon counting
    2. Electronic noise (Gaussian) - detector readout
    3. Correlated noise - low-frequency artifacts
    4. Streak artifacts - beam hardening simulation
    """
    
    def __init__(self, dose_reduction_factor=12, seed=42):
        self.dose_factor = dose_reduction_factor
        self.seed = seed
        np.random.seed(seed)
        
        # 노이즈 파라미터 (실제 CT 기반 조정)
        self.quantum_scale = 0.15  # Quantum noise 강도
        self.electronic_std = 0.015  # Electronic noise
        self.correlated_sigma = 1.5  # Correlated noise blur
        self.correlated_strength = 0.02  # Correlated noise 강도
        
    def add_quantum_noise(self, image):
        """
        Quantum (Poisson) noise
        
        실제 CT: photon count ∝ e^(-μx)
        저선량: 더 적은 photon → 더 큰 variance
        """
        # HU space로 변환 (CT 물리학 적용)
        # [0, 1] → [-1000, 1000] HU (approximate)
        hu = (image - 0.5) * 2000
        
        # Exponential attenuation 시뮬레이션
        # I = I0 * e^(-μx)
        mu = 0.02  # Typical attenuation coefficient
        I0 = 10000 / self.dose_factor  # Incident photon count (dose-dependent)
        
        # Linear attenuation coefficient from HU
        # μ ∝ HU + 1000
        mu_eff = (hu + 1000) / 1000 * mu
        
        # Transmitted photon count
        transmitted = I0 * np.exp(-mu_eff)
        transmitted = np.maximum(transmitted, 1)  # Avoid zero
        
        # Poisson noise
        noisy_transmitted = np.random.poisson(transmitted)
        
        # Back to HU
        noisy_mu = -np.log(noisy_transmitted / I0)
        noisy_hu = noisy_mu / mu * 1000 - 1000
        
        # Clip and normalize back to [0, 1]
        noisy_hu = np.clip(noisy_hu, -1000, 1000)
        noisy_image = noisy_hu / 2000 + 0.5
        
        return noisy_image
    
    def add_electronic_noise(self, image):
        """
        Electronic (Gaussian) noise
        Detector readout 과정의 노이즈
        """
        noise = np.random.normal(0, self.electronic_std, image.shape)
        return image + noise
    
    def add_correlated_noise(self, image):
        """
        Correlated noise
        실제 CT에서 인접 픽셀은 상관관계 있음
        """
        # Low-frequency noise 생성
        low_freq_noise = np.random.normal(0, self.correlated_strength, image.shape)
        
        # Gaussian blur로 상관관계 부여
        correlated_noise = gaussian_filter(low_freq_noise, sigma=self.correlated_sigma)
        
        return image + correlated_noise
    
    def add_streak_artifacts(self, image, num_streaks=3):
        """
        Streak artifacts (선형 아티팩트)
        뼈, metal 주변에서 발생하는 beam hardening 시뮬레이션
        """
        h, w = image.shape
        artifact_image = image.copy()
        
        # 고밀도 영역 찾기 (뼈 등)
        high_density_mask = image > 0.7
        
        if high_density_mask.sum() > 100:  # 충분한 고밀도 영역이 있으면
            # 고밀도 영역에서 랜덤하게 streak 시작점 선택
            high_density_coords = np.argwhere(high_density_mask)
            
            for _ in range(num_streaks):
                if len(high_density_coords) == 0:
                    break
                
                # 랜덤 시작점
                start_idx = np.random.randint(len(high_density_coords))
                start_y, start_x = high_density_coords[start_idx]
                
                # 랜덤 방향
                angle = np.random.uniform(0, 2 * np.pi)
                
                # Streak 생성 (가느다란 선)
                length = np.random.randint(50, 150)
                intensity = np.random.uniform(-0.02, 0.02)
                
                for i in range(length):
                    y = int(start_y + i * np.sin(angle))
                    x = int(start_x + i * np.cos(angle))
                    
                    if 0 <= y < h and 0 <= x < w:
                        # Gaussian falloff
                        falloff = np.exp(-i / (length * 0.3))
                        artifact_image[y, x] += intensity * falloff
        
        return np.clip(artifact_image, 0, 1)
    
    def simulate_low_dose(self, image):
        """
        전체 저선량 시뮬레이션 파이프라인
        """
        # Step 1: Quantum noise (가장 중요)
        noisy = self.add_quantum_noise(image)
        
        # Step 2: Electronic noise
        noisy = self.add_electronic_noise(noisy)
        
        # Step 3: Correlated noise
        noisy = self.add_correlated_noise(noisy)
        
        # Step 4: Streak artifacts (occasional)
        if np.random.random() > 0.7:  # 30% 확률
            noisy = self.add_streak_artifacts(noisy, num_streaks=np.random.randint(1, 4))
        
        # Final clip
        noisy = np.clip(noisy, 0, 1)
        
        return noisy


def process_patient(nc_path, output_path, simulator, sample_dir=None, visualize=False):
    """환자 1명 처리"""
    img = sitk.ReadImage(str(nc_path))
    arr = sitk.GetArrayFromImage(img)
    
    noisy_arr = np.zeros_like(arr)
    for z in range(arr.shape[0]):
        noisy_arr[z] = simulator.simulate_low_dose(arr[z])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_img = sitk.GetImageFromArray(noisy_arr)
    noisy_img.CopyInformation(img)
    sitk.WriteImage(noisy_img, str(output_path))
    
    if visualize and sample_dir is not None:
        sample_dir = Path(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        patient_id = output_path.parent.name
        num_slices = arr.shape[0]
        
        slice_indices = [
            num_slices // 4,
            num_slices // 2 - 5,
            num_slices // 2 + 5,
            3 * num_slices // 4
        ]
        
        fig, axes = plt.subplots(4, 4, figsize=(18, 20))
        fig.suptitle(f'Patient {patient_id} - Realistic CT Noise Simulation\n'
                    f'Dose Reduction: 1/{simulator.dose_factor}x', 
                    fontsize=16, fontweight='bold')
        
        for row, slice_idx in enumerate(slice_indices):
            # Original
            axes[row, 0].imshow(arr[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_title(f'Slice {slice_idx}: Original NC')
            axes[row, 0].axis('off')
            
            # Synthetic LD
            axes[row, 1].imshow(noisy_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title(f'Synthetic LD (Realistic)')
            axes[row, 1].axis('off')
            
            # Noise Map
            diff = np.abs(arr[slice_idx] - noisy_arr[slice_idx])
            im = axes[row, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
            axes[row, 2].set_title(f'Noise Map (mean: {diff.mean():.4f})')
            axes[row, 2].axis('off')
            
            # Noise histogram
            axes[row, 3].hist(diff.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[row, 3].set_title(f'Noise Distribution')
            axes[row, 3].set_xlabel('Noise Level')
            axes[row, 3].set_ylabel('Frequency')
            axes[row, 3].grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=axes[:, 2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = sample_dir / f'{patient_id}_realistic_noise.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return arr, noisy_arr
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Generate Realistic Synthetic Low-Dose CT')
    
    parser.add_argument('--input-dir', type=str, 
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\synthetic_ld\NC')
    parser.add_argument('--sample-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\synthetic_ld')
    parser.add_argument('--dose-reduction', type=int, default=12)
    parser.add_argument('--visualize-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir)
    
    print("="*80)
    print("Realistic Synthetic Low-Dose CT 생성")
    print("="*80)
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print(f"피폭량 감소: 1/{args.dose_reduction}x")
    print("\n물리적 노이즈 모델:")
    print("  - Quantum noise (Poisson)")
    print("  - Electronic noise (Gaussian)")
    print("  - Correlated noise (Low-frequency)")
    print("  - Streak artifacts (Beam hardening)")
    print("="*80)
    
    simulator = ImprovedCTNoiseSimulator(
        dose_reduction_factor=args.dose_reduction,
        seed=args.seed
    )
    
    patient_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print(f"\n총 환자 수: {len(patient_dirs)}")
    
    visualize_count = 0
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        patient_id = patient_dir.name
        nc_path = patient_dir / 'NC_norm.nii.gz'
        
        if not nc_path.exists():
            continue
        
        output_path = output_dir / patient_id / 'NC_synthetic_ld.nii.gz'
        visualize = visualize_count < args.visualize_samples
        
        clean, noisy = process_patient(nc_path, output_path, simulator, sample_dir, visualize)
        
        if visualize:
            visualize_count += 1
            
            if clean is not None:
                noise = noisy - clean
                print(f"\n환자 {patient_id} 노이즈 통계:")
                print(f"  Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")
                print(f"  Noisy mean: {noisy.mean():.4f}, std: {noisy.std():.4f}")
                print(f"  Noise mean: {noise.mean():.4f}, std: {noise.std():.4f}")
                print(f"  SNR (estimate): {clean.mean() / noise.std():.2f}")
    
    print("\n" + "="*80)
    print("완료!")
    print(f"출력 경로: {output_dir}")
    print(f"샘플 경로: {sample_dir}")
    print("="*80)


if __name__ == '__main__':
    main()