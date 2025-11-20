# inference_n2n.py - Inference script for trained N2N model

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# Add SwinIR to path
sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR


class N2NInference:
    """
    Inference class for N2N denoising
    
    Features:
    - Patch-based processing (memory efficient)
    - Overlap stitching (seamless results)
    - Full volume processing
    - HU windowing
    """
    
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.hu_window = self.config['preprocessing']['hu_window']
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Device: {self.device}")
        print(f"HU window: {self.hu_window}")
        print("Ready for inference!\n")
    
    def _load_model(self, model_path):
        """Load trained model"""
        # Build model
        model = SwinIR(
            upscale=self.config['model']['upscale'],
            in_chans=self.config['model']['in_chans'],
            img_size=self.config['model']['img_size'],
            window_size=self.config['model']['window_size'],
            img_range=self.config['model']['img_range'],
            depths=self.config['model']['depths'],
            embed_dim=self.config['model']['embed_dim'],
            num_heads=self.config['model']['num_heads'],
            mlp_ratio=self.config['model']['mlp_ratio'],
            upsampler=self.config['model']['upsampler'],
            resi_connection=self.config['model']['resi_connection']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def normalize_hu(self, img):
        """Normalize HU values to [0, 1]"""
        img = np.clip(img, self.hu_window[0], self.hu_window[1])
        img = (img - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0])
        return img.astype(np.float32)
    
    def denormalize_hu(self, img):
        """Convert [0, 1] back to HU values"""
        img = img * (self.hu_window[1] - self.hu_window[0]) + self.hu_window[0]
        return img.astype(np.float32)
    
    def denoise_slice(self, slice_2d, patch_size=128, overlap=16):
        """
        Denoise a single 2D slice using overlapping patches
        
        Args:
            slice_2d: 2D numpy array [H, W]
            patch_size: patch size for processing
            overlap: overlap size between patches
        
        Returns:
            denoised: denoised 2D array [H, W]
        """
        H, W = slice_2d.shape
        stride = patch_size - overlap
        
        # Pad image if necessary
        pad_h = (stride - H % stride) % stride
        pad_w = (stride - W % stride) % stride
        
        if pad_h > 0 or pad_w > 0:
            slice_2d = np.pad(slice_2d, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        H_pad, W_pad = slice_2d.shape
        
        # Initialize output and weight map
        output = np.zeros((H_pad, W_pad), dtype=np.float32)
        weight_map = np.zeros((H_pad, W_pad), dtype=np.float32)
        
        # Process patches
        with torch.no_grad():
            for i in range(0, H_pad - patch_size + 1, stride):
                for j in range(0, W_pad - patch_size + 1, stride):
                    # Extract patch
                    patch = slice_2d[i:i+patch_size, j:j+patch_size]
                    
                    # To tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Denoise
                    denoised_patch = self.model(patch_tensor)
                    denoised_patch = torch.clamp(denoised_patch, 0, 1)
                    
                    # To numpy
                    denoised_patch = denoised_patch.squeeze().cpu().numpy()
                    
                    # Add to output with weight
                    output[i:i+patch_size, j:j+patch_size] += denoised_patch
                    weight_map[i:i+patch_size, j:j+patch_size] += 1
        
        # Average overlapping regions
        output = output / np.maximum(weight_map, 1)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:H, :W]
        
        return output
    
    def denoise_volume(self, volume_path, output_path, visualize=True):
        """
        Denoise a full 3D volume
        
        Args:
            volume_path: path to input NIfTI file
            output_path: path to save denoised NIfTI
            visualize: whether to save visualization
        """
        print(f"\nProcessing: {volume_path}")
        
        # Load volume
        nii = nib.load(str(volume_path))
        volume = nii.get_fdata()
        H, W, D = volume.shape
        
        print(f"Volume shape: {volume.shape}")
        
        # Normalize
        volume_norm = self.normalize_hu(volume)
        
        # Denoise slice by slice
        denoised_volume = np.zeros_like(volume_norm)
        
        for slice_idx in tqdm(range(D), desc="Denoising slices"):
            slice_2d = volume_norm[:, :, slice_idx]
            denoised_slice = self.denoise_slice(slice_2d)
            denoised_volume[:, :, slice_idx] = denoised_slice
        
        # Denormalize
        denoised_volume_hu = self.denormalize_hu(denoised_volume)
        
        # Save as NIfTI
        output_nii = nib.Nifti1Image(denoised_volume_hu, nii.affine, nii.header)
        nib.save(output_nii, str(output_path))
        
        print(f"Saved: {output_path}")
        
        # Visualize
        if visualize:
            vis_path = output_path.parent / f"{output_path.stem}_comparison.png"
            self.visualize_results(volume, denoised_volume_hu, vis_path)
    
    def visualize_results(self, noisy_volume, denoised_volume, save_path, num_slices=5):
        """
        Create before/after comparison visualization
        
        Args:
            noisy_volume: original volume (HU values)
            denoised_volume: denoised volume (HU values)
            save_path: path to save visualization
            num_slices: number of slices to show
        """
        D = noisy_volume.shape[2]
        slice_indices = np.linspace(D // 4, 3 * D // 4, num_slices, dtype=int)
        
        fig, axes = plt.subplots(2, num_slices, figsize=(4*num_slices, 8))
        
        for idx, slice_idx in enumerate(slice_indices):
            # Noisy
            axes[0, idx].imshow(noisy_volume[:, :, slice_idx], cmap='gray', 
                               vmin=self.hu_window[0], vmax=self.hu_window[1])
            axes[0, idx].set_title(f'Noisy (Slice {slice_idx})')
            axes[0, idx].axis('off')
            
            # Denoised
            axes[1, idx].imshow(denoised_volume[:, :, slice_idx], cmap='gray',
                               vmin=self.hu_window[0], vmax=self.hu_window[1])
            axes[1, idx].set_title(f'Denoised (Slice {slice_idx})')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='N2N Inference - Denoise CT images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input NIfTI file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = N2NInference(args.model, args.config, args.device)
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    if input_path.is_file():
        # Single file
        output_path = output_dir / f"{input_path.stem}_denoised.nii.gz"
        inferencer.denoise_volume(input_path, output_path, visualize=not args.no_viz)
    
    elif input_path.is_dir():
        # Batch processing
        nii_files = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
        
        if len(nii_files) == 0:
            print(f"No NIfTI files found in {input_path}")
            return
        
        print(f"\nFound {len(nii_files)} files")
        
        for nii_file in nii_files:
            output_path = output_dir / f"{nii_file.stem}_denoised.nii.gz"
            inferencer.denoise_volume(nii_file, output_path, visualize=not args.no_viz)
    
    else:
        print(f"Invalid input path: {input_path}")
        return
    
    print("\n" + "="*80)
    print("Inference completed!")
    print("="*80)


if __name__ == '__main__':
    main()