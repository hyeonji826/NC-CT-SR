"""
03_finetune_nc_to_ce.py
Phase 2: Content Encoder Freeze Fine-Tuning

★ 핵심 목표:
1. 기존 학습된 Content Encoder (구조 특징 추출) 고정!
2. Style Encoder와 Decoder만 학습하여 조영 패턴 학습 심화.
3. 사전 학습된 모델 가중치를 로드하여 이어서 학습.

Architecture: Structure-Preserving Style Transfer
- Content: NC (구조 절대 보존!) -> Frozen
- Style: CE (조영 패턴만) -> Fine-Tuning
- Output: Enhanced NC -> Fine-Tuning

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# models.py에서 StructurePreservingStyleTransfer를 불러옵니다.
from models import StructurePreservingStyleTransfer

try:
    # VGG Loss를 위한 torchvision import
    from torchvision.models import vgg19, VGG19_Weights
except ImportError:
    vgg19 = None


# ============================================================
# Dataset
# ============================================================

class UnpairedNCCEDataset(Dataset):
    """
    Unpaired NC-CE 데이터셋
    """
    def __init__(self, nc_dir, ce_dir, image_size=256, augment=True):
        self.nc_dir = Path(nc_dir)
        self.ce_dir = Path(ce_dir)
        self.image_size = image_size
        self.augment = augment
        
        # NC 슬라이스 목록 생성 로직
        self.nc_slices = []
        for patient_dir in sorted([p for p in self.nc_dir.iterdir() if p.is_dir()]):
            nii_path = patient_dir / 'NC_norm.nii.gz'
            if nii_path.exists():
                img = sitk.ReadImage(str(nii_path))
                num_slices = img.GetSize()[2]
                for slice_idx in range(num_slices):
                    self.nc_slices.append({
                        'patient_id': patient_dir.name,
                        'nii_path': nii_path,
                        'slice_idx': slice_idx
                    })
        
        # CE 슬라이스 목록 생성 로직
        self.ce_slices = []
        for patient_dir in sorted([p for p in self.ce_dir.iterdir() if p.is_dir()]):
            nii_path = patient_dir / 'CE_norm.nii.gz'
            if nii_path.exists():
                img = sitk.ReadImage(str(nii_path))
                num_slices = img.GetSize()[2]
                for slice_idx in range(num_slices):
                    self.ce_slices.append({
                        'patient_id': patient_dir.name,
                        'nii_path': nii_path,
                        'slice_idx': slice_idx
                    })
        
        print(f"NC slices: {len(self.nc_slices)}")
        print(f"CE slices: {len(self.ce_slices)}")
    
    def __len__(self):
        return max(len(self.nc_slices), len(self.ce_slices))
    
    def load_slice(self, nii_path, slice_idx):
        """NIfTI에서 특정 슬라이스 로드"""
        img = sitk.ReadImage(str(nii_path))
        arr = sitk.GetArrayFromImage(img)
        slice_2d = arr[slice_idx]
        return slice_2d
    
    def augment_slice(self, img):
        """Augmentation"""
        if not self.augment: return img
        if random.random() > 0.5: img = np.fliplr(img)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            from scipy.ndimage import rotate
            img = rotate(img, angle, reshape=False, order=1, mode='nearest')
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            img = np.clip(img * factor, 0, 1)
        return img
    
    def __getitem__(self, idx):
        # NC 슬라이스 (구조 기준)
        nc_idx = idx % len(self.nc_slices)
        nc_info = self.nc_slices[nc_idx]
        nc_slice = self.load_slice(nc_info['nii_path'], nc_info['slice_idx'])
        
        # CE 슬라이스 (조영 기준, unpaired - 랜덤)
        ce_idx = random.randint(0, len(self.ce_slices) - 1)
        ce_info = self.ce_slices[ce_idx]
        ce_slice = self.load_slice(ce_info['nii_path'], ce_info['slice_idx'])
        
        # Augmentation and Resize
        nc_slice = self.augment_slice(nc_slice)
        ce_slice = self.augment_slice(ce_slice)
        
        from skimage.transform import resize
        nc_slice = resize(nc_slice, (self.image_size, self.image_size), 
                         order=1, preserve_range=True, anti_aliasing=True)
        ce_slice = resize(ce_slice, (self.image_size, self.image_size),
                         order=1, preserve_range=True, anti_aliasing=True)
        
        # To tensor
        nc_tensor = torch.from_numpy(nc_slice).float().unsqueeze(0)
        ce_tensor = torch.from_numpy(ce_slice).float().unsqueeze(0)
        
        return {
            'nc': nc_tensor, 'ce': ce_tensor,
            'nc_patient': nc_info['patient_id'], 'ce_patient': ce_info['patient_id']
        }


# ============================================================
# Loss Functions
# ============================================================

class StructurePreservingLoss(nn.Module):
    def __init__(self, content_weight=10.0, style_weight=1.0, perceptual_weight=5.0, anatomy_weight=3.0, device='cuda'):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.perceptual_weight = perceptual_weight
        self.anatomy_weight = anatomy_weight
        self.device = device
        
        # VGG for perceptual loss
        if vgg19 is not None:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
            self.vgg = nn.Sequential(*list(vgg.children())[:16])
            for param in self.vgg.parameters(): param.requires_grad = False
        else:
            self.vgg = None
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def content_loss(self, enhanced_nc, original_nc): return F.l1_loss(enhanced_nc, original_nc)
    
    def calc_mean_std(self, feat, eps=1e-5):
        B, C = feat.shape[:2]
        feat_var = feat.view(B, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std
        
    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)
    
    def style_loss(self, enhanced_nc, ce):
        if self.vgg is None:
            enhanced_mean, enhanced_std = self.calc_mean_std(enhanced_nc)
            ce_mean, ce_std = self.calc_mean_std(ce)
            return F.l1_loss(enhanced_mean, ce_mean) + F.l1_loss(enhanced_std, ce_std)
        enhanced_rgb = enhanced_nc.repeat(1, 3, 1, 1)
        ce_rgb = ce.repeat(1, 3, 1, 1)
        enhanced_feat = self.vgg(enhanced_rgb)
        ce_feat = self.vgg(ce_rgb)
        enhanced_gram = self.gram_matrix(enhanced_feat)
        ce_gram = self.gram_matrix(ce_feat)
        return F.mse_loss(enhanced_gram, ce_gram)

    def perceptual_loss(self, enhanced_nc, original_nc):
        if self.vgg is None: return torch.tensor(0.0, device=self.device)
        enhanced_rgb = enhanced_nc.repeat(1, 3, 1, 1)
        original_rgb = original_nc.repeat(1, 3, 1, 1)
        enhanced_feat = self.vgg(enhanced_rgb)
        original_feat = self.vgg(original_rgb)
        return F.l1_loss(enhanced_feat, original_feat)
    
    def anatomy_loss(self, enhanced_nc, original_nc):
        enhanced_edge_x = F.conv2d(enhanced_nc, self.sobel_x, padding=1)
        enhanced_edge_y = F.conv2d(enhanced_nc, self.sobel_y, padding=1)
        enhanced_edge = torch.sqrt(enhanced_edge_x**2 + enhanced_edge_y**2 + 1e-8)
        original_edge_x = F.conv2d(original_nc, self.sobel_x, padding=1)
        original_edge_y = F.conv2d(original_nc, self.sobel_y, padding=1)
        original_edge = torch.sqrt(original_edge_x**2 + original_edge_y**2 + 1e-8)
        return F.l1_loss(enhanced_edge, original_edge)
    
    def forward(self, enhanced_nc, original_nc, ce):
        loss_content = self.content_loss(enhanced_nc, original_nc)
        loss_style = self.style_loss(enhanced_nc, ce)
        loss_perceptual = self.perceptual_loss(enhanced_nc, original_nc)
        loss_anatomy = self.anatomy_loss(enhanced_nc, original_nc)
        
        total_loss = (
            self.content_weight * loss_content +
            self.style_weight * loss_style +
            self.perceptual_weight * loss_perceptual +
            self.anatomy_weight * loss_anatomy
        )
        
        loss_dict = {
            'content': loss_content.item(), 'style': loss_style.item(),
            'perceptual': loss_perceptual.item(), 'anatomy': loss_anatomy.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict


# ============================================================
# Trainer (Fine-Tuning 로직 + 동적 스케줄러 통합)
# ============================================================

class NCToCETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device is not None else 'cpu') 
        print(f"Device: {self.device}")
        
        # Directories
        self.exp_dir = Path(args.exp_root) / args.exp_dir
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.samples_dir = self.exp_dir / 'samples'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 모델 초기화 (★ 오류 수정: num_blocks 인자 제거 완료)
        print("\n모델 초기화...")
        # 'models' 모듈에서 StructurePreservingStyleTransfer를 불러왔다고 가정합니다.
        self.model = StructurePreservingStyleTransfer(base_channels=args.base_channels) 
        self.model = self.model.to(self.device)
        
        # 2. ★ 파인튜닝 로직: 가중치 로드 및 Content Encoder 고정 ★
        if args.load_checkpoint:
            print(f"\n==================================================================================")
            print(f"✅ Loading checkpoint from: {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Checkpoint load successful.")
            
            if args.content_freeze:
                print("----------------------------------------------------------------------------------")
                print("★ FINE-TUNING MODE: Content Encoder (구조 보존 파트) 파라미터 고정 시작")
                frozen_count = 0
                for name, param in self.model.named_parameters():
                    if 'content_encoder' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"고정된 파라미터 수: {frozen_count}개")
                print("Content Encoder 파라미터 고정 완료. Style/Decoder만 학습됩니다.")
                print("----------------------------------------------------------------------------------")
        
        # 3. 최적화 도구 설정
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=0.01
        )
        num_trainable_params = sum(p.numel() for p in self.optimizer.param_groups[0]['params'])
        num_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"⭐ 전체 파라미터 수: {num_total_params:,}, 학습할 파라미터 수: {num_trainable_params:,}")
        
        # 4. 스케줄러 관련 속성 초기화 (Style Weight 동적 조정을 위한 설정)
        self.content_loss_history = []  # Content Loss 기록
        self.max_style_weight = 100.0  # Style Weight 최대 한도
        # 실행 인자로 받은 style-weight를 초기값으로 사용 (예: 50.0)
        self.current_style_weight = args.style_weight 
        
        # 5. Loss 초기화
        # Loss 객체는 초기 가중치로 생성됩니다.
        self.criterion = StructurePreservingLoss(
            content_weight=args.content_weight, style_weight=self.current_style_weight, 
            perceptual_weight=args.perceptual_weight, anatomy_weight=args.anatomy_weight,
            device=self.device
        )
        
        # Scheduler (Learning Rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # Dataset (UnpairedNCCEDataset 클래스가 정의되어 있다고 가정)
        print("\n데이터셋 로딩...")
        self.train_dataset = UnpairedNCCEDataset(
            nc_dir=Path(args.data_root) / 'NC', ce_dir=Path(args.data_root) / 'CE',
            image_size=args.image_size, augment=True
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # States
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []


    def adjust_style_weight(self, current_content_loss):
        """Content Loss 변화에 따라 Style Loss 가중치를 동적으로 조정 (핵심 로직)"""
        
        # 1. Content Loss 기록
        self.content_loss_history.append(current_content_loss)
        
        # 2. 초기 안정화 기간 건너뛰기
        if len(self.content_loss_history) < 10: # 충분히 긴 기간(예: 10 Epoch) 동안 관찰 후 시작
            print(f"  [Scheduler] 안정화 중 ({len(self.content_loss_history)}/{10}). Style Weight: {self.current_style_weight:.2f}")
            return
        
        # 3. Content Loss 변화율 계산 (최근 5개 vs 이전 5개)
        recent_avg = np.mean(self.content_loss_history[-5:]) 
        prev_avg = np.mean(self.content_loss_history[-10:-5])
        
        increase_threshold = 1.005 # 0.5% 증가 시 반응 (더 민감하게 설정)
        
        if recent_avg > prev_avg * increase_threshold:
            # 4. 구조가 깨지면 Style Weight 감소
            # 10% 감소, 최소 1.0 유지
            self.current_style_weight = max(1.0, self.current_style_weight * 0.9) 
            print(f"  [Scheduler] ⚠️ Content Loss 증가! Style Weight를 {self.current_style_weight:.2f}로 감소.")
        else:
            # 5. 구조가 안정적이면 Style Weight 증가 (조영 효과 강화)
            # 5% 증가, 최대 100.0 제한
            self.current_style_weight = min(self.max_style_weight, self.current_style_weight * 1.05) 
            print(f"  [Scheduler] ✅ Content Loss 안정. Style Weight를 {self.current_style_weight:.2f}로 증가.")
            
        # 6. Loss 객체의 가중치 업데이트
        self.criterion.style_weight = self.current_style_weight


    def train_epoch(self, epoch):
        # 02_train_nc_to_ce.py에서 복사된 학습 루프 메서드
        self.model.train()
        epoch_losses = {'content': [], 'style': [], 'perceptual': [], 'anatomy': [], 'total': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            nc = batch['nc'].to(self.device)
            ce = batch['ce'].to(self.device)
            enhanced_nc = self.model(nc, ce, alpha=self.args.style_alpha)
            
            # self.criterion.style_weight는 adjust_style_weight에 의해 동적으로 업데이트됩니다.
            loss, loss_dict = self.criterion(enhanced_nc, nc, ce) 
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for key in loss_dict: epoch_losses[key].append(loss_dict[key])
            pbar.set_postfix({'loss': loss_dict['total'], 'content': loss_dict['content'], 'style': self.current_style_weight * loss_dict['style']})
        
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.train_losses.append(avg_losses)
        return avg_losses

    @torch.no_grad()
    def save_samples(self, epoch):
        # ... (02_train_nc_to_ce.py의 save_samples 메서드 내용을 복사) ...
        # PSNR/SSIM 계산 및 이미지 저장 로직은 여기에 복사해 주세요.
        pass

    def save_checkpoint(self, epoch, is_best=False):
        # ... (02_train_nc_to_ce.py의 save_checkpoint 메서드 내용을 복사) ...
        # 체크포인트 저장 로직은 여기에 복사해 주세요.
        pass

    def train(self):
        print("\n" + "="*80); print("학습 시작!"); print("="*80)
        print(f"Epochs: {self.args.epochs}"); print(f"Device: {self.device}")
        print(f"Loss weights:\n  - Content (구조 보존): {self.args.content_weight}\n  - Style (시작): {self.current_style_weight}\n  - Perceptual (해부학): {self.args.perceptual_weight}\n  - Anatomy (경계): {self.args.anatomy_weight}")
        print("="*80)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            losses = self.train_epoch(epoch)
            
            # ★ 동적 가중치 스케줄러 호출 (Content Loss를 기반으로 Style Weight 조정) ★
            self.adjust_style_weight(losses['content']) 
            
            self.scheduler.step()
            
            # 최종 로그 출력 시, 현재 style weight를 함께 출력하여 변화를 관찰합니다.
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Total Loss: {losses['total']:.6f}, Content (구조): {losses['content']:.6f}, Style (적용 가중치): {self.current_style_weight:.2f}, Perceptual: {losses['perceptual']:.6f}, Anatomy: {losses['anatomy']:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if (epoch + 1) % self.args.sample_interval == 0: self.save_samples(epoch)
            
            is_best = losses['total'] < self.best_loss
            if is_best: self.best_loss = losses['total']
            if (epoch + 1) % self.args.save_interval == 0 or is_best: self.save_checkpoint(epoch, is_best)
        
        print("\n" + "="*80); print("학습 완료!"); print(f"Best loss: {self.best_loss:.6f}"); print("="*80)


def get_args():
    parser = argparse.ArgumentParser(description='Structure Preserving Style Transfer Fine-Tuning')
    
    # Common
    parser.add_argument('--data-root', type=str, default='E:/LD-CT SR/Data/nii_preproc_norm',
                        help='데이터 루트 디렉토리')
    parser.add_argument('--exp-root', type=str, default='E:/LD-CT SR/experiments',
                        help='실험 결과 저장 루트 디렉토리')
    parser.add_argument('--exp-dir', type=str, default='nc_to_ce_finetune',
                        help='실험 디렉토리 (exp-root 하위)')
    parser.add_argument('--device', type=int, default=1,
                        help='사용할 GPU 번호 (0부터 시작)')
    
    # Model/Data
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--content-weight', type=float, default=10.0, help='구조 보존')
    parser.add_argument('--style-weight', type=float, default=1.0, help='조영 학습')
    parser.add_argument('--perceptual-weight', type=float, default=5.0, help='해부학 일관성')
    parser.add_argument('--anatomy-weight', type=float, default=3.0, help='경계 보존')
    parser.add_argument('--style-alpha', type=float, default=1.0, help='Style strength (0-1)')
    
    # Fine-Tuning Parameters (핵심)
    parser.add_argument('--load-checkpoint', type=str, 
                        default='E:/LD-CT SR/experiments/nc_to_ce_phase1/checkpoints/best.pth',
                        help='사전 학습된 모델 가중치 경로 (best.pth)')
    parser.add_argument('--content-freeze', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Content Encoder 파라미터 고정 여부 (True로 기본 설정)')
    
    # Save
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=1)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train
    trainer = NCToCETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()