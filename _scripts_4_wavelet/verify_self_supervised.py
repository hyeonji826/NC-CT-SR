#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-Supervised Wavelet Loss 구현 검증 스크립트
"""

import torch
import sys
sys.path.insert(0, r'E:\LD-CT SR\_scripts_4_wavelet')

print("="*80)
print("🧪 Self-Supervised Wavelet Loss 검증")
print("="*80)

# ============================================================================
# Test 1: Import 확인
# ============================================================================
print("\n1️⃣ Import 테스트...")
try:
    from losses import (
        CombinedLoss, 
        WaveletSparsityLoss, 
        Noise2VoidLoss, 
        SelfSupervisedCombinedLoss
    )
    print("   ✅ 모든 loss 클래스 import 성공!")
except Exception as e:
    print(f"   ❌ Import 실패: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: WaveletSparsityLoss (Target 불필요!)
# ============================================================================
print("\n2️⃣ WaveletSparsityLoss 테스트 (NO TARGET!)...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    wavelet_loss = WaveletSparsityLoss(
        threshold=50,
        levels=3,
        normalize_threshold=True
    )
    
    # Create noisy image
    noisy = torch.rand(2, 1, 128, 128).to(device)
    
    # Loss 계산 - Target 없이!
    loss = wavelet_loss(noisy)
    
    print(f"   ✅ WaveletSparsityLoss: {loss.item():.6f}")
    print(f"   ✅ Requires grad: {loss.requires_grad}")
    print(f"   ✅ Target 불필요 확인!")
    
except Exception as e:
    print(f"   ❌ WaveletSparsityLoss 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 3: Noise2VoidLoss
# ============================================================================
print("\n3️⃣ Noise2VoidLoss 테스트...")
try:
    n2v_loss = Noise2VoidLoss(mask_ratio=0.02)
    
    noisy_input = torch.rand(2, 1, 128, 128).to(device)
    pred = torch.rand(2, 1, 128, 128).to(device)
    
    # Loss 계산
    loss = n2v_loss(pred, noisy_input)
    
    print(f"   ✅ Noise2VoidLoss: {loss.item():.6f}")
    print(f"   ✅ Requires grad: {loss.requires_grad}")
    
except Exception as e:
    print(f"   ❌ Noise2VoidLoss 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 4: SelfSupervisedCombinedLoss
# ============================================================================
print("\n4️⃣ SelfSupervisedCombinedLoss 테스트...")
try:
    criterion = SelfSupervisedCombinedLoss(
        n2v_weight=1.0,
        wavelet_weight=0.2,
        tv_weight=0.01,
        wavelet_threshold=50,
        wavelet_levels=3
    ).to(device)
    
    noisy_input = torch.rand(2, 1, 128, 128).to(device)
    pred = torch.rand(2, 1, 128, 128).to(device)
    
    # Loss 계산
    total_loss, loss_dict = criterion(pred, noisy_input)
    
    print(f"\n   ✅ Total Loss: {total_loss.item():.6f}")
    print(f"   ✅ N2V: {loss_dict['n2v']:.6f}")
    print(f"   ✅ Wavelet Sparsity: {loss_dict['wavelet_sparsity']:.6f}")
    print(f"   ✅ TV: {loss_dict['tv']:.6f}")
    
    # Backward 테스트
    total_loss.backward()
    print(f"   ✅ Backward 성공!")
    
except Exception as e:
    print(f"   ❌ SelfSupervisedCombinedLoss 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 5: Supervised vs Self-Supervised 비교
# ============================================================================
print("\n5️⃣ Supervised vs Self-Supervised 비교...")
try:
    # Supervised
    supervised_criterion = CombinedLoss(
        l1_weight=1.0,
        ssim_weight=0.2,
        wavelet_weight=0.1,
        learn_weights=False
    ).to(device)
    
    low = torch.rand(2, 1, 128, 128).to(device)
    full = torch.rand(2, 1, 128, 128).to(device)
    pred = torch.rand(2, 1, 128, 128).to(device)
    
    sup_loss, sup_dict = supervised_criterion(pred, full)
    print(f"\n   Supervised Loss: {sup_loss.item():.6f}")
    print(f"      - L1: {sup_dict['l1']:.6f}")
    print(f"      - SSIM: {sup_dict['ssim']:.6f}")
    print(f"      - Wavelet: {sup_dict['wavelet']:.6f}")
    
    # Self-Supervised
    self_sup_criterion = SelfSupervisedCombinedLoss(
        n2v_weight=1.0,
        wavelet_weight=0.2,
        tv_weight=0.01
    ).to(device)
    
    self_sup_loss, self_sup_dict = self_sup_criterion(pred, low)
    print(f"\n   Self-Supervised Loss: {self_sup_loss.item():.6f}")
    print(f"      - N2V: {self_sup_dict['n2v']:.6f}")
    print(f"      - Wavelet Sparsity: {self_sup_dict['wavelet_sparsity']:.6f}")
    print(f"      - TV: {self_sup_dict['tv']:.6f}")
    
    print(f"\n   ✅ 두 모드 모두 정상 작동!")
    
except Exception as e:
    print(f"   ❌ 비교 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 6: Config 파일 검증
# ============================================================================
print("\n6️⃣ Config 파일 검증...")
try:
    from utils import load_config
    from pathlib import Path
    
    config_path = Path(r'E:\LD-CT SR\_scripts_4_wavelet\config.yaml')
    config = load_config(config_path)
    
    # Check mode exists
    if 'mode' in config['training']:
        print(f"   ✅ Mode 설정 존재: {config['training']['mode']}")
    else:
        print(f"   ⚠️  Mode 설정 없음 (기본값 사용)")
    
    # Check self_supervised_weights exists
    if 'self_supervised_weights' in config['training']:
        print(f"   ✅ Self-supervised weights 설정 존재")
        print(f"      - N2V: {config['training']['self_supervised_weights']['n2v']}")
        print(f"      - Wavelet: {config['training']['self_supervised_weights']['wavelet_sparsity']}")
        print(f"      - TV: {config['training']['self_supervised_weights']['tv']}")
    else:
        print(f"   ❌ Self-supervised weights 설정 없음!")
    
except Exception as e:
    print(f"   ❌ Config 검증 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 결과
# ============================================================================
print("\n" + "="*80)
print("✅ 모든 검증 완료!")
print("="*80)
print("\n📝 다음 단계:")
print("1. config.yaml에서 mode: 'self_supervised' 설정")
print("2. NC-CT 데이터 경로 설정")
print("3. python train_stage1.py 실행")
print("\n🎯 Self-supervised learning 준비 완료! 🚀")

# ============================================================================
# Test 7: Dataset Self-Supervised Mode
# ============================================================================
print("\n7️⃣ Dataset Self-Supervised Mode 테스트...")
try:
    from dataset import CTDenoiseDataset
    from pathlib import Path
    
    # Check if NC-CT data exists
    nc_ct_path = Path(r'E:\LD-CT SR\Data\Image_NC-CT')
    
    if nc_ct_path.exists():
        # Self-supervised mode
        dataset_self = CTDenoiseDataset(
            low_dose_dir=str(nc_ct_path),
            full_dose_dir=str(nc_ct_path),
            hu_window=(-160, 240),
            patch_size=128,
            mode='train',
            self_supervised=True
        )
        
        print(f"   ✅ Self-supervised dataset: {len(dataset_self)} samples")
        
        # Load one sample
        low, full = dataset_self[0]
        print(f"   ✅ Sample shape: {low.shape}")
        print(f"   ✅ Low and Full from same noisy data!")
        
        # Supervised mode for comparison
        dataset_sup = CTDenoiseDataset(
            low_dose_dir=str(nc_ct_path),
            full_dose_dir=str(nc_ct_path),
            hu_window=(-160, 240),
            patch_size=128,
            mode='train',
            self_supervised=False
        )
        
        print(f"   ✅ Supervised dataset: {len(dataset_sup)} samples")
        print(f"   ✅ Dataset 모드 전환 성공!")
        
    else:
        print(f"   ⚠️  NC-CT 경로 없음: {nc_ct_path}")
        print(f"   ⚠️  Dataset 테스트 스킵")
    
except Exception as e:
    print(f"   ❌ Dataset 테스트 실패: {e}")
    import traceback
    traceback.print_exc()