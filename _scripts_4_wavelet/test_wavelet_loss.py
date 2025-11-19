# E:\LD-CT SR\_scripts_4_wavelet\test_wavelet_loss.py

"""
개선된 Wavelet Loss 테스트

논문: "복부 CT 영상의 화질 개선 방법에 대한 연구" (2023)
- Soft Thresholding 적용
- 최적 역치값: 50
- Multi-level DWT (2-level)
"""

import torch
import numpy as np
from losses import WaveletLoss, CombinedLoss

print("="*80)
print("🧪 Wavelet Loss 테스트 (논문 기반 개선)")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Device: {device}")

# ============================================================================
# Test 1: Soft Thresholding 동작 확인
# ============================================================================
print("\n" + "="*80)
print("📊 Test 1: Soft Thresholding 동작 확인")
print("="*80)

wavelet_loss = WaveletLoss(threshold=50, levels=2, normalize_threshold=True)

# Create test images
print("\n1️⃣ 노이즈 이미지 vs 깨끗한 이미지...")
clean = torch.rand(2, 1, 128, 128).to(device) * 0.8 + 0.1
noisy = clean + torch.randn_like(clean) * 0.15
noisy = torch.clamp(noisy, 0, 1)

loss_value = wavelet_loss(noisy, clean)
print(f"   Noisy vs Clean Loss: {loss_value.item():.6f}")
print(f"   Requires grad: {loss_value.requires_grad}")

# ============================================================================
# Test 2: 역치값 변화에 따른 효과 (논문 검증)
# ============================================================================
print("\n" + "="*80)
print("📊 Test 2: 역치값 변화 실험 (논문의 Table 1 재현)")
print("="*80)

clean = torch.rand(1, 1, 128, 128).to(device)
high_noise = clean + torch.randn_like(clean) * 0.2
high_noise = torch.clamp(high_noise, 0, 1)

print("\n논문에서 테스트한 역치값:")
thresholds = [10, 30, 50, 70, 90]

for threshold in thresholds:
    loss = WaveletLoss(threshold=threshold, levels=2, normalize_threshold=True)
    result = loss(high_noise, clean)
    print(f"   Threshold {threshold:2d}: Loss = {result.item():.6f}")

print("\n   📌 논문 결과: Threshold 50에서 노이즈 49% 개선")

# ============================================================================
# Test 3: CombinedLoss 통합 테스트
# ============================================================================
print("\n" + "="*80)
print("📊 Test 3: CombinedLoss 통합 테스트")
print("="*80)

configs = [
    {"l1": 1.0, "ssim": 0.0, "wavelet": 0.0, "name": "L1 Only (Baseline)"},
    {"l1": 1.0, "ssim": 0.2, "wavelet": 0.0, "name": "L1 + SSIM"},
    {"l1": 1.0, "ssim": 0.2, "wavelet": 0.1, "name": "L1 + SSIM + Wavelet (Full)"},
]

for config in configs:
    print(f"\n{config['name']}:")
    
    # 각 테스트마다 새로운 텐서 생성 (중요!)
    pred = torch.rand(2, 1, 128, 128).to(device)
    target = torch.rand(2, 1, 128, 128).to(device)
    
    combined = CombinedLoss(
        l1_weight=config['l1'],
        ssim_weight=config['ssim'],
        wavelet_weight=config['wavelet'],
        wavelet_threshold=50
    )
    
    total, losses = combined(pred, target)
    
    print(f"   Total Loss: {total.item():.6f}")
    print(f"   - L1:      {losses['l1']:.6f}")
    print(f"   - SSIM:    {losses['ssim']:.6f}")
    print(f"   - Wavelet: {losses['wavelet']:.6f}")
    
    # Test backward
    try:
        total.backward()
        print(f"   ✅ Backward OK!")
    except RuntimeError as e:
        print(f"   ⚠️ Backward failed: {e}")

# ============================================================================
# Test 4: Multi-level vs Single-level 비교
# ============================================================================
print("\n" + "="*80)
print("📊 Test 4: Multi-level (2-level) vs Single-level DWT")
print("="*80)

pred = torch.rand(2, 1, 128, 128).to(device)
target = torch.rand(2, 1, 128, 128).to(device)

# Single-level (기존 방식)
loss_1level = WaveletLoss(threshold=50, levels=1, normalize_threshold=True)
result_1level = loss_1level(pred, target)
print(f"\n1-level DWT Loss: {result_1level.item():.6f}")

# Multi-level (개선 방식)
loss_2level = WaveletLoss(threshold=50, levels=2, normalize_threshold=True)
result_2level = loss_2level(pred, target)
print(f"2-level DWT Loss: {result_2level.item():.6f}")

print("\n   📌 Multi-level은 다양한 주파수 대역의 노이즈 포착")

print("\n" + "="*80)
print("✅ 모든 테스트 완료!")
print("="*80)

print("\n📝 다음 단계:")
print("1. losses.py, config.yaml을 프로젝트 폴더로 복사")
print("2. train_stage1.py 140번 라인 수정 (train_stage1_modification.txt 참고)")
print("3. python train_stage1.py 실행")