# E:\LD-CT SR\_scripts_4_wavelet\test_simple.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm

sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR

from dataset import CTDenoiseDataset

print("="*80)
print("🧪 Simple Training Test (L1 Loss Only)")
print("="*80)

device = torch.device('cuda')
print(f"Device: {device}")

# Dataset
dataset = CTDenoiseDataset(
    low_dose_dir=r"E:\LD-CT SR\Data\Images_low_dose",
    full_dose_dir=r"E:\LD-CT SR\Data\Images_full_dose",
    hu_window=(-160, 240),
    patch_size=128,
    config_aug=None,
    mode='train'
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Simple model
model = SwinIR(
    upscale=1,
    in_chans=1,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6],  # Smaller
    embed_dim=60,  # Smaller
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='',
    resi_connection='1conv'
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("\n🚀 Training 10 iterations...")

for i, (low, full) in enumerate(tqdm(loader, total=10)):
    if i >= 10:
        break
    
    low = low.to(device)
    full = full.to(device)
    
    print(f"\nBatch {i}:")
    print(f"  Input: [{low.min():.4f}, {low.max():.4f}], mean={low.mean():.4f}")
    
    # Forward
    optimizer.zero_grad()
    pred = model(low)
    
    print(f"  Output: [{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")
    print(f"  Has NaN: {torch.isnan(pred).any().item()}")
    print(f"  Has Inf: {torch.isinf(pred).any().item()}")
    
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("  ❌ NaN/Inf detected! Stopping.")
        break
    
    loss = criterion(pred, full)
    print(f"  Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()

print("\n✅ Test complete")