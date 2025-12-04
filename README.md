# NS-N2N CT Denoising - Plan B Implementation

## 📋 Overview

Complete implementation of Plan B strategy for ultra-low-dose CT denoising using Residual-based Neighbor Slice Noise2Noise (NS-N2N) with 3D UNet + Transformer architecture.

**Plan B Strategy: Structure-First, Then Noise Reduction**
- Phase 1 (Epochs 1-30): Strong structure/HU/edge preservation, weak noise term
- Phase 2 (Epochs 31+): Gradually increase noise reduction while maintaining structure

## 🔧 Key Features

### 1. **Input-Based ROI Loss** (Prevents "Scale-Down Cheating")
- Noise loss uses INPUT-based tissue mask (not output-based)
- Flat region detection avoids structure/edge confusion
- Mean-centered std measurement blocks HU scale manipulation

### 2. **High-Pass HU Calculation**
- Removes low-frequency structure via Gaussian filter
- Measures noise only on high-pass residual
- True noise reduction visible in HP residual, not just global darkening

### 3. **Epoch-Based Lambda Scheduling**
```python
# Early epochs (1-30): Structure-first
lambda_rc = 2.0 * 1.5      # Strong reconstruction
lambda_hu = 1.0 * 1.5      # Strong HU preservation
lambda_edge = 0.7 * 1.2    # Strong edge preservation
lambda_noise = 0.05 * 0.1  # Weak noise term

# Later epochs (31+): Gradual noise ramp-up
lambda_noise = 0.05 * (0.3 + 0.7 * t)  # t = progress ratio
```

### 4. **Simplified Sample Output**
```
samples/
├── origin/          # Noisy images
│   ├── epoch_5_HN.png
│   ├── epoch_5_LN.png
│   └── ...
└── denoise/         # Denoised images
    ├── epoch_5_HN.png
    ├── epoch_5_LN.png
    └── ...
```

Console output:
```
Saving samples for epoch 10:
  [HN] Original: 65.3 HU → Denoised: 38.2 HU (41.5% reduction)
  [LN] Original: 54.7 HU → Denoised: 32.1 HU (41.3% reduction)
```

## 📁 File Structure

```
/mnt/user-data/outputs/
├── config_n2n.yaml           # Configuration (Plan B settings)
├── train_n2n.py              # Main training script
├── dataset_n2n.py            # 3D dataset with NS-N2N pairing
├── model_3d_unet_trans.py    # 3D UNet + Transformer
├── losses_n2n.py             # Input-based flat ROI loss
└── utils.py                  # High-pass HU + simple sampling
```

## 🚀 Quick Start

### 1. Configuration

Edit `config_n2n.yaml`:

```yaml
data:
  nc_ct_dir: "YOUR_DATA_PATH"
  output_dir: "YOUR_OUTPUT_PATH"

loss:
  lambda_rc: 2.0        # Strong structure preservation
  lambda_hu: 1.0        # Strong HU preservation
  lambda_edge: 0.7      # Strong edge preservation
  lambda_noise: 0.05    # Weak initially (will ramp up)
  
training:
  noise_warmup_epochs: 30  # Structure-first duration
  batch_size: 8
  num_epochs: 500
```

### 2. Run Training

```bash
python train_n2n.py
```

### 3. Monitor Progress

**TensorBoard:**
```bash
tensorboard --logdir=YOUR_OUTPUT_PATH/residual_denoising_planb/logs
```

**Console Output:**
```
Epoch 001/500 [Structure-First] | Train Loss: 0.0234 | Val Loss: 0.0245 | LR: 5.00e-05
  Lambdas: RC=3.00, Noise=0.005, Edge=0.84, HU=1.50

Epoch 031/500 [Noise-Ramp (0%)] | Train Loss: 0.0189 | Val Loss: 0.0198 | LR: 5.00e-05
  Lambdas: RC=2.00, Noise=0.015, Edge=0.70, HU=1.00
```

## 🔍 Implementation Details

### Training Pipeline Fix

**❌ Previous (Double Residual Bug):**
```python
noise_map = model(x_i_aug).squeeze(2)
denoised = x_center - noise_map  # Wrong: double subtraction
```

**✅ Current (Correct):**
```python
denoised = model(x_i_aug)  # Model handles residual internally
loss, loss_dict = criterion(denoised, batch_dict)
```

### Loss Function Design

**Input-Based Flat ROI (losses_n2n.py):**
```python
# Use INPUT for masking (not output!)
tissue_mask = (x_i > 0.2) & (x_i < 0.8)

# Detect flat regions
grad_mag_input = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
flat_mask = (grad_mag_input < 0.15)

# Noise ROI = body AND flat
noise_roi = tissue_mask & flat_mask

# Mean-centered std (blocks scale manipulation)
denoised_std = (y_pred[roi] - y_pred[roi].mean()).std()
input_std = (x_i[roi] - x_i[roi].mean()).std()
```

### High-Pass HU Calculation (utils.py)

```python
def compute_noise_hu(x_01, hu_window, body_hu_range):
    # Convert to HU
    roi_hu = roi * (hu_max - hu_min) + hu_min
    
    # Remove structure (low-pass)
    lp = gaussian_filter(roi_hu, sigma=1.0)
    hp = roi_hu - lp  # High-pass residual
    
    # Measure noise on HP only
    noise_std_hu = hp[body_mask].std()
    return noise_std_hu
```

## 📊 Expected Behavior

### Phase 1: Structure-First (Epochs 1-30)
- Output looks like clear CT (structure preserved)
- Noise HU may not decrease much yet
- Edge/detail quality improves significantly
- HU values remain stable (no darkening)

### Phase 2: Noise Reduction (Epochs 31+)
- Gradual noise reduction begins
- Structure quality maintained
- HU values stay consistent
- True noise decrease visible in high-pass residual

### Success Indicators
✅ **Good:**
- Clear anatomical structures
- Stable HU values across epochs
- Gradual, consistent noise reduction
- High-pass HU decreases proportionally

❌ **Bad (If These Occur, Tune Config):**
- Image darkening or contrast loss
- Blurred edges/structures
- HU drift (mean shift)
- Noise reduction but image quality degradation

## 🎛️ Hyperparameter Tuning

### If Images Are Too Dark/Low Contrast
```yaml
loss:
  lambda_hu: 1.5      # Increase HU preservation
  lambda_rc: 2.5      # Increase reconstruction
```

### If Edges Are Blurry
```yaml
loss:
  lambda_edge: 1.0    # Increase edge preservation
  lambda_hf: 0.1      # Increase high-frequency preservation
```

### If Noise Reduction Is Too Slow
```yaml
training:
  noise_warmup_epochs: 20  # Shorter warmup

loss:
  lambda_noise: 0.08  # Start with higher base value
```

### If Noise Reduction Is Too Aggressive
```yaml
loss:
  target_noise_ratio: 0.8  # Higher ratio (keep more noise)
```

## 📈 Checkpoint Management

**Auto-saved:**
- Every 10 epochs: `model_epoch_XXX.pth`
- Best model: `best_model.pth`
- Keeps last 5 checkpoints only

**Resume training:**
```yaml
training:
  resume: "path/to/checkpoint.pth"
```

## 🐛 Troubleshooting

### Issue: "No NIfTI files found"
**Solution:** Check `nc_ct_dir` path in config

### Issue: CUDA out of memory
**Solution:** Reduce batch_size or patch_size in config

### Issue: Loss becomes NaN
**Solution:** 
- Reduce learning rate
- Check for corrupted data files
- Verify HU window ranges

### Issue: No visible improvement
**Solution:**
- Verify data quality (check sample outputs)
- Ensure noise_warmup_epochs allows structure to settle
- Check lambda ratios (structure terms should dominate early)

## 📝 Citation

If you use this code, please cite the relevant papers:
- Noise2Noise: Learning Image Restoration without Clean Data (Lehtinen et al., 2018)
- Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images (Huang et al., 2021)

## 📧 Notes

**Key Differences from Original:**
1. ✅ Fixed double residual bug in training loop
2. ✅ Input-based ROI prevents scale-down cheating
3. ✅ High-pass HU calculation for true noise measurement
4. ✅ Plan B epoch scheduling for structure-first learning
5. ✅ Simplified sample output (separate folders, console HU logs)
6. ✅ Disabled adaptive weights (unstable with current metrics)

**All files ready to use!**
Location: `/mnt/user-data/outputs/`