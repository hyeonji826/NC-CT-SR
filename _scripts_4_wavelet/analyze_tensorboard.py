"""
N2N Training TensorBoard Analyzer
Current training progress analysis
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def analyze_n2n_training(log_dir):
    """Analyze N2N training logs"""
    
    print("="*80)
    print("N2N Training Analysis")
    print("="*80)
    
    # Find event file
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return
    
    event_file = event_files[0]
    print(f"\nLog file: {event_file.name}")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Extract data
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    print(f"\nLoaded tags: {list(data.keys())}")
    
    # ========================================
    # 1. BALANCE RATIO ANALYSIS (Most Important!)
    # ========================================
    print("\n" + "="*80)
    print("1. BALANCE RATIO ANALYSIS (Critical!)")
    print("="*80)
    
    if 'Train/balance_ratio' in data:
        ratios = np.array(data['Train/balance_ratio']['values'])
        steps = data['Train/balance_ratio']['steps']
        
        # Recent analysis
        recent = ratios[-100:] if len(ratios) > 100 else ratios
        
        print(f"\n   Current balance ratio: {ratios[-1]:.2f}")
        print(f"   Target: ~20.0")
        print(f"   Recent mean (last 100 steps): {recent.mean():.2f}")
        print(f"   Recent std: {recent.std():.2f}")
        
        # Status
        current = ratios[-1]
        if current < 10:
            print(f"   âŒ WARNING: Ratio too low! Wavelet too strong â†’ Oversmoothing risk")
        elif current < 15:
            print(f"   âš ï¸  CAUTION: Ratio slightly low, monitor for blur")
        elif current <= 25:
            print(f"   âœ… GOOD: Balance is optimal")
        elif current <= 30:
            print(f"   âš ï¸  CAUTION: Ratio slightly high, wavelet weak")
        else:
            print(f"   âŒ WARNING: Ratio too high! Wavelet too weak â†’ Overfitting risk")
        
        # Trend
        if len(ratios) > 50:
            first_half = ratios[:len(ratios)//2].mean()
            second_half = ratios[len(ratios)//2:].mean()
            trend = second_half - first_half
            
            print(f"\n   Trend:")
            print(f"      First half mean: {first_half:.2f}")
            print(f"      Second half mean: {second_half:.2f}")
            print(f"      Change: {trend:+.2f}")
            
            if abs(trend) < 2:
                print(f"      Status: Stable (good)")
            elif trend < -5:
                print(f"      Status: Decreasing (expected, monitor)")
            else:
                print(f"      Status: Variable")
    
    # ========================================
    # 2. LOSS COMPONENTS ANALYSIS
    # ========================================
    print("\n" + "="*80)
    print("2. LOSS COMPONENTS BREAKDOWN")
    print("="*80)
    
    if 'Train/n2n_total' in data and 'Train/wavelet' in data:
        n2n = np.array(data['Train/n2n_total']['values'])
        wavelet = np.array(data['Train/wavelet']['values'])
        
        # Recent values
        n2n_recent = n2n[-100:].mean() if len(n2n) > 100 else n2n.mean()
        wavelet_recent = wavelet[-100:].mean() if len(wavelet) > 100 else wavelet.mean()
        
        total_recent = n2n_recent + wavelet_recent
        n2n_percent = n2n_recent / total_recent * 100
        wavelet_percent = wavelet_recent / total_recent * 100
        
        print(f"\n   Recent loss composition (last 100 steps):")
        print(f"      N2N:     {n2n_recent:.6f} ({n2n_percent:.1f}%)")
        print(f"      Wavelet: {wavelet_recent:.6f} ({wavelet_percent:.1f}%)")
        print(f"      Total:   {total_recent:.6f}")
        
        if n2n_percent < 85:
            print(f"      âš ï¸  Wavelet contribution too high!")
        elif n2n_percent > 98:
            print(f"      âš ï¸  Wavelet contribution too low!")
        else:
            print(f"      âœ… Good balance (N2N dominant)")
    
    # N2N internal breakdown
    if 'Train/n2n_rec' in data and 'Train/n2n_reg' in data:
        rec = np.array(data['Train/n2n_rec']['values'])
        reg = np.array(data['Train/n2n_reg']['values'])
        
        rec_recent = rec[-100:].mean() if len(rec) > 100 else rec.mean()
        reg_recent = reg[-100:].mean() if len(reg) > 100 else reg.mean()
        
        print(f"\n   N2N breakdown:")
        print(f"      Reconstruction: {rec_recent:.6f}")
        print(f"      Regularization: {reg_recent:.6f}")
        print(f"      Rec/Reg ratio: {rec_recent/reg_recent:.2f}")
    
    # Adaptive metrics (if available)
    if 'Train/estimated_noise' in data:
        noise = np.array(data['Train/estimated_noise']['values'])
        noise_recent = noise[-100:].mean() if len(noise) > 100 else noise.mean()
        
        print(f"\n   Adaptive metrics:")
        print(f"      Estimated noise: {noise_recent:.4f} (normalized)")
        print(f"      Noise in HU: ~{noise_recent * 400:.1f} HU")
    
    if 'Train/adaptive_weight' in data:
        weights = np.array(data['Train/adaptive_weight']['values'])
        weight_recent = weights[-100:].mean() if len(weights) > 100 else weights.mean()
        
        print(f"      Adaptive wavelet weight: {weight_recent:.6f}")
        print(f"      ✓ Dynamic weighting active")
    
    # ========================================
    # 3. TRAINING PROGRESS
    # ========================================
    print("\n" + "="*80)
    print("3. TRAINING PROGRESS")
    print("="*80)
    
    if 'Epoch/train_loss' in data and 'Epoch/val_loss' in data:
        train_losses = np.array(data['Epoch/train_loss']['values'])
        val_losses = np.array(data['Epoch/val_loss']['values'])
        epochs = data['Epoch/train_loss']['steps']
        
        print(f"\n   Total epochs: {len(epochs)}")
        print(f"   Current epoch: {epochs[-1]}")
        
        # Current losses
        print(f"\n   Current losses:")
        print(f"      Train: {train_losses[-1]:.6f}")
        print(f"      Val:   {val_losses[-1]:.6f}")
        print(f"      Gap:   {(val_losses[-1] - train_losses[-1]):.6f}")
        
        # Best so far
        best_val_idx = np.argmin(val_losses)
        print(f"\n   Best validation:")
        print(f"      Epoch: {epochs[best_val_idx]}")
        print(f"      Loss:  {val_losses[best_val_idx]:.6f}")
        
        # Improvement
        if len(train_losses) >= 5:
            initial = train_losses[:3].mean()
            recent = train_losses[-3:].mean()
            improvement = (initial - recent) / initial * 100
            
            print(f"\n   Overall improvement:")
            print(f"      Initial loss: {initial:.6f}")
            print(f"      Recent loss:  {recent:.6f}")
            print(f"      Improvement:  {improvement:.1f}%")
            
            if improvement > 20:
                print(f"      âœ… Great progress!")
            elif improvement > 5:
                print(f"      âœ… Good progress")
            elif improvement > 0:
                print(f"      âš ï¸  Slow progress")
            else:
                print(f"      âŒ No improvement or worsening")
        
        # Overfitting check
        if len(train_losses) >= 5:
            recent_train = train_losses[-5:].mean()
            recent_val = val_losses[-5:].mean()
            gap = recent_val - recent_train
            gap_percent = gap / recent_train * 100
            
            print(f"\n   Overfitting check (last 5 epochs):")
            print(f"      Val - Train gap: {gap:.6f} ({gap_percent:+.1f}%)")
            
            if gap_percent > 20:
                print(f"      âš ï¸  Possible overfitting (val >> train)")
            elif gap_percent < -5:
                print(f"      âš ï¸  Unusual (train > val)")
            else:
                print(f"      âœ… Normal")
    
    # ========================================
    # 4. LEARNING RATE
    # ========================================
    print("\n" + "="*80)
    print("4. LEARNING RATE SCHEDULE")
    print("="*80)
    
    if 'Epoch/learning_rate' in data:
        lrs = data['Epoch/learning_rate']['values']
        
        print(f"\n   Initial LR: {lrs[0]:.6f}")
        print(f"   Current LR: {lrs[-1]:.6f}")
        print(f"   Reduction:  {(1 - lrs[-1]/lrs[0])*100:.1f}%")
    
    # ========================================
    # 5. VISUALIZATION
    # ========================================
    print("\n" + "="*80)
    print("5. CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Balance Ratio (MOST IMPORTANT!)
    if 'Train/balance_ratio' in data:
        ax = fig.add_subplot(gs[0, :])
        ratios = data['Train/balance_ratio']['values']
        steps = data['Train/balance_ratio']['steps']
        
        ax.plot(steps, ratios, 'b-', linewidth=1, alpha=0.3)
        
        # Moving average
        if len(ratios) > 50:
            window = 50
            ma = np.convolve(ratios, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], ma, 'r-', linewidth=2, label='Moving Avg (50)')
        
        # Target range
        ax.axhspan(15, 25, alpha=0.2, color='green', label='Optimal Range (15-25)')
        ax.axhline(20, color='g', linestyle='--', linewidth=2, label='Target (20)')
        ax.axhline(10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (<10)')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Balance Ratio', fontsize=12)
        ax.set_title('Balance Ratio: N2N / Wavelet (CRITICAL!)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Epoch Losses
    if 'Epoch/train_loss' in data:
        ax = fig.add_subplot(gs[1, 0])
        train = data['Epoch/train_loss']['values']
        val = data['Epoch/val_loss']['values']
        epochs = data['Epoch/train_loss']['steps']
        
        ax.plot(epochs, train, 'b-', linewidth=2, label='Train', marker='o', markersize=3)
        ax.plot(epochs, val, 'r-', linewidth=2, label='Val', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Val Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss Components
    if 'Train/n2n_total' in data:
        ax = fig.add_subplot(gs[1, 1])
        
        if 'Train/n2n_total' in data:
            ax.plot(data['Train/n2n_total']['steps'], 
                   data['Train/n2n_total']['values'], 
                   label='N2N', linewidth=2, alpha=0.7)
        
        if 'Train/wavelet' in data:
            wavelet_vals = np.array(data['Train/wavelet']['values'])
            # Scale up for visibility
            ax.plot(data['Train/wavelet']['steps'], 
                   wavelet_vals * 20,  # Scale for visibility
                   label='Wavelet (Ã—20)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: N2N Breakdown
    if 'Train/n2n_rec' in data:
        ax = fig.add_subplot(gs[1, 2])
        
        ax.plot(data['Train/n2n_rec']['steps'], 
               data['Train/n2n_rec']['values'], 
               label='Reconstruction', linewidth=2)
        
        if 'Train/n2n_reg' in data:
            ax.plot(data['Train/n2n_reg']['steps'], 
                   data['Train/n2n_reg']['values'], 
                   label='Regularization', linewidth=2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('N2N: Rec vs Reg')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate
    if 'Epoch/learning_rate' in data:
        ax = fig.add_subplot(gs[2, 0])
        
        lrs = data['Epoch/learning_rate']['values']
        epochs = data['Epoch/learning_rate']['steps']
        
        ax.plot(epochs, lrs, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot 6: Recent Loss Trend
    if 'Train/total_loss' in data:
        ax = fig.add_subplot(gs[2, 1])
        
        steps = data['Train/total_loss']['steps'][-1000:]
        values = data['Train/total_loss']['values'][-1000:]
        
        ax.plot(steps, values, 'b-', alpha=0.3, linewidth=1)
        
        # Moving average
        if len(values) > 50:
            window = 50
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], ma, 'r-', linewidth=2, label='MA(50)')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Recent Total Loss (Last 1000 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Loss Distribution
    if 'Epoch/train_loss' in data:
        ax = fig.add_subplot(gs[2, 2])
        
        losses = data['Epoch/train_loss']['values']
        
        ax.hist(losses, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(losses), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
        ax.axvline(np.median(losses), color='g', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(losses):.4f}')
        
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Train Loss Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('N2N Training Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = log_dir.parent / 'training_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n   Saved to: {output_file}")
    
    plt.close()
    
    # ========================================
    # FINAL RECOMMENDATIONS
    # ========================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check balance ratio
    if 'Train/balance_ratio' in data:
        current_ratio = data['Train/balance_ratio']['values'][-1]
        if current_ratio < 10:
            recommendations.append("âš ï¸  Reduce wavelet_weight (current ratio < 10)")
        elif current_ratio > 30:
            recommendations.append("âš ï¸  Increase wavelet_weight (current ratio > 30)")
        else:
            recommendations.append("âœ… Balance ratio is good")
    
    # Check progress
    if 'Epoch/train_loss' in data:
        train_losses = np.array(data['Epoch/train_loss']['values'])
        if len(train_losses) >= 10:
            recent_std = train_losses[-10:].std()
            if recent_std < 0.00001:
                recommendations.append("âœ… Loss converged - training can stop")
            elif len(train_losses) > 50 and train_losses[-1] > train_losses[-10]:
                recommendations.append("âš ï¸  Loss increasing - check for issues")
    
    # Check overfitting
    if 'Epoch/train_loss' in data and 'Epoch/val_loss' in data:
        train = np.array(data['Epoch/train_loss']['values'][-5:]).mean()
        val = np.array(data['Epoch/val_loss']['values'][-5:]).mean()
        if val > train * 1.2:
            recommendations.append("âš ï¸  Possible overfitting - consider early stopping")
    
    if recommendations:
        for rec in recommendations:
            print(f"\n   {rec}")
    else:
        print("\n   âœ… Training looks good - continue!")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # Default: find latest experiment
        base_dir = Path(r"E:\LD-CT SR\Outputs\n2n_experiments")
        if base_dir.exists():
            experiments = sorted([d for d in base_dir.iterdir() if d.is_dir()], 
                               key=lambda x: x.stat().st_mtime, reverse=True)
            if experiments:
                log_dir = experiments[0] / 'logs'
            else:
                print("No experiments found")
                sys.exit(1)
        else:
            print(f"Base directory not found: {base_dir}")
            sys.exit(1)
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)
    
    analyze_n2n_training(log_dir)