"""
TensorBoard Log Analyzer
상세한 학습 진행 상황 분석
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def analyze_tensorboard_log(log_file):
    """TensorBoard 로그 파일 상세 분석"""
    
    print("="*80)
    print("TensorBoard Log Analysis")
    print("="*80)
    print(f"\nLog file: {log_file}")
    
    # Load event file
    ea = event_accumulator.EventAccumulator(
        str(log_file),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()
    
    print(f"\nAvailable tags: {ea.Tags()['scalars']}")
    
    # Extract all scalar data
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    # Analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # 1. Check for NaN/Inf
    print("\n1. NaN/Inf Check:")
    has_problem = False
    for tag, d in data.items():
        values = np.array(d['values'])
        nan_count = np.isnan(values).sum()
        inf_count = np.isinf(values).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   {tag}: NaN={nan_count}, Inf={inf_count}")
            has_problem = True
    if not has_problem:
        print("   No NaN/Inf detected")
    
    # 2. Loss value ranges
    print("\n2. Loss Value Ranges:")
    for tag in sorted(data.keys()):
        if 'loss' in tag.lower():
            values = np.array(data[tag]['values'])
            values = values[~np.isnan(values)]  # Remove NaN
            values = values[~np.isinf(values)]  # Remove Inf
            
            if len(values) > 0:
                print(f"\n   {tag}:")
                print(f"      Min:    {values.min():.6f}")
                print(f"      Max:    {values.max():.6f}")
                print(f"      Mean:   {values.mean():.6f}")
                print(f"      Median: {np.median(values):.6f}")
                print(f"      Std:    {values.std():.6f}")
                
                # Check recent trend (last 100 steps)
                if len(values) > 100:
                    recent = values[-100:]
                    print(f"      Recent mean (last 100): {recent.mean():.6f}")
    
    # 3. Loss component sum check
    print("\n3. Loss Component Sum Check:")
    
    # Get train losses at same steps
    train_total_tag = 'Train/loss_total'
    train_self_tag = 'Train/loss_self'
    train_neighbor_tag = 'Train/loss_neighbor'
    train_reg_tag = 'Train/loss_regularization'
    
    if all(tag in data for tag in [train_total_tag, train_self_tag, train_neighbor_tag, train_reg_tag]):
        # Find common steps
        steps_total = set(data[train_total_tag]['steps'])
        steps_self = set(data[train_self_tag]['steps'])
        steps_neighbor = set(data[train_neighbor_tag]['steps'])
        steps_reg = set(data[train_reg_tag]['steps'])
        
        common_steps = steps_total & steps_self & steps_neighbor & steps_reg
        common_steps = sorted(list(common_steps))[-10:]  # Last 10 common steps
        
        print(f"\n   Checking last {len(common_steps)} steps:")
        
        for step in common_steps:
            idx_total = data[train_total_tag]['steps'].index(step)
            idx_self = data[train_self_tag]['steps'].index(step)
            idx_neighbor = data[train_neighbor_tag]['steps'].index(step)
            idx_reg = data[train_reg_tag]['steps'].index(step)
            
            total = data[train_total_tag]['values'][idx_total]
            self_val = data[train_self_tag]['values'][idx_self]
            neighbor_val = data[train_neighbor_tag]['values'][idx_neighbor]
            reg_val = data[train_reg_tag]['values'][idx_reg]
            
            computed_sum = self_val + neighbor_val + reg_val
            diff = abs(total - computed_sum)
            
            print(f"\n   Step {step}:")
            print(f"      Total (logged):    {total:.6f}")
            print(f"      Self:              {self_val:.6f}")
            print(f"      Neighbor:          {neighbor_val:.6f}")
            print(f"      Regularization:    {reg_val:.6f}")
            print(f"      Sum (computed):    {computed_sum:.6f}")
            print(f"      Difference:        {diff:.6f} ({diff/total*100:.2f}%)")
            
            if diff > 0.0001:
                print(f"      WARNING: Large difference!")
    
    # 4. Training trend
    print("\n4. Training Trend (Epoch-level):")
    
    epoch_train_tag = 'Epoch/train_loss'
    epoch_val_tag = 'Epoch/val_loss'
    
    if epoch_train_tag in data and epoch_val_tag in data:
        train_epochs = data[epoch_train_tag]['steps']
        train_losses = data[epoch_train_tag]['values']
        val_epochs = data[epoch_val_tag]['steps']
        val_losses = data[epoch_val_tag]['values']
        
        print(f"\n   Total epochs logged: {len(train_epochs)}")
        
        if len(train_losses) >= 5:
            print(f"\n   First 5 epochs:")
            for i in range(min(5, len(train_losses))):
                print(f"      Epoch {train_epochs[i]}: Train={train_losses[i]:.6f}, Val={val_losses[i]:.6f}")
            
            print(f"\n   Last 5 epochs:")
            for i in range(max(0, len(train_losses)-5), len(train_losses)):
                print(f"      Epoch {train_epochs[i]}: Train={train_losses[i]:.6f}, Val={val_losses[i]:.6f}")
            
            # Check if decreasing
            if len(train_losses) > 1:
                initial = np.mean(train_losses[:3])
                recent = np.mean(train_losses[-3:])
                improvement = (initial - recent) / initial * 100
                
                print(f"\n   Overall improvement: {improvement:.2f}%")
                
                if improvement > 0:
                    print(f"   Status: Loss is decreasing (GOOD)")
                elif improvement < -10:
                    print(f"   Status: Loss is INCREASING significantly (BAD)")
                else:
                    print(f"   Status: Loss is relatively stable")
    
    # 5. Check for overfitting
    print("\n5. Overfitting Check:")
    
    if epoch_train_tag in data and epoch_val_tag in data:
        train_losses = np.array(data[epoch_train_tag]['values'])
        val_losses = np.array(data[epoch_val_tag]['values'])
        
        if len(train_losses) >= 5:
            recent_train = train_losses[-5:].mean()
            recent_val = val_losses[-5:].mean()
            gap = recent_val - recent_train
            gap_percent = gap / recent_train * 100
            
            print(f"\n   Recent 5 epochs:")
            print(f"      Train loss: {recent_train:.6f}")
            print(f"      Val loss:   {recent_val:.6f}")
            print(f"      Gap:        {gap:.6f} ({gap_percent:.2f}%)")
            
            if gap_percent > 20:
                print(f"      WARNING: Possible overfitting (val >> train)")
            elif gap_percent < -20:
                print(f"      WARNING: Unusual (train >> val)")
            else:
                print(f"      Status: Normal gap")
    
    # 6. Learning rate schedule
    print("\n6. Learning Rate Schedule:")
    
    if 'Train/lr' in data or any('lr' in tag.lower() for tag in data.keys()):
        lr_tag = None
        for tag in data.keys():
            if 'lr' in tag.lower():
                lr_tag = tag
                break
        
        if lr_tag:
            lr_values = data[lr_tag]['values']
            print(f"\n   Initial LR: {lr_values[0]:.6f}")
            print(f"   Current LR: {lr_values[-1]:.6f}")
            print(f"   Reduction:  {(1 - lr_values[-1]/lr_values[0])*100:.2f}%")
    
    # 7. Visualization
    print("\n7. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Epoch-level losses
    if epoch_train_tag in data and epoch_val_tag in data:
        ax = axes[0, 0]
        ax.plot(data[epoch_train_tag]['steps'], data[epoch_train_tag]['values'], 'b-', label='Train', linewidth=2)
        ax.plot(data[epoch_val_tag]['steps'], data[epoch_val_tag]['values'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Epoch-level Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Train loss components
    ax = axes[0, 1]
    for tag in [train_total_tag, train_self_tag, train_neighbor_tag, train_reg_tag]:
        if tag in data:
            label = tag.split('/')[-1].replace('loss_', '')
            ax.plot(data[tag]['steps'], data[tag]['values'], label=label, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss distribution (histogram)
    ax = axes[1, 0]
    if train_total_tag in data:
        values = np.array(data[train_total_tag]['values'])
        values = values[~np.isnan(values)]
        values = values[~np.isinf(values)]
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(values.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.6f}')
        ax.axvline(np.median(values), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.6f}')
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Train Loss Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Recent trend (last 1000 steps)
    ax = axes[1, 1]
    if train_total_tag in data:
        steps = data[train_total_tag]['steps'][-1000:]
        values = data[train_total_tag]['values'][-1000:]
        ax.plot(steps, values, 'b-', alpha=0.5, linewidth=1)
        
        # Add moving average
        if len(values) > 50:
            window = 50
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving avg (window={window})')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Recent Training Loss (Last 1000 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(log_file).parent.parent / 'log_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Final assessment
    print("\nFINAL ASSESSMENT:")
    
    if epoch_train_tag in data:
        train_losses = np.array(data[epoch_train_tag]['values'])
        
        # Check loss magnitude
        recent_loss = train_losses[-5:].mean()
        
        print(f"\n1. Loss Magnitude:")
        print(f"   Recent average loss: {recent_loss:.6f}")
        
        if recent_loss < 0.001:
            print(f"   WARNING: Loss is very small - might be too easy or misconfigured")
        elif recent_loss > 1.0:
            print(f"   WARNING: Loss is very large - might have issues")
        else:
            print(f"   Status: Loss magnitude seems reasonable")
        
        # Check convergence
        if len(train_losses) >= 10:
            last_10_std = train_losses[-10:].std()
            print(f"\n2. Convergence:")
            print(f"   Last 10 epochs std: {last_10_std:.6f}")
            
            if last_10_std < 0.00001:
                print(f"   Status: Converged (very stable)")
            elif last_10_std < 0.0001:
                print(f"   Status: Converging (stable)")
            else:
                print(f"   Status: Still training (fluctuating)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = r"E:\LD-CT SR\Outputs\n2n_experiments\n2n_wavelet_20251120_125344\logs\events.out.tfevents.1763610825.DESKTOP-QMNSDLC.6332.0"
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    analyze_tensorboard_log(log_file)