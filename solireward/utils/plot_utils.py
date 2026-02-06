import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from matplotlib.gridspec import GridSpec


def neg_log_sigmoid(x):
    """More numerically stable implementation of -log(sigmoid(x))"""
    x = np.array(x)
    result = np.zeros_like(x, dtype=float)
    
    # For numerical stability
    positive_mask = x > 0
    negative_mask = x <= 0
    
    # For positive x: -log(1/(1+exp(-x))) = log(1+exp(-x))
    result[positive_mask] = np.log1p(np.exp(-x[positive_mask]))
    
    # For negative x: -log(1/(1+exp(-x))) = -x + log(1+exp(x))
    result[negative_mask] = -x[negative_mask] + np.log1p(np.exp(x[negative_mask]))
    
    return result


def smooth(arr, window):
    """Apply smoothing to array"""
    arr = np.array([x if x is not None else np.nan for x in arr])
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    arr_smooth = np.convolve(arr, kernel, mode='same')
    return arr_smooth


def plot_distribution(output_dir, group_dict, idx):
    """Plot distribution for a single group with classification analysis"""
    pos_rewards = np.array(group_dict['reward_chosen_tensor'], dtype=float)
    neg_rewards = np.array(group_dict['reward_rejected_tensor'], dtype=float)
    reward_diff = np.array(group_dict['reward_diff'], dtype=float)

    # Filter out NaN and infinite values
    pos_rewards = pos_rewards[np.isfinite(pos_rewards)]
    neg_rewards = neg_rewards[np.isfinite(neg_rewards)]
    reward_diff = reward_diff[np.isfinite(reward_diff)]
    
    # Calculate pairwise differences: all pos vs all neg combinations
    pairwise_diff = pos_rewards[:,None] - neg_rewards[None,:]
    pairwise_diff = pairwise_diff.flatten()

    # Create a larger figure with custom grid layout
    fig = plt.figure(figsize=(24, 16), dpi=120)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create custom grid: first row is 3 columns, second row has 2 colspan + 1 single
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.2])
    
    # Create subplots for first row (1x3 grid)
    axs = np.empty((3, 3), dtype=object)
    for j in range(3):
        axs[0, j] = fig.add_subplot(gs[0, j])
    
    # Second row: one subplot spanning 2 columns + one single subplot
    axs[1, 2] = fig.add_subplot(gs[1, 2])  # ROC curve (single column)
    
    # Third row subplots (for potential future use)
    for j in range(3):
        axs[2, j] = fig.add_subplot(gs[2, j])

    # First row: Histograms (original 3 plots)
    # Subplot 1: Histogram of positive and negative samples
    if len(pos_rewards) > 0:
        axs[0, 0].hist(pos_rewards, bins=20, alpha=0.7, color='tab:green', label='Win Samples', edgecolor='black')
    if len(neg_rewards) > 0:
        axs[0, 0].hist(neg_rewards, bins=20, alpha=0.7, color='tab:red', label='Lose Samples', edgecolor='black')
    axs[0, 0].set_title('Histogram: Win vs Lose Samples', fontsize=14)
    axs[0, 0].set_xlabel('Reward Score', fontsize=12)
    axs[0, 0].set_ylabel('Count', fontsize=12)
    axs[0, 0].legend(fontsize=11)

    # Subplot 2: Histogram of original sample pair differences
    if len(reward_diff) > 0:
        axs[0, 1].hist(reward_diff, bins=min(20, int(np.sqrt(len(reward_diff)))), color='tab:purple', alpha=0.7, edgecolor='black')
        
        # Add statistics for original differences
        mean_orig = np.mean(reward_diff)
        median_orig = np.median(reward_diff)
        pos_orig = np.sum(np.array(reward_diff) > 0)
        zero_orig = np.sum(np.array(reward_diff) == 0)
        neg_orig = np.sum(np.array(reward_diff) < 0)
        total_orig = len(reward_diff)
        
        # Add vertical lines
        axs[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        axs[0, 1].axvline(x=mean_orig, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_orig:.3f}')
        axs[0, 1].axvline(x=median_orig, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_orig:.3f}')
        
        # Add text with statistics
        stats_text = f'Positive: {pos_orig}/{total_orig} ({pos_orig/total_orig*100:.1f}%)\n'
        stats_text += f'Zero: {zero_orig}/{total_orig} ({zero_orig/total_orig*100:.1f}%)\n'
        stats_text += f'Negative: {neg_orig}/{total_orig} ({neg_orig/total_orig*100:.1f}%)'
        axs[0, 1].text(0.02, 0.98, stats_text, transform=axs[0, 1].transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[0, 1].legend(fontsize=9)
    axs[0, 1].set_title('Histogram: Original Sample Pair Differences', fontsize=14)
    axs[0, 1].set_xlabel('Score Difference (Win - Lose)', fontsize=12)
    axs[0, 1].set_ylabel('Count', fontsize=12)

    # Subplot 3: Histogram of all pairwise differences
    if len(pairwise_diff) > 0:
        axs[0, 2].hist(pairwise_diff, bins=min(100, int(np.sqrt(len(pairwise_diff)))), color='tab:blue', alpha=0.7, edgecolor='black')
        
        # Add statistics for pairwise differences
        mean_pair = np.mean(pairwise_diff)
        median_pair = np.median(pairwise_diff)
        pos_pair = np.sum(pairwise_diff > 0)
        zero_pair = np.sum(pairwise_diff == 0)
        neg_pair = np.sum(pairwise_diff < 0)
        total_pair = len(pairwise_diff)
        
        # Add vertical lines
        axs[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        axs[0, 2].axvline(x=mean_pair, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_pair:.3f}')
        axs[0, 2].axvline(x=median_pair, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_pair:.3f}')
        
        # Add text with statistics
        stats_text = f'Positive: {pos_pair}/{total_pair} ({pos_pair/total_pair*100:.1f}%)\n'
        stats_text += f'Zero: {zero_pair}/{total_pair} ({zero_pair/total_pair*100:.1f}%)\n'
        stats_text += f'Negative: {neg_pair}/{total_pair} ({neg_pair/total_pair*100:.1f}%)'
        axs[0, 2].text(0.02, 0.98, stats_text, transform=axs[0, 2].transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[0, 2].legend(fontsize=9)
    axs[0, 2].set_title('Histogram: All Pairwise Differences', fontsize=14)
    axs[0, 2].set_xlabel('Score Difference (Win - Lose)', fontsize=12)
    axs[0, 2].set_ylabel('Count', fontsize=12)

    # Third row: Classification analysis
    # Prepare data for binary classification
    if len(pos_rewards) > 0 and len(neg_rewards) > 0:
        all_scores = np.concatenate([pos_rewards, neg_rewards])
        all_labels = np.concatenate([np.ones(len(pos_rewards)), np.zeros(len(neg_rewards))])
        
        # Define threshold range
        min_score = all_scores.min()
        max_score = all_scores.max()
        thresholds = np.linspace(min_score, max_score, 100)
        
        # Calculate metrics for each threshold
        metrics_results = {
            'threshold': [],
            'accuracy': [],
            'weighted_precision': [],
            'weighted_recall': [],
            'weighted_f1': []
        }
        
        for threshold in thresholds:
            # Binary prediction: score >= threshold -> 1, score < threshold -> 0
            y_pred = (all_scores >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(all_labels, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, y_pred, average='weighted', zero_division=0
            )
            
            metrics_results['threshold'].append(threshold)
            metrics_results['accuracy'].append(acc)
            metrics_results['weighted_precision'].append(precision)
            metrics_results['weighted_recall'].append(recall)
            metrics_results['weighted_f1'].append(f1)
        
        # Find best thresholds
        metrics_results = {k: np.array(v) for k, v in metrics_results.items()}
        best_f1_idx = np.argmax(metrics_results['weighted_f1'])
        best_f1_threshold = metrics_results['threshold'][best_f1_idx]
        best_f1_score = metrics_results['weighted_f1'][best_f1_idx]
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
        auc_score = auc(fpr, tpr)
        
        # Find best ROC point (closest to top-left corner)
        best_roc_idx = np.argmax(tpr - fpr)
        best_roc_fpr = fpr[best_roc_idx]
        best_roc_tpr = tpr[best_roc_idx]
        
        # Define colors for classification metrics
        colors = {
            'accuracy': '#FF6B6B',
            'weighted_precision': '#4ECDC4', 
            'weighted_recall': '#45B7D1',
            'weighted_f1': '#96CEB4'
        }
        
        # Subplot 7: Performance metrics vs threshold (spans 2 columns)
        ax_metrics = fig.add_subplot(gs[1, 0:2])  # Spans columns 0 and 1
        ax_metrics.plot(metrics_results['threshold'], metrics_results['accuracy'], 
                       color=colors['accuracy'], linewidth=3, alpha=0.8, label='Accuracy', marker='o', markersize=2)
        ax_metrics.plot(metrics_results['threshold'], metrics_results['weighted_precision'], 
                       color=colors['weighted_precision'], linewidth=3, alpha=0.8, label='Weighted Precision', marker='s', markersize=2)
        ax_metrics.plot(metrics_results['threshold'], metrics_results['weighted_recall'], 
                       color=colors['weighted_recall'], linewidth=3, alpha=0.8, label='Weighted Recall', marker='^', markersize=2)
        ax_metrics.plot(metrics_results['threshold'], metrics_results['weighted_f1'], 
                       color=colors['weighted_f1'], linewidth=3, alpha=0.8, label='Weighted F1 Score', marker='d', markersize=2)
        
        # Mark best F1 point
        ax_metrics.scatter([best_f1_threshold], [best_f1_score], 
                          color='red', s=150, zorder=10, edgecolors='white', linewidth=3, 
                          label=f'Best F1: {best_f1_score:.3f}@{best_f1_threshold:.3f}')
        
        # Add vertical line for best F1 threshold
        ax_metrics.axvline(x=best_f1_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        ax_metrics.set_title('Binary Classification Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax_metrics.set_xlabel('Threshold', fontsize=12)
        ax_metrics.set_ylabel('Metric Value', fontsize=12)
        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.legend(fontsize=10, loc='best')
        ax_metrics.set_ylim(0, 1.05)
        
        # Subplot 8: ROC curve
        axs[1, 2].plot(fpr, tpr, color='#2E86AB', linewidth=3, alpha=0.8, 
                      label=f'ROC Curve (AUC = {auc_score:.3f})')
        axs[1, 2].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                      label='Random Classifier')
        
        # Mark best ROC point
        axs[1, 2].scatter([best_roc_fpr], [best_roc_tpr], 
                         color='red', s=150, zorder=10, edgecolors='white', linewidth=3,
                         label=f'Best Point: ({best_roc_fpr:.3f}, {best_roc_tpr:.3f})')
        
        axs[1, 2].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axs[1, 2].set_xlabel('False Positive Rate', fontsize=12)
        axs[1, 2].set_ylabel('True Positive Rate', fontsize=12)
        axs[1, 2].grid(True, alpha=0.3)
        axs[1, 2].legend(fontsize=10, loc='lower right')
        axs[1, 2].set_xlim(0, 1)
        axs[1, 2].set_ylim(0, 1)
        
        # Add AUC text annotation on ROC plot
        axs[1, 2].text(0.6, 0.2, f'AUC = {auc_score:.4f}', fontsize=12, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Third row: Additional analysis plots (optional, can be used for future extensions)
        # For now, hide these subplots
        for j in range(3):
            axs[2, j].axis('off')
    else:
        # If no valid data for classification, hide the second and third rows
        for i in range(1, 3):
            for j in range(3):
                axs[i, j].axis('off')

    # Apply grid and styling to visible plots
    for j in range(3):
        axs[0, j].grid(True, linestyle='--', alpha=0.6)
        axs[0, j].tick_params(axis='both', labelsize=10)

    plt.tight_layout(pad=2)
    save_path = os.path.join(output_dir, f'eval_step_{idx:06d}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate metrics to return
    metrics_dict = {
        'save_path': save_path,
        'sample_pair_positive_ratio': 0.0,
        'all_pair_positive_ratio': 0.0,
        'best_f1_score': 0.0
    }
    
    # Calculate sample pair positive difference ratio
    if len(reward_diff) > 0:
        pos_sample_pairs = np.sum(np.array(reward_diff) > 0)
        total_sample_pairs = len(reward_diff)
        metrics_dict['sample_pair_positive_ratio'] = pos_sample_pairs / total_sample_pairs if total_sample_pairs > 0 else 0.0
    
    # Calculate all pair positive difference ratio  
    if len(pairwise_diff) > 0:
        pos_all_pairs = np.sum(pairwise_diff > 0)
        total_all_pairs = len(pairwise_diff)
        metrics_dict['all_pair_positive_ratio'] = pos_all_pairs / total_all_pairs if total_all_pairs > 0 else 0.0
    
    # Get best F1 score (calculated earlier in the function)
    if len(pos_rewards) > 0 and len(neg_rewards) > 0:
        metrics_dict['best_f1_score'] = best_f1_score
    
    return metrics_dict


def plot_eval_results(eval_results, output_dir, eval_step=None, smooth_window=1):
    """
    Plot evaluation results from a single evaluation step
    
    Args:
        eval_results: List of dictionaries containing evaluation results
        output_dir: Directory to save plots
        eval_step: Current evaluation step for naming
        smooth_window: Window size for smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not eval_results:
        print("No evaluation results to plot")
        return
    
    # Extract data from eval_results
    bt_loss = [d.get('bt_loss', None) for d in eval_results]
    reward_chosen = [d.get('chosen_reward_mean', None) for d in eval_results]
    reward_rejected = [d.get('rejected_reward_mean', None) for d in eval_results]
    reward_diff = [c - r if c is not None and r is not None else None for c, r in zip(reward_chosen, reward_rejected)]
    
    # Clean data
    def clean_values(values):
        return [float(x) for x in values if x is not None and not (isinstance(x, (int, float)) and np.isnan(x))]
    
    bt_loss_clean = clean_values(bt_loss)
    reward_chosen_clean = clean_values(reward_chosen)
    reward_rejected_clean = clean_values(reward_rejected)
    reward_diff_clean = clean_values(reward_diff)
    
    # Create group dictionary for distribution plot
    group_dict = {
        'bt_loss': bt_loss_clean,
        'reward_chosen_tensor': reward_chosen_clean,
        'reward_rejected_tensor': reward_rejected_clean,
        'reward_diff': reward_diff_clean,
    }
    
    # Save eval results as JSON
    step_name = f"eval_step_{eval_step}" if eval_step is not None else "eval_latest"
    json_path = os.path.join(output_dir, f'{step_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    # Plot detailed distribution
    plot_idx = eval_step if eval_step is not None else 0
    distribution_metrics = plot_distribution(output_dir, group_dict, plot_idx)
    
    print(f"Saved evaluation plots: {distribution_metrics['save_path']}")
    print(f"Sample pair positive ratio: {distribution_metrics['sample_pair_positive_ratio']:.3f}")
    print(f"All pair positive ratio: {distribution_metrics['all_pair_positive_ratio']:.3f}")
    print(f"Best F1 score: {distribution_metrics['best_f1_score']:.3f}")
    return distribution_metrics


def plot_training_metrics(metrics_history, output_dir):
    """
    Plot training metrics (sample pair positive ratio, all pair positive ratio, best F1) over global steps
    
    Args:
        metrics_history: List of dicts containing {'global_step': int, 'sample_pair_positive_ratio': float, 
                        'all_pair_positive_ratio': float, 'best_f1_score': float}
        output_dir: Directory to save the plot
    """
    if not metrics_history:
        print("No training metrics history to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    global_steps = [m['global_step'] for m in metrics_history]
    sample_pair_ratios = [m['sample_pair_positive_ratio'] for m in metrics_history]
    all_pair_ratios = [m['all_pair_positive_ratio'] for m in metrics_history]
    best_f1_scores = [m['best_f1_score'] for m in metrics_history]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot sample pair positive ratio
    ax1.plot(global_steps, sample_pair_ratios, 'o-', color='tab:blue', linewidth=2, markersize=4)
    ax1.set_title('Sample Pair Positive Difference Ratio', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sample Pair Positive Ratio', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot all pair positive ratio
    ax2.plot(global_steps, all_pair_ratios, 'o-', color='tab:orange', linewidth=2, markersize=4)
    ax2.set_title('All Pair Positive Difference Ratio', fontsize=14, fontweight='bold')
    ax2.set_ylabel('All Pair Positive Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Plot best F1 score
    ax3.plot(global_steps, best_f1_scores, 'o-', color='tab:green', linewidth=2, markersize=4)
    ax3.set_title('Best F1 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Global Step', fontsize=12)
    ax3.set_ylabel('Best F1 Score', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    plt.tight_layout(pad=2)
    
    # Save plot (overwrite existing)
    metrics_plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(metrics_plot_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    
    print(f"Saved training metrics plot: {metrics_plot_path}")
    return metrics_plot_path