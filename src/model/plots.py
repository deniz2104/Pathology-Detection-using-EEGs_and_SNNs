import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.constants import PLOTS_DIR

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def plot_training_curves(training_history: dict[str, dict[str, list[float]]], save=True):
    ensure_plots_dir()
    
    n_tasks = len(training_history)
    if n_tasks == 0:
        print("No training history to plot.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (task_name, history) in enumerate(training_history.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (SmoothL1)')
        ax.set_title(f'{task_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(training_history), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training & Validation Loss per Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/training_curves.png")
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for task_name, history in training_history.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], label=task_name, linewidth=2)
        ax2.plot(epochs, history['val_loss'], label=task_name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss - All Tasks', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss - All Tasks', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'training_curves_combined.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/training_curves_combined.png")
    plt.close()


def plot_predictions_vs_actual(pred_df: pd.DataFrame, gt_df: pd.DataFrame, save=True):
    ensure_plots_dir()
    
    pred_df['subject_id_norm'] = pred_df['subject_id'].astype(str).str.upper().str.strip()
    gt_df['subject_id_norm'] = gt_df['participant_id'].astype(str).str.upper().str.strip()
    merged = pred_df.merge(gt_df, on='subject_id_norm', how='inner')
    
    if len(merged) == 0:
        print("No matched subjects for plotting.")
        return
    
    targets = [
        ('P-Factor', 'pred_p_factor', 'p_factor'),
        ('Externalizing', 'pred_ext', 'externalizing'),
        ('Internalizing', 'pred_int', 'internalizing'),
        ('Attention', 'pred_attn', 'attention')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, pred_col, true_col) in enumerate(targets):
        ax = axes[idx]
        y_pred = merged[pred_col].values
        y_true = merged[true_col].values
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(y_true), p(np.sort(y_true)), 'b-', linewidth=2, alpha=0.7, label='Regression Line')
        
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mae = np.mean(np.abs(y_pred - y_true))
        
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\nCorr: {corr:.3f} | MAE: {mae:.3f}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Actual Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'predictions_vs_actual.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/predictions_vs_actual.png")
    plt.close()


def plot_error_distribution(pred_df: pd.DataFrame, gt_df: pd.DataFrame, save=True):
    ensure_plots_dir()
    
    pred_df['subject_id_norm'] = pred_df['subject_id'].astype(str).str.upper().str.strip()
    gt_df['subject_id_norm'] = gt_df['participant_id'].astype(str).str.upper().str.strip()
    merged = pred_df.merge(gt_df, on='subject_id_norm', how='inner')
    
    targets = [
        ('P-Factor', 'pred_p_factor', 'p_factor'),
        ('Externalizing', 'pred_ext', 'externalizing'),
        ('Internalizing', 'pred_int', 'internalizing'),
        ('Attention', 'pred_attn', 'attention')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, pred_col, true_col) in enumerate(targets):
        ax = axes[idx]
        errors = merged[pred_col].values - merged[true_col].values
        
        ax.hist(errors, bins=20, edgecolor='black', alpha=0.7, density=True)
        
        mu, std = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, 1/(std * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/std)**2), 
                'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
        
        ax.axvline(x=0, color='g', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Error Distribution', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Error Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'error_distribution.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/error_distribution.png")
    plt.close()


def plot_correlation_heatmap(pred_df: pd.DataFrame, gt_df: pd.DataFrame, save=True):
    ensure_plots_dir()
    
    pred_df['subject_id_norm'] = pred_df['subject_id'].astype(str).str.upper().str.strip()
    gt_df['subject_id_norm'] = gt_df['participant_id'].astype(str).str.upper().str.strip()
    merged = pred_df.merge(gt_df, on='subject_id_norm', how='inner')
    
    cols_of_interest = ['pred_p_factor', 'pred_ext', 'pred_int', 'pred_attn',
                        'p_factor', 'externalizing', 'internalizing', 'attention']
    
    corr_matrix = merged[cols_of_interest].corr()
    
    rename_map = {
        'pred_p_factor': 'Pred P-Factor', 'pred_ext': 'Pred External',
        'pred_int': 'Pred Internal', 'pred_attn': 'Pred Attention',
        'p_factor': 'True P-Factor', 'externalizing': 'True External',
        'internalizing': 'True Internal', 'attention': 'True Attention'
    }
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    
    ax.set_title('Correlation Matrix: Predictions vs Ground Truth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/correlation_heatmap.png")
    plt.close()


def plot_task_comparison(training_history: dict[str, dict[str, list[float]]], save=True):
    ensure_plots_dir()
    
    if not training_history:
        return
    
    task_names = []
    final_train_loss = []
    final_val_loss = []
    best_val_loss = []
    epochs_to_converge = []
    
    for task_name, history in training_history.items():
        task_names.append(task_name)
        final_train_loss.append(history['train_loss'][-1])
        final_val_loss.append(history['val_loss'][-1])
        best_val_loss.append(min(history['val_loss']))
        epochs_to_converge.append(np.argmin(history['val_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(task_names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, best_val_loss, width, label='Best Val Loss', color='steelblue', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, final_train_loss, width, label='Final Train Loss', color='coral', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(task_names, rotation=45, ha='right')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison by Task', fontweight='bold')
    axes[0].legend()
    axes[0].bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
    
    colors = sns.color_palette("husl", len(task_names))
    bars3 = axes[1].bar(x, epochs_to_converge, color=colors, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(task_names, rotation=45, ha='right')
    axes[1].set_ylabel('Epochs')
    axes[1].set_title('Epochs to Best Validation Loss', fontweight='bold')
    axes[1].bar_label(bars3, padding=3)
    
    plt.suptitle('Task-wise Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'task_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/task_comparison.png")
    plt.close()


def generate_all_plots(training_history: dict[str, dict[str, list[float]]], 
                       predictions_path: str = "final_predictions.csv",
                       gt_df: pd.DataFrame = None):
    
    ensure_plots_dir()
    
    if training_history:
        plot_training_curves(training_history)
        plot_task_comparison(training_history)
    
    try:
        pred_df = pd.read_csv(predictions_path)
        
        if gt_df is not None:
            plot_predictions_vs_actual(pred_df, gt_df)
            plot_error_distribution(pred_df, gt_df)
            plot_correlation_heatmap(pred_df, gt_df)
    except FileNotFoundError:
        print(f"Predictions file not found: {predictions_path}")