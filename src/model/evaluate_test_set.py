import pandas as pd
import numpy as np
from src.google_drive_utils.fetch_results_from_csv_files import fetch_content_from_csv_files

def evaluate_final_predictions(pred_csv_path="final_predictions.csv"):
    try:
        pred_df = pd.read_csv(pred_csv_path)
        print(f"Loaded predictions from {pred_csv_path}: {len(pred_df)} subjects")
    except FileNotFoundError:
        print(f"Error: File not found at {pred_csv_path}")
        return

    gt_df = fetch_content_from_csv_files()

    pred_df['subject_id_norm'] = pred_df['subject_id'].astype(str).str.upper().str.strip()
    gt_df['subject_id_norm'] = gt_df['participant_id'].astype(str).str.upper().str.strip()
    
    merged = pred_df.merge(gt_df, on='subject_id_norm', how='inner')
    
    targets = [
        ('P-Factor', 'pred_p_factor', 'p_factor'),
        ('Externalizing', 'pred_ext', 'externalizing'),
        ('Internalizing', 'pred_int', 'internalizing'),
        ('Attention', 'pred_attn', 'attention')
    ]

    summary_data = []

    for name, pred_col, true_col in targets:
        y_pred = merged[pred_col].values
        y_true = merged[true_col].values
        abs_diff = np.abs(y_pred - y_true)
        
        mae = np.mean(abs_diff)
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
        
        thresholds = [0.25, 0.5, 1.0]
        acc_results = {}
        for t in thresholds:
            acc = (abs_diff <= t).mean() * 100
            acc_results[f'Acc < {t}'] = acc
            
        print(f"--- {name} ---")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"Corr : {correlation:.4f}")
        for t in thresholds:
            print(f"Accuracy (within +/- {t}): {acc_results[f'Acc < {t}']:.2f}%")
        print("")
        
        row = {
            'Target': name,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation
        }
        row.update(acc_results)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("SUMMARY TABLE")
    print(summary_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    evaluate_final_predictions()
