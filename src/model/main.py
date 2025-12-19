import os
import numpy as np
import pandas as pd
from src.google_drive_utils.fetch_results_from_csv_files import fetch_content_from_csv_files
from src.model.dataset import SingleTaskDataset
from src.model.trainer import train_expert
from src.model.inference import predict_participant
from src.config.constants import ACCEPTED_TASKS, ROOT_DIR, RANDOM_SEED


def main():
    """
    Main training and inference pipeline.
    
    Returns:
        training_history: Dict mapping task_name -> {'train_loss': [...], 'val_loss': [...]}
        df_labels: Ground truth DataFrame for evaluation
    """
    # Load labels
    df_labels = fetch_content_from_csv_files()

    # Split participants into train/test
    all_participant_ids = df_labels['participant_id'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_participant_ids)
    
    split_idx = int(len(all_participant_ids) * 0.8)
    train_subjects = set(all_participant_ids[:split_idx])
    test_subjects = set(all_participant_ids[split_idx:])

    # Normalize target columns
    target_cols = ['p_factor', 'externalizing', 'internalizing', 'attention']
    train_df = df_labels[df_labels['participant_id'].isin(train_subjects)]
    
    means = train_df[target_cols].mean()
    stds = train_df[target_cols].std()
    
    df_labels_norm = df_labels.copy()
    for col in target_cols:
        df_labels_norm[col] = (df_labels[col] - means[col]) / stds[col]

    # Train expert models for each task
    expert_models = {}
    training_history = {}

    for task in ACCEPTED_TASKS:
        full_dataset_train = SingleTaskDataset(ROOT_DIR, task, df_labels_norm, allowed_subjects=train_subjects, augment=True)
        full_dataset_val = SingleTaskDataset(ROOT_DIR, task, df_labels_norm, allowed_subjects=train_subjects, augment=False)
        
        if len(full_dataset_train) > 0:
            model, history = train_expert(task, full_dataset_train, full_dataset_val)
            expert_models[task] = model
            training_history[task] = history
        else:
            print(f"Skipping {task} - Data not found.")

    # Evaluate on test set
    results = []
    
    print("\n--- Evaluating on Test Set ---")
    for subj_id in test_subjects:
        subj_path = os.path.join(ROOT_DIR, subj_id)
        
        if not os.path.exists(subj_path):
            continue
    
        pred_scores_norm = predict_participant(subj_path, expert_models)
        
        if pred_scores_norm is not None:
            pred_scores = pred_scores_norm * stds[target_cols].values + means[target_cols].values
            
            print(f"Subject {subj_id}: {pred_scores}")
            results.append({
                'subject_id': subj_id,
                'pred_p_factor': pred_scores[0],
                'pred_ext': pred_scores[1],
                'pred_int': pred_scores[2],
                'pred_attn': pred_scores[3]
            })
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("final_predictions.csv", index=False)
    print(f"\nResults saved to final_predictions.csv")
    
    return training_history, df_labels


if __name__ == "__main__":
    main()
