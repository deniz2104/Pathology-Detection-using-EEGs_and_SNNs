from torch.utils.data import Dataset
import numpy as np
import os
import glob
import torch
from torch.nn.utils.rnn import pad_sequence

class SingleTaskDataset(Dataset):
    def __init__(self, root_dir, task_name, labels_df, allowed_subjects=None, augment=False):
        self.file_paths = []
        self.labels = []
        self.augment = augment
        self.time_mask_ratio = 0.1  
        self.feature_mask_ratio = 0.05 
        
        search_pattern = os.path.join(root_dir, "*", f"{task_name}_spikes.npy")
        found_files = glob.glob(search_pattern)
        
        for file_path in found_files:
            subject_folder = os.path.basename(os.path.dirname(file_path)).upper()
            
            if allowed_subjects is not None and subject_folder not in allowed_subjects:
                continue

            self.file_paths.append(file_path)
            row = labels_df[labels_df['participant_id'].astype(str) == subject_folder]

            targets = [row['p_factor'].values[0],
                       row['externalizing'].values[0],
                       row['internalizing'].values[0],
                       row['attention'].values[0]]
            
            self.labels.append(np.array(targets, dtype=float))
        
        if len(self.file_paths) == 0:
            print(f"Warning: No files found for task '{task_name}'")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spikes = np.load(self.file_paths[idx])
        
        if spikes.ndim == 3:
            spikes = np.mean(spikes, axis=1) 
        
        if self.augment:
            if np.random.random() < 0.5:
                num_time_mask = int(spikes.shape[0] * self.time_mask_ratio)
                mask_indices = np.random.choice(spikes.shape[0], num_time_mask, replace=False)
                spikes[mask_indices] = 0
            
            if np.random.random() < 0.5:
                num_feat_mask = int(spikes.shape[1] * self.feature_mask_ratio)
                mask_indices = np.random.choice(spikes.shape[1], num_feat_mask, replace=False)
                spikes[:, mask_indices] = 0
            
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                spikes = spikes * scale
        
        x = torch.tensor(spikes, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return x, y

def custom_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    targets_stacked = torch.stack(targets)
    
    return inputs_padded, targets_stacked
