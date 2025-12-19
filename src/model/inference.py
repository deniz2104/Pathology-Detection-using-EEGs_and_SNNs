import os
import glob
import numpy as np
import torch
from src.model.model import device
from src.config.constants import HIGH_PRIORITY_TASKS, MEDIUM_PRIORITY_TASKS, WEIGHT_MAP


def get_task_priority_weight(task_name):
    """Get the weight for a task based on its priority."""
    if any(t in task_name for t in HIGH_PRIORITY_TASKS):
        return WEIGHT_MAP['HIGH']
    elif any(t in task_name for t in MEDIUM_PRIORITY_TASKS):
        return WEIGHT_MAP['MEDIUM']
    else:
        return WEIGHT_MAP['LOW']


def predict_participant(subject_folder_path, models):
    """Generate predictions for a single participant using all available task models."""
    total_prediction = torch.zeros(4).to(device)
    total_weight = 0.0
    
    npy_files = glob.glob(os.path.join(subject_folder_path, "*_spikes.npy"))
    
    for f_path in npy_files:
        filename = os.path.basename(f_path)
        task_name = filename.replace("_spikes.npy", "")
        
        if task_name not in models:
            continue
            
        spikes = np.load(f_path)

        if spikes.ndim == 3:
            spikes = np.mean(spikes, axis=1) 
            
        x = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0).to(device)
            
        model = models[task_name]
        model.eval()
        with torch.no_grad():
            pred = model(x).squeeze(0)
            
        weight = get_task_priority_weight(task_name)
            
        total_prediction += (pred * weight)
        total_weight += weight
            
    if total_weight == 0:
        return None
    
    final_score = total_prediction / total_weight
    return final_score.cpu().numpy()
