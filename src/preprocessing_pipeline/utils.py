import os
import numpy as np
from config import constants

def export_spikes(subject_folder_id, filename, spikes):
    output_dir = "preprocessed_participants"
    subject_dir = os.path.join(output_dir, f"SUB-{subject_folder_id}")
    os.makedirs(subject_dir, exist_ok=True)
    
    task_name = "unknown_task"
    for task in constants.ACCEPTED_TASKS:
        if task in filename:
            task_name = task
            break
    
    output_file = os.path.join(subject_dir, f"{task_name}_spikes.npy")
    np.save(output_file, spikes)
    print(f"Exported spikes for {subject_folder_id} - {task_name} to {output_file}")
