from src.config.constants import LOW_PRIORITY_TASKS, MEDIUM_PRIORITY_TASKS, HIGH_PRIORITY_TASKS

def calculate_task_weights(tasks, low_weight=1, medium_weight=2, high_weight=3):
    task_points = {}
    total_points = 0
    
    for task in tasks:
        points = 0
        
        is_high = any(t in task for t in HIGH_PRIORITY_TASKS)
        is_medium = any(t in task for t in MEDIUM_PRIORITY_TASKS)
        is_low = any(t in task for t in LOW_PRIORITY_TASKS)
        
        if is_high:
            points = high_weight
        elif is_medium:
            points = medium_weight
        elif is_low:
            points = low_weight
        else:
            points = low_weight 
        
        task_points[task] = points
        total_points += points
        
    if total_points == 0:
        return {task: 0 for task in tasks}
        
    task_weights = {task: (points / total_points) * 100 for task, points in task_points.items()}
    return task_weights
