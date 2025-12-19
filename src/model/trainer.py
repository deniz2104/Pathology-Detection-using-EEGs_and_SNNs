import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.model import TaskSNN, device
from src.model.dataset import custom_collate_fn
from src.model.callbacks import EarlyStopping
from src.config.constants import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS


def train_expert(task_name, dataset_train, dataset_val):
    """
    Train a single expert model for a specific task.
    
    Returns:
        model: Trained model
        history: Dict with 'train_loss' and 'val_loss' lists
    """
    total_len = len(dataset_train)
    train_size = int(0.9 * total_len)
    val_size = total_len - train_size
    
    indices = torch.randperm(total_len).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(dataset_train, train_indices)
    val_subset = torch.utils.data.Subset(dataset_val, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    sample_x, _ = dataset_train[0]
    input_size = sample_x.shape[1]
    print(f"Task: {task_name} | Input Shape: {sample_x.shape} | Feature Size: {input_size}")
    print(f"Training Samples: {train_size} | Validation Samples: {val_size}")
    
    model = TaskSNN(num_inputs=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    loss_fn = nn.SmoothL1Loss()
    
    early_stopping = EarlyStopping(patience=15, verbose=True, restore_best_weights=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Track training history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            preds = model(x)
            loss = loss_fn(preds, y)
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = loss_fn(preds, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if early_stopping(model, avg_val_loss):
            print("Early stopping triggered")
            break
            
    early_stopping.restore(model)
    return model, history
