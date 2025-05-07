import torch

# ==================== Training Loop ====================

def train_model(model, train_loader, val_loader, config):
    # Configure optimizer (with nanoGPT's version)
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=config['betas'],
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create iterator outside the loop
    train_iter = iter(train_loader)
    
    for step in range(config['max_iters']):
        # Evaluation
        if step % config['eval_interval'] == 0 or step == config['max_iters'] - 1:
            model.eval()
            with torch.no_grad():
                train_loss = evaluate_model(model, train_loader, config['device'])
                val_loss = evaluate_model(model, val_loader, config['device'])
                
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optional: Save model checkpoint
                # torch.save(model.state_dict(), 'best_model.pth')
            
            model.train()
        
        # Training step
        try:
            xb, yb, mask = next(train_iter)
        except StopIteration:
            # Reset iterator if we've exhausted the dataset
            train_iter = iter(train_loader)
            xb, yb, mask = next(train_iter)
        
        xb, yb, mask = xb.to(config['device']), yb.to(config['device']), mask.to(config['device'])
        
        # Forward pass
        _, loss = model(xb, targets=yb, padding_mask=mask)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if config['grad_clip'] != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
    
    return train_losses, val_losses
        

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    
    with torch.no_grad():
        for xb, yb, mask in data_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            _, loss = model(xb, targets=yb, padding_mask=mask)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    
    model.train()
    return total_loss / total_items if total_items > 0 else float('inf')