

def erlystop(history, patience=5, min_delta=0.001):
    if len(history) < patience:
        return False
    last_losses = history[-patience:]
    for i in range(patience):
        if last_losses[i] - last_losses[i-1] > min_delta:
            return False
    return True

def best_epoch(history):
    last_val_losses = history["val_loss"]
    last_train_losses = history["train_loss"]
    
    gapp = history.get("gap",10000)

    gap = abs(last_val_losses[-1] - last_train_losses[-1])

    if gap < gapp :
        history["gap"] = gap
        return True
    
    return False
    