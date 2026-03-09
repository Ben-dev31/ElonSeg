
import os
import time

import torch
from tqdm import tqdm
from models.utils import erlystop, best_epoch

def train_model(model, train_loader, val_loader, device,
                epochs=50, lr=3e-4, save_path=".",
                model_name="model.pth",
                metrics:dict = None, criterion = None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = criterion or torch.nn.SmoothL1Loss(beta=0.01)  #RootDistanceSmoothLoss(alpha=5.0, beta=0.05)

    best_model = str(save_path) + "\\" + model_name.replace(".pth", "_best.pth")
    last_model = str(save_path) + "\\" + model_name.replace(".pth", "_last.pth")

    history = {}

    history["train_loss"] = []
    history["val_loss"] = []
    history["best_val_loss"] = []
    history["best_epoch"] = []

    for metric_name in metrics.keys():
        history[metric_name] = []

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ========================
        # TRAIN
        # ========================
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader):

            images = images.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ========================
        # VALIDATION
        # ========================
        model.eval()
        for metric_name in metrics.keys():
            history[metric_name].append(0.0)
        
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:

                images = images.to(device)
                targets = targets.to(device).float()

                outputs = model(images)

                loss = criterion(outputs, targets)

                val_loss += loss.item()

                for metric_name, metric_fn in metrics.items():
                    metric_value = metric_fn(outputs, targets).item()
                    history[metric_name][-1] += metric_value

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        for metric_name in metrics.keys():
            history[metric_name][-1] /= len(val_loader)

        # ========================
        # LOG
        # ========================
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss : {train_loss:.6f}, Val Loss   : {val_loss:.6f}")

        # ========================
        # SAVE BEST MODEL
        # ========================
        if best_epoch(history):
            best_val_loss = val_loss

            history["best_val_loss"].append(best_val_loss)
            history["best_epoch"].append(epoch+1)

            torch.save(model.state_dict(), best_model)
            print("✅ Best model saved.")
        
        torch.save(model.state_dict(), last_model)

        # ========================
        stop = erlystop(history["val_loss"], patience=5, min_delta=0.001)

        if stop:
            print("✅ Early stopping triggered.")
            break

    return history


if __name__ == "__main__":

    from models import UNetRegressor
    from utils import DistanceMapDataset
    from torch.utils.data import DataLoader
    import json 
    from metrics import mae, r2_score, rmse

    #hyperparameters
    learning_rate = 1e-4
    num_epochs = 10
    batch_size = 20
    target_size = (512, 512)


    # Paths
    data_path = "D:\\Betterave\\Dataset_mat"
    save_path = "D:\\Betterave\\models"
    model_name = "unet_smoothL1-model1.pth"

    # Datasets and Loaders

    train_dataset = DistanceMapDataset(
    image_dir=os.path.join(data_path, "train/images"),
    mask_dir=os.path.join(data_path, "train/masks"),
    target_size=target_size
    )

    val_dataset = DistanceMapDataset(
        image_dir=os.path.join(data_path, "val/images"),
        mask_dir=os.path.join(data_path, "val/masks"),
        target_size=target_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    # metrics
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2_score
    }

  
    criterion = torch.nn.SmoothL1Loss(beta=0.05) 
    begin = time.time()


    model = UNetRegressor(in_channels=3, n_filters=32).to(device)

    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=num_epochs,
        lr=learning_rate,
        save_path=save_path,
        model_name=model_name,
        metrics=metrics,
        criterion=criterion
    )

    # Save training history
    
    history_path = os.path.join(save_path, f"training_history_{model_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print("✅ Training completed for all criterions.")

    end = time.time()
    print(f"Total training time: {(end - begin)/60:.2f} minutes.")