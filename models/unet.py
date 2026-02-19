# unet_trainable.py

from pathlib import Path
from typing import Optional, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import torchvision.transforms as T
import cv2

# ----------------------------
# Architecture U-Net
# ----------------------------
class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch: Optional[int] = None):
        """
        mid_ch: intermediate channels; if None, set to out_ch

        """
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv. Use ConvTranspose2d for learnable upsample."""
    def __init__(self, in_ch, out_ch, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        # x1: decoder feature, x2: skip connection from encoder
        x1 = self.up(x1)
        # pad if needed (when image sizes are odd)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2,
                                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, device=None, input_size: Tuple[int,int]=(256,256)):
        super(UNet, self).__init__()

        self.history = {}
        # store dynamic attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size

        # Encodeur (réutilise Down pour le pooling + conv)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # effectue pool + conv => bottleneck

        # Décodeur (réutilise Up pour upsampling + conv)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        # Sortie
        self.outc = OutConv(64, out_channels)

        # Gestion du GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Décodeur
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.outc(x)
    

class TrainableUNet:
    """
    Classe U-Net avec des méthodes d'entraînement et de prédiction.
    """    
    def __init__(self, model: UNet):
        self.model = model
        self.device = model.device
        self.input_size = model.input_size
        self.history = {}

    def save_history(self, history: dict, save_path: str):
        """
        Sauvegarde l'historique d'entraînement dans un fichier JSON lisible.

        Args:
            history (dict) : dictionnaire contenant l'historique (ex: {'loss': [...], 'val_loss': [...]})
            save_path (str) : chemin du fichier de sauvegarde
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


    def save_checkpoint(self, path: str, optimizer=None, epoch: int = None):
        """
        Sauvegarde un checkpoint complet (optionnellement optimizer et epoch).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {'model_state': self.model.state_dict()}
        if optimizer is not None:
            ckpt['optimizer_state'] = optimizer.state_dict()
        if epoch is not None:
            ckpt['epoch'] = epoch
        torch.save(ckpt, path)
        logging.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, optimizer=None, map_location=None):
        """
        Charge un checkpoint et restaure le modèle (et l'optimizer si fourni).
        Retourne l'epoch si présent dans le checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        epoch = ckpt.get('epoch', None)
        logging.info(f"Loaded checkpoint from {path} (epoch={epoch})")
        return epoch

    def train_model(self, train_loader,val_loader, epochs=10, lr=1e-4, criterion=None, 
                    save_dir: str = "./checkpoints", save_history_path: str = "training_history.json"):
        """
        Entraîne le modèle sur un DataLoader contenant des couples (image, masque).
        Sauvegarde automatiquement le meilleur modèle (écart minimum entre train et validation).

        Args:
            train_loader : torch.utils.data.DataLoader
            epochs (int) : nombre d'époques
            lr (float) : taux d'apprentissage
            criterion : fonction de perte (par défaut BCEWithLogitsLoss)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = criterion or nn.BCEWithLogitsLoss()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_gap = float('inf')
        best_epoch = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss_avg = epoch_loss / max(1, n_batches)
            val_loss = self.validate(val_loader, criterion)
            gap = abs(epoch_loss_avg - val_loss)
            
            # append averaged epoch metrics
            self.history.setdefault('loss', []).append(epoch_loss_avg)
            self.history.setdefault('val_loss', []).append(val_loss)
            self.history.setdefault('gap', []).append(gap)

            # checkpoint dict (save model + optimizer + epoch)
           
            ckpt_path = save_dir / "unet_last_epoch.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state': self.model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, ckpt_path)
            
            # save best model based on minimum gap
            if gap <= best_gap:
                best_gap = gap
                best_epoch = epoch + 1
                best_ckpt_path = save_dir / "unet_best.pth"
                torch.save({
                    'epoch': epoch+1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'gap': gap
                }, best_ckpt_path)
                logging.info(f"✓ Best model saved at epoch {epoch+1} with gap: {gap:.4f}")

            logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss_avg:.4f} - Val Loss: {val_loss:.4f} - Gap: {gap:.4f}")

        logging.info(f"Best model: epoch {best_epoch} with gap {best_gap:.4f}")
        # save history as JSON for human-readability
        self.save_history(self.history, save_history_path)
    
    def validate(self, val_loader, criterion=None):
        """
        Évalue le modèle sur un DataLoader de validation.

        Args:
            val_loader : torch.utils.data.DataLoader
            criterion : fonction de perte (par défaut BCEWithLogitsLoss)

        Returns:
            float : perte moyenne sur le jeu de validation
        """
        self.model.eval()
        criterion = criterion or nn.BCEWithLogitsLoss()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                n_batches += 1
        avg = val_loss / max(1, n_batches)
        logging.info(f"Validation - Avg loss: {avg:.4f}")
        return avg
    
    def prepare_image(self, image_input, color_order: str = 'RGB') -> torch.Tensor:
        """
        Prépare une image pour la prédiction.
        Accepte :
            - chemin vers une image (str)
            - numpy.ndarray (RGB ou BGR)
            - PIL.Image
        
        Retourne :
            torch.Tensor de forme [1, 3, H, W], dtype float32, valeurs normalisées [0,1]
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0)

        elif isinstance(image_input, np.ndarray):
            img = image_input
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if color_order.upper() == 'BGR' and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            # if values look like 0-255 scale, convert to 0-1
            if img.max() > 1.5:
                img = img / 255.0
            img = cv2.resize(img, tuple(self.input_size[::-1]))  # cv2 expects (w,h)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0)

        elif isinstance(image_input, Image.Image):
            transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor()
            ])
            img_tensor = transform(image_input).unsqueeze(0)

        else:
            raise TypeError("Format d'image non reconnu. Utilise un chemin, un numpy.ndarray ou un PIL.Image.")

        return img_tensor.to(dtype=torch.float32)


    
    def predict(self, image_input, color_order: str = 'RGB', threshold: Optional[float] = None, to_numpy: bool = False) -> Any:
        """
        Prédit un masque à partir d'une image.
        Args:
            image_input: chemin / numpy array / PIL.Image
            color_order: 'RGB' or 'BGR' for numpy input
            threshold: si fourni, convertit les probabilités en masque binaire selon ce seuil
            to_numpy: si True, renvoie un ndarray numpy au lieu d'un Tensor
        Returns:
            torch.Tensor or numpy.ndarray: probabilités (C,H,W) ou masque binaire (C,H,W)
        """
        self.model.eval()
        with torch.no_grad():
            img_tensor = self.prepare_image(image_input, color_order=color_order).to(self.device)
            output = self.model(img_tensor)
            probs = torch.sigmoid(output)  # shape [1, C, H, W]
            probs = probs.squeeze(0)  # remove batch dim
            if threshold is not None:
                mask = (probs > threshold).float()
                result = mask
            else:
                result = probs
            if to_numpy:
                return result.cpu().numpy()
            return result

