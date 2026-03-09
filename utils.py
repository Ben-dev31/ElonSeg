
from pathlib import Path
from typing import Optional, Tuple, Any
from torchvision import transforms
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import numpy as np
import json
import os
import cv2 
from tifffile import imread

def make_dirs(base):
    p = Path(base)
    imgs = p / 'images'
    masks = p / 'masks'
    imgs.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    return imgs, masks

class PredictionDataset(torch.utils.data.Dataset):
    """Loads images from a directory and returns (tensor, filename)."""
    def __init__(self, images_dir: str, transform=None, target_size: Optional[Tuple[int,int]] = None):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
        if transform:
            self.transform = transform
        else:
            if target_size:
                self.transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
            else:
                self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        return img_t, fname

class SegmentationDataset(Dataset):
    """
    Dataset that expects two folders: images/ and masks/
    - images: RGB images (png/jpg/tiff)
    - masks: single-channel masks (0 background, 255 object) or multi-class index maps
    Both must have same filenames.
    """
    def __init__(self, images_dir: str, masks_dir: str, transform=None, target_size: Optional[Tuple[int,int]] = None, 
                 mode='binary'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        # only keep common image extensions
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.images_dir / fname
        # Try to find mask with the same stem but any common extension
        stem = Path(fname).stem
        mask_path = None
        if self.masks_dir is not None:
            for p in self.masks_dir.iterdir():
                if p.is_file() and p.stem == stem:
                    mask_path = p
                    break
            if mask_path is None:
                # fallback: same filename (may raise later)
                mask_path = self.masks_dir / fname

        image = Image.open(img_path).convert("RGB")
        if mask_path:
            mask = Image.open(mask_path).convert("L")  # grayscale mask
        if self.transform:
            sample = self.transform(image=np.array(image), mask=np.array(mask))
            image = sample['image']
            mask = sample['mask']
        else:
            # minimal transform: resize (if requested) -> to tensor + normalize
            tf_list = []
            if self.target_size:
                tf_list.append(transforms.Resize(self.target_size))
            tf_list.append(transforms.ToTensor())
            tf = transforms.Compose(tf_list)
            image = tf(image)

            # resize mask using PIL nearest if required
            if self.target_size:
                mask = mask.resize(self.target_size, resample=Image.NEAREST)
            if self.mode == 'regression':
                mask = np.array(mask, dtype=np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0).float()  # [0,1], shape [1,H,W]
            else:
                mask = np.array(mask, dtype=np.uint8)
                mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # [0,1], shape [1,H,W]

        return image, mask
    
def make_basic_transform(target_size: Tuple[int,int]=(256,256)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # image -> [C,H,W] float 0..1
    ])
# If you want synchronized augmentations for image+mask, prefer albumentations (not used here).

def load_checkpoint(model: nn.Module, ckpt_path: str, optimizer: Optional[torch.optim.Optimizer]=None, scaler: Optional[GradScaler]=None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # support two formats:
    # 1) {'model_state': ..., 'opt_state': ..., 'epoch':..., 'val_loss':...}
    # 2) state_dict (legacy)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        if optimizer and 'opt_state' in ckpt:
            optimizer.load_state_dict(ckpt['opt_state'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        return ckpt.get('epoch', None), ckpt.get('val_loss', None)
    else:
        # assume ckpt is a state_dict
        model.load_state_dict(ckpt)
        return None, None

def save_in_history(history: dict, save_path: str):
    p = Path(save_path)
    if not p.exists():
        p.touch()
    with p.open('r') as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    existing.update(history)
    with p.open('w') as f:
        json.dump(existing, f, indent=4)



class DistanceMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256,256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # ---- IMAGE ----
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image from {img_path}. Returning a black image.")
            image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # ---- MASK (distance map) ----
        try:
            mask = imread(mask_path) # Open with PIL and convert to grayscale
            mask = np.array(mask) # Convert PIL Image to NumPy array
            # Corrected: cv2.resize expects (width, height), so we swap target_size dimensions
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]))
        except Exception as e:
            print(f"Warning: Could not load mask from {mask_path}: {e}")
            mask = np.zeros(self.target_size, dtype=np.float32)

        mask = mask.astype(np.float32)

        # Normalisation obligatoire
        if mask.max() > 0:
            mask = mask / mask.max()

        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)