import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class WeedDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, img_size=(800,800), transforms=None, return_metadata=False):
        """
        WeedDataset for training and inference.
        
        Args:
            image_dir: Path to directory containing images
            label_dir: Path to directory containing labels (None for inference mode)
            img_size: Target image size as (width, height) tuple
            transforms: Optional transforms to apply (only used in training)
            return_metadata: If True, returns filename and original dimensions
        """
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        self.img_size   = img_size
        self.images     = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        self.transforms = transforms
        self.return_metadata = return_metadata
        self.is_inference = label_dir is None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        fn     = self.images[idx]
        img_path = os.path.join(self.image_dir, fn)
        img    = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Resize
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        img_t = torch.from_numpy(img).permute(2,0,1).float().div(255.0)

        # Inference mode - no labels
        if self.is_inference:
            if self.return_metadata:
                metadata = {
                    "filename": fn,
                    "original_size": (w0, h0),
                    "resized_size": self.img_size
                }
                return img_t, metadata
            return img_t

        # Training mode - load labels (YOLO format: cls cx cy w h)
        boxes, labels = [], []
        lbl_path = os.path.join(self.label_dir, fn.rsplit('.',1)[0] + '.txt')
        if os.path.isfile(lbl_path):
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts)
                cx,cy,bw,bh = cx*w0, cy*h0, bw*w0, bh*h0
                x1,y1 = cx - bw/2, cy - bh/2
                x2,y2 = cx + bw/2, cy + bh/2
                # Scale to resized image
                if self.img_size:
                    sx, sy = self.img_size[0]/w0, self.img_size[1]/h0
                    x1,x2 = x1*sx, x2*sx
                    y1,y2 = y1*sy, y2*sy
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        
        # Add metadata if requested
        if self.return_metadata:
            target["filename"] = fn
            target["original_size"] = (w0, h0)

        # Apply optional transform (e.g., horizontal flip) - only in training
        if self.transforms is not None and not self.is_inference:
            img_t, target = self.transforms(img_t, target)

        return img_t, target

# Simple horizontal flip transform
import random

def hflip_transform(img, target, p=0.5):
    if random.random() < p:
        _, H, W = img.shape
        img = img.flip(-1)
        boxes = target["boxes"]
        x1 = W - boxes[:, 2]
        x2 = W - boxes[:, 0]
        target["boxes"] = torch.stack([x1, boxes[:,1], x2, boxes[:,3]], dim=1)
    return img, target

# DataLoader factory functions

def make_loader(image_dir, label_dir, batch_size=4, shuffle=True, img_size=(800,800), return_metadata=False):
    """
    Create a DataLoader for training/validation.
    
    Args:
        image_dir: Path to images directory
        label_dir: Path to labels directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        img_size: Target image size (width, height)
        return_metadata: Whether to include metadata in targets
    """
    ds = WeedDataset(image_dir, label_dir, img_size=img_size, 
                     transforms=hflip_transform if shuffle else None,
                     return_metadata=return_metadata)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

def make_inference_loader(image_dir, batch_size=1, img_size=(800,800), return_metadata=True):
    """
    Create a DataLoader for inference/predictions.
    
    Args:
        image_dir: Path to images directory
        batch_size: Batch size (default 1 for inference)
        img_size: Target image size (width, height)
        return_metadata: Whether to return filename and original dimensions
    
    Returns:
        DataLoader that yields (images, metadata) tuples
    """
    ds = WeedDataset(image_dir, label_dir=None, img_size=img_size, 
                     transforms=None, return_metadata=return_metadata)
    
    def collate_inference(batch):
        if return_metadata:
            return tuple(zip(*batch))
        else:
            return torch.stack(batch)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_inference
    )