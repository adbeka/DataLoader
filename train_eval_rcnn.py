# File: train_eval_rcnn.py

import os
import time
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from weeds_loader import make_loader
from config_loader import Config

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_one_epoch(model, optimizer, data_loader, device, epoch, config, batch_idx_start=0):
    """Train for one epoch with optional wandb logging."""
    model.train()
    running_loss = 0.0
    batch_idx = 0
    
    for batch_idx, (imgs, targets) in enumerate(data_loader, start=batch_idx_start):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Log batch metrics
        if (batch_idx + 1) % config.logging.log_interval == 0:
            avg_loss_so_far = running_loss / (batch_idx + 1)
            if config.logging.verbose:
                print(f"Epoch {epoch:>2} | Batch {batch_idx + 1:>4} — loss: {avg_loss_so_far:.4f}")
            
            if HAS_WANDB and config.logging.use_wandb:
                wandb.log({
                    "train/loss": avg_loss_so_far,
                    "train/epoch": epoch,
                    "train/batch": batch_idx + 1,
                })
    
    avg_loss = running_loss / len(data_loader)
    if config.logging.verbose:
        print(f"Epoch {epoch:>2} — avg train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, data_loader, device, epoch, config):
    """Evaluate model with mAP metrics and optional wandb logging."""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            gt = [{"boxes": t["boxes"].to(device),
                   "labels": t["labels"].to(device)} for t in targets]
            outputs = model(imgs)
            preds = [{
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu()
            } for out in outputs]

            metric.update(preds, gt)

    results = metric.compute()
    map50 = results["map_50"].item()
    map50_95 = results["map"].item()
    
    if config.logging.verbose:
        print(f"  Eval Results (Epoch {epoch}):")
        print(f"    — mAP@0.50:      {map50:.4f}")
        print(f"    — mAP@0.50–0.95: {map50_95:.4f}")
    
    if HAS_WANDB and config.logging.use_wandb:
        wandb.log({
            "val/map50": map50,
            "val/map50_95": map50_95,
            "epoch": epoch,
        })
    
    return map50, map50_95


def main():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN with config-based setup"
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment (for wandb)")
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)
    device = torch.device(config.device)
    os.makedirs(config.checkpointing.output_dir, exist_ok=True)

    # Initialize wandb
    if HAS_WANDB and config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=args.experiment_name or "weed_detection_rcnn",
            config=config.to_dict(),
        )
        print(f"✓ Weights & Biases initialized: {wandb.run.url}")
    elif config.logging.use_wandb:
        print("⚠ wandb requested but not installed. Install with: pip install wandb")

    # Data loaders
    print("Loading datasets...")
    train_loader = make_loader(
        image_dir=config.data.train_img_dir,
        label_dir=config.data.train_lbl_dir,
        batch_size=config.training.batch_size,
        shuffle=True,
        img_size=tuple(config.data.img_size)
    )
    val_loader = make_loader(
        image_dir=config.data.val_img_dir,
        label_dir=config.data.val_lbl_dir,
        batch_size=config.training.batch_size,
        shuffle=False,
        img_size=tuple(config.data.img_size)
    )
    print(f"✓ Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Model & optimizer
    print(f"Loading {config.model.architecture} model with {config.model.backbone} backbone...")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, config.training.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    # Learning rate scheduler
    if config.training.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    else:
        scheduler = None

    # Training loop
    best_map50 = 0.0
    best_path = None
    
    print(f"\nStarting training for {config.training.num_epochs} epochs...\n")
    for epoch in range(1, config.training.num_epochs + 1):
        start = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch, config)
        train_time = time.time() - start

        if config.logging.verbose:
            print(f"  Evaluating after epoch {epoch} (training took {train_time:.1f}s)...")
        
        map50, map50_95 = evaluate(model, val_loader, device, epoch, config)

        if scheduler:
            scheduler.step()
            if HAS_WANDB and config.logging.use_wandb:
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

        # Save checkpoint
        if epoch % config.checkpointing.save_every_n_epochs == 0:
            ckpt_path = os.path.join(
                config.checkpointing.output_dir, f"fasterrcnn_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            if config.logging.verbose:
                print(f"  Checkpoint saved: {ckpt_path}")

        # Track best
        if map50 > best_map50:
            best_map50 = map50
            best_path = os.path.join(config.checkpointing.output_dir, "best_fasterrcnn.pth")
            torch.save(model.state_dict(), best_path)
            if config.logging.verbose:
                print(f"  ★ New best model saved (mAP@0.50: {best_map50:.4f})")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best mAP@0.50: {best_map50:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"{'='*60}")

    if HAS_WANDB and config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
