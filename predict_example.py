"""
Example script demonstrating how to use the updated dataloader for predictions.
"""
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from weeds_loader import make_inference_loader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(checkpoint_path, num_classes=2, device='cpu'):
    """Load a trained Faster R-CNN model."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def predict_and_save(model, loader, device, score_threshold=0.5, output_file='predictions.txt'):
    """
    Run predictions and save results to file.
    
    Args:
        model: Trained detection model
        loader: Inference dataloader with metadata
        device: Device to run on
        score_threshold: Minimum confidence score
        output_file: Path to save predictions
    """
    model.eval()
    
    with open(output_file, 'w') as f:
        f.write("filename,box_idx,x1,y1,x2,y2,score,label\n")
        
        for imgs, metadata_batch in loader:
            # Move images to device
            imgs = [img.to(device) for img in imgs]
            
            # Run inference
            with torch.no_grad():
                predictions = model(imgs)
            
            # Process each image in batch
            for img_idx, (pred, meta) in enumerate(zip(predictions, metadata_batch)):
                filename = meta['filename']
                orig_w, orig_h = meta['original_size']
                resize_w, resize_h = meta['resized_size']
                
                # Calculate scaling factors to convert back to original image size
                scale_x = orig_w / resize_w
                scale_y = orig_h / resize_h
                
                boxes = pred['boxes'].cpu()
                scores = pred['scores'].cpu()
                labels = pred['labels'].cpu()
                
                # Filter by score threshold
                keep = scores >= score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Save predictions (scaled back to original image size)
                for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    x1, y1, x2, y2 = box.tolist()
                    # Scale back to original size
                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    x2_orig = x2 * scale_x
                    y2_orig = y2 * scale_y
                    
                    f.write(f"{filename},{box_idx},{x1_orig:.2f},{y1_orig:.2f},"
                           f"{x2_orig:.2f},{y2_orig:.2f},{score:.4f},{label}\n")
                
                print(f"Processed {filename}: {len(boxes)} detections")
    
    print(f"Predictions saved to {output_file}")

def visualize_predictions(model, loader, device, num_samples=5, score_threshold=0.5):
    """
    Visualize predictions on sample images.
    
    Args:
        model: Trained detection model
        loader: Inference dataloader with metadata
        device: Device to run on
        num_samples: Number of images to visualize
        score_threshold: Minimum confidence score
    """
    model.eval()
    
    for i, (imgs, metadata_batch) in enumerate(loader):
        if i >= num_samples:
            break
        
        # Get first image in batch
        img = imgs[0].to(device)
        meta = metadata_batch[0]
        
        # Run inference
        with torch.no_grad():
            pred = model([img])[0]
        
        # Filter predictions
        keep = pred['scores'] >= score_threshold
        boxes = pred['boxes'][keep].cpu()
        scores = pred['scores'][keep].cpu()
        labels = pred['labels'][keep].cpu()
        
        # Visualize
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img.permute(1, 2, 0).cpu())
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'Class {label}: {score:.2f}',
                   bbox=dict(facecolor='red', alpha=0.5),
                   fontsize=10, color='white')
        
        ax.set_title(f"{meta['filename']} - {len(boxes)} detections")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    IMAGE_DIR = "/workspaces/DataLoader/dataset/images/test"
    MODEL_PATH = "/workspaces/DataLoader/dataset/outputs/fasterrcnn_epoch3.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCORE_THRESHOLD = 0.5
    
    print(f"Using device: {DEVICE}")
    
    # Create inference loader with metadata
    inference_loader = make_inference_loader(
        image_dir=IMAGE_DIR,
        batch_size=1,
        img_size=(800, 800),
        return_metadata=True
    )
    
    print(f"Loaded {len(inference_loader.dataset)} images for inference")
    
    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, num_classes=2, device=DEVICE)
    
    # Run predictions and save
    print("\nRunning predictions...")
    predict_and_save(
        model, 
        inference_loader, 
        DEVICE,
        score_threshold=SCORE_THRESHOLD,
        output_file='predictions/predictions_detailed.txt'
    )
    
    # Visualize some predictions
    print("\nVisualizing predictions...")
    inference_loader_viz = make_inference_loader(
        image_dir=IMAGE_DIR,
        batch_size=1,
        img_size=(800, 800),
        return_metadata=True
    )
    visualize_predictions(
        model,
        inference_loader_viz,
        DEVICE,
        num_samples=3,
        score_threshold=SCORE_THRESHOLD
    )

if __name__ == "__main__":
    main()
