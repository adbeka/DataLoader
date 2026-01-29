# Setup Guide for New Features

## 1. Config-Based Training

### What Changed
- Replaced CLI arguments with YAML configuration
- All hyperparameters now in `config/default_config.yaml`
- Easy experiment tracking and reproducibility

### How to Use
```bash
# Default config
python train_eval_rcnn.py

# Custom config
python train_eval_rcnn.py --config config/custom_config.yaml --experiment_name "my_exp"
```

### Example Custom Config
Create `config/custom_config.yaml`:
```yaml
training:
  num_epochs: 30
  batch_size: 8
  lr: 0.0002
  lr_scheduler: "cosine"
  
logging:
  use_wandb: true
  wandb_project: "weed-detection"
```

---

## 2. Weights & Biases Integration

### What's New
- Automatic experiment tracking with **wandb**
- Logs: training loss, mAP, learning rate, hyperparameters
- Compare runs visually in the wandb dashboard

### Setup
```bash
pip install wandb
wandb login  # Follow the prompt
```

### Enable in Config
```yaml
logging:
  use_wandb: true
  wandb_project: "weed-detection"
  wandb_entity: "your-username"  # optional
```

### Features
- ✓ Logs training loss per batch
- ✓ Validates mAP@50 and mAP@50-95
- ✓ Tracks learning rate changes
- ✓ Full hyperparameter logging
- ✓ Model checkpointing

---

## 3. FastAPI Inference Server

### What's New
- REST API for making predictions on images
- Single image and batch inference
- Returns bounding boxes with confidence scores
- Optional image annotation with predictions

### Start Server
```bash
pip install fastapi uvicorn

uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "return_image=true"
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

#### Interactive Docs
Open browser: `http://localhost:8000/docs`

### Response Format
```json
{
  "filename": "image.jpg",
  "original_size": {"width": 1024, "height": 768},
  "detections": [
    {
      "box": [100, 200, 150, 250],
      "score": 0.95,
      "label": 1,
      "label_name": "weed"
    }
  ],
  "num_detections": 5,
  "annotated_image": "base64_encoded_image"
}
```

---

## 4. Model Comparison Dashboard (Jupyter)

### What's New
- Compare Faster R-CNN vs YOLOv8 side-by-side
- Confidence score distributions
- Detection count per image
- Performance metrics summary

### Open Notebook
```bash
jupyter notebook model_comparison_dashboard.ipynb
```

### Features
- Side-by-side visualization of predictions
- Confidence score histograms
- Model statistics and comparisons
- Interactive analysis

---

## Installation

### Full Setup
```bash
# Core dependencies (already have these)
pip install torch torchvision pytorch-metrics opencv-python matplotlib

# Config and experiment tracking
pip install pyyaml wandb

# API server
pip install fastapi uvicorn

# Optional: Jupyter
pip install jupyter
```

### Quick Install
```bash
pip install pyyaml wandb fastapi uvicorn
```

---

## Next Steps

1. **Train with Config**: `python train_eval_rcnn.py --config config/default_config.yaml`
2. **Monitor with wandb**: Check your project dashboard in real-time
3. **Deploy API**: `uvicorn inference_api:app --host 0.0.0.0 --port 8000`
4. **Analyze Results**: Open `model_comparison_dashboard.ipynb`

---

## File Structure
```
DataLoader/
├── config/
│   └── default_config.yaml          # Training configuration
├── config_loader.py                 # Config management
├── train_eval_rcnn.py              # Updated with config + wandb
├── inference_api.py                # FastAPI server (NEW)
├── model_comparison_dashboard.ipynb # Comparison dashboard (NEW)
└── ... (other files)
```

**Happy training! 🚀**
