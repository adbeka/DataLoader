"""
FastAPI inference server for weed detection model.

Usage:
    uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000

Then visit http://localhost:8000/docs for interactive API documentation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from io import BytesIO
from typing import List, Optional
import logging

app = FastAPI(
    title="Weed Detection API",
    description="Real-time weed detection using Faster R-CNN",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
model = None
device = None
model_path = "outputs/best_fasterrcnn.pth"
num_classes = 2
confidence_threshold = 0.5


def load_model(checkpoint_path: str, device_name: str = "cuda"):
    """Load the Faster R-CNN model from checkpoint."""
    global model, device
    
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    try:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_feats = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"✓ Model loaded from {checkpoint_path} on {device}")
        return True
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found: {checkpoint_path}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    if not load_model(model_path):
        logger.warning("Model will be loaded on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    return_image: bool = False
):
    """
    Predict weeds in an image.
    
    Args:
        file: Image file (JPG or PNG)
        confidence_threshold: Minimum confidence for predictions (0-1)
        return_image: Whether to return annotated image as base64
        
    Returns:
        JSON with detected boxes, scores, and labels
    """
    
    if model is None:
        if not load_model(model_path):
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Read image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get original dimensions
        orig_h, orig_w = img_cv.shape[:2]
        
        # Resize to model input size
        img_resized = cv2.resize(img_cv, (800, 800))
        
        # Convert to RGB and torch tensor
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model([img_tensor])
        
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        
        # Scale boxes back to original size
        scale_x = orig_w / 800.0
        scale_y = orig_h / 800.0
        
        # Filter by confidence and format results
        results = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                x1, y1, x2, y2 = box
                results.append({
                    "box": [float(x1 * scale_x), float(y1 * scale_y), 
                           float(x2 * scale_x), float(y2 * scale_y)],
                    "score": float(score),
                    "label": int(label),
                    "label_name": "weed" if label == 1 else "background"
                })
        
        # Optionally return annotated image
        annotated_image_b64 = None
        if return_image:
            import base64
            
            # Draw boxes on original image
            img_annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            for result in results:
                x1, y1, x2, y2 = [int(v) for v in result["box"]]
                score = result["score"]
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_annotated, f"{score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Encode to base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            annotated_image_b64 = base64.b64encode(buffer).decode()
        
        return JSONResponse({
            "filename": file.filename,
            "original_size": {"width": int(orig_w), "height": int(orig_h)},
            "detections": results,
            "num_detections": len(results),
            "annotated_image": annotated_image_b64 if annotated_image_b64 else None
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5
):
    """
    Predict on multiple images.
    
    Args:
        files: List of image files
        confidence_threshold: Minimum confidence for predictions
        
    Returns:
        List of prediction results
    """
    
    results = []
    for file in files:
        try:
            result = await predict(file, confidence_threshold, return_image=False)
            results.append(result.body.decode() if isinstance(result.body, bytes) else result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"batch_results": results, "total_files": len(files)}


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "Weed Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
