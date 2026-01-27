# Updated DataLoader for Predictions

## Key Improvements

### 1. **Inference Mode**
The dataloader now supports inference without requiring labels:
```python
from weeds_loader import make_inference_loader

# Create inference loader (no labels needed)
loader = make_inference_loader(
    image_dir="path/to/images",
    batch_size=1,
    img_size=(800, 800),
    return_metadata=True
)
```

### 2. **Metadata Tracking**
Returns filename and original image dimensions for proper prediction tracking:
```python
for imgs, metadata_batch in loader:
    for img, meta in zip(imgs, metadata_batch):
        print(f"Processing: {meta['filename']}")
        print(f"Original size: {meta['original_size']}")
        print(f"Resized to: {meta['resized_size']}")
```

### 3. **Coordinate Scaling**
Easily scale predictions back to original image coordinates:
```python
orig_w, orig_h = meta['original_size']
resize_w, resize_h = meta['resized_size']
scale_x = orig_w / resize_w
scale_y = orig_h / resize_h

# Scale predicted box back to original size
x1_orig = x1 * scale_x
y1_orig = y1 * scale_y
```

### 4. **Backward Compatibility**
The original training mode still works exactly as before:
```python
from weeds_loader import make_loader

# Training loader (with labels and transforms)
train_loader = make_loader(
    image_dir="path/to/images",
    label_dir="path/to/labels",
    batch_size=4,
    shuffle=True
)
```

## Usage Examples

### Basic Inference
```python
import torch
from weeds_loader import make_inference_loader

loader = make_inference_loader(
    image_dir="dataset/images/test",
    batch_size=1,
    return_metadata=True
)

for imgs, metadata in loader:
    img = imgs[0].to(device)
    with torch.no_grad():
        predictions = model([img])
    # Process predictions...
```

### Full Prediction Pipeline
See `predict_example.py` for a complete example that includes:
- Loading a trained model
- Running inference with metadata tracking
- Saving predictions to file
- Visualizing results with bounding boxes
- Scaling coordinates back to original image size

### Running the Example
```bash
python predict_example.py
```

## API Reference

### `make_inference_loader(image_dir, batch_size=1, img_size=(800,800), return_metadata=True)`
Create a DataLoader for inference/predictions.

**Parameters:**
- `image_dir`: Path to directory containing images
- `batch_size`: Number of images per batch (default: 1)
- `img_size`: Target size as (width, height) tuple (default: (800, 800))
- `return_metadata`: If True, returns filename and original dimensions (default: True)

**Returns:**
- DataLoader that yields (images, metadata) tuples when `return_metadata=True`

### `make_loader(image_dir, label_dir, batch_size=4, shuffle=True, img_size=(800,800), return_metadata=False)`
Create a DataLoader for training/validation.

**Parameters:**
- `image_dir`: Path to images directory
- `label_dir`: Path to labels directory
- `batch_size`: Number of images per batch (default: 4)
- `shuffle`: Whether to shuffle data (default: True)
- `img_size`: Target size as (width, height) tuple (default: (800, 800))
- `return_metadata`: Whether to include metadata in targets (default: False)

**Returns:**
- DataLoader that yields (images, targets) tuples

## Benefits

1. **No Label Requirement**: Run predictions on images without needing annotation files
2. **Metadata Tracking**: Track which predictions belong to which original images
3. **Coordinate Conversion**: Easy scaling between resized and original image coordinates
4. **Flexible Batching**: Support for batch processing during inference
5. **Clean Separation**: Clear distinction between training and inference modes
6. **Production Ready**: Suitable for deployment in real-world prediction pipelines
