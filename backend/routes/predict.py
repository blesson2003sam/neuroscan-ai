import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import base64
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.efficientnet import load_model, CLASS_NAMES, CLASS_INFO

router = APIRouter()

# Load model once when server starts
print("Initialising model...")
model, device = load_model()
print("Model ready!")

# Image preprocessing — same as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ── Grad-CAM class ──
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        return cam

gradcam = GradCAM(model)

def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image to base64 string for sending to frontend"""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint — receives MRI image, returns:
    - Predicted class
    - Confidence scores for all 4 classes
    - Grad-CAM heatmap images (base64)
    """
    # ── Validate file ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    # ── Load image ──
    contents = await file.read()
    original_img = Image.open(BytesIO(contents)).convert("RGB")
    original_img_resized = original_img.resize((224, 224))
    img_array = np.array(original_img_resized)

    # ── Preprocess ──
    img_tensor = transform(original_img_resized).unsqueeze(0)

    # ── Predict ──
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)

    # Get all confidence scores
    confidences = {}
    for i, cls in enumerate(CLASS_NAMES):
        confidences[cls] = round(float(probabilities[0][i]) * 100, 2)

    # Get top prediction
    predicted_idx = probabilities.argmax(1).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = confidences[predicted_class]

    # ── Generate Grad-CAM ──
    img_tensor_grad = transform(original_img_resized).unsqueeze(0)
    cam = gradcam.generate(img_tensor_grad, predicted_idx)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_rgb, 0.4, 0)

    # ── Return results ──
    return {
        "success": True,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence,
            "confidences": confidences,
            "info": CLASS_INFO[predicted_class]
        },
        "images": {
            "original": image_to_base64(img_array),
            "heatmap": image_to_base64(heatmap_rgb),
            "overlay": image_to_base64(overlay)
        }
    }

@router.get("/classes")
def get_classes():
    """Returns info about all 4 tumor classes"""
    return {"classes": CLASS_NAMES, "info": CLASS_INFO}