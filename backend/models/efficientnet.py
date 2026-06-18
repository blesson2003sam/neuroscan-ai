import torch
import torch.nn as nn
import torchvision.models as models

# These are the 4 tumor classes our model knows
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Class descriptions shown in the UI
CLASS_INFO = {
    "glioma": {
        "description": "Glioma is a tumor that occurs in the brain and spinal cord.",
        "severity": "high",
        "color": "#E24B4A"
    },
    "meningioma": {
        "description": "Meningioma is a tumor that forms on membranes covering the brain.",
        "severity": "medium",
        "color": "#534AB7"
    },
    "notumor": {
        "description": "No tumor detected. Brain appears normal.",
        "severity": "none",
        "color": "#1D9E75"
    },
    "pituitary": {
        "description": "Pituitary tumor forms in the pituitary gland at brain base.",
        "severity": "medium",
        "color": "#BA7517"
    }
}

def load_model(weights_path: str = "C:/Users/Lenovo/brain-tumor-ai/outputs/weights/best_model.pth"):
    """Load our trained EfficientNet model"""
    print("Loading AI model...")
    
    device = torch.device("cpu")
    
    # Build same model architecture as training
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)
    
    # Load our trained weights
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    model.eval()
    
    print(f"Model loaded successfully!")
    return model, device