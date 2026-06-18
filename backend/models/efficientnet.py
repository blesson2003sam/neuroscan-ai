import torch
import torch.nn as nn
import torchvision.models as models
import os

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

def load_model(weights_path: str = None):
    """Load our trained EfficientNet model"""

    # Try multiple possible paths
    possible_paths = [
        "/app/weights/best_model.pth",
        "weights/best_model.pth",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "best_model.pth"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best_model.pth"),
    ]

    if weights_path:
        possible_paths.insert(0, weights_path)

    # Find which path works
    found_path = None
    for path in possible_paths:
        print(f"Trying path: {path}")
        if os.path.exists(path):
            found_path = path
            print(f"Found model at: {path}")
            break

    if not found_path:
        raise FileNotFoundError(f"Model not found! Tried: {possible_paths}")

    print("Loading AI model...")
    device = torch.device("cpu")
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)
    model.load_state_dict(
        torch.load(found_path, map_location=device)
    )
    model.eval()
    print(f"Model loaded successfully from {found_path}!")
    return model, device