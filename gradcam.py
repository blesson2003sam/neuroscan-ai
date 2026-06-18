import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

print("🧠 Brain Tumor AI — Grad-CAM Explainability")
print("=" * 50)

# ── Step 1: Setup classes ──
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Step 2: Load our trained model ──
print("📂 Loading trained model...")
device = torch.device("cpu")

model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)
model.load_state_dict(torch.load("outputs/weights/best_model.pth", map_location=device))
model.eval()
print("✅ Model loaded!")

# ── Step 3: Grad-CAM class ──
# This hooks into the last layer of the model
# and captures which areas activated most
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into the last convolutional layer
        # In EfficientNet this is features[-1]
        target_layer = model.features[-1]

        # Forward hook — captures what the layer sees
        target_layer.register_forward_hook(self.save_activation)

        # Backward hook — captures gradients flowing back
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        # Forward pass
        output = self.model(image_tensor)

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for our predicted class
        output[0, class_idx].backward()

        # Calculate weights — how important each channel is
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Create heatmap by combining weights with activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # keep only positive values
        cam = cam.squeeze().numpy()

        # Normalize to 0-255
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)

        return cam

# ── Step 4: Image preprocessing ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Step 5: Run Grad-CAM on sample images ──
print("\n🔍 Running Grad-CAM on sample MRI images...")

# Pick one sample image from each class
sample_images = []
for cls in CLASS_NAMES:
    folder = f"dataset/val/{cls}"
    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            img_path = os.path.join(folder, files[0])
            sample_images.append((img_path, cls))

print(f"✅ Found {len(sample_images)} sample images")

# Create Grad-CAM object
gradcam = GradCAM(model)

# Create output folder
os.makedirs("outputs/gradcam", exist_ok=True)

# ── Step 6: Generate and save heatmaps ──
fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4 * len(sample_images)))
fig.suptitle("Brain Tumor AI — Grad-CAM Heatmaps", fontsize=16, fontweight="bold")

for i, (img_path, true_class) in enumerate(sample_images):

    # Load and preprocess image
    original_img = Image.open(img_path).convert("RGB")
    original_img = original_img.resize((224, 224))
    img_array = np.array(original_img)

    img_tensor = transform(original_img).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    # Generate Grad-CAM
    # We need gradients so temporarily enable them
    img_tensor.requires_grad = True
    cam = gradcam.generate(img_tensor, predicted_idx.item())

    # Create coloured heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    # Plot: Original | Heatmap | Overlay
    axes[i, 0].imshow(img_array)
    axes[i, 0].set_title(f"Original\nTrue: {true_class}", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(heatmap)
    axes[i, 1].set_title("Grad-CAM Heatmap\n(Red = AI focused here)", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f"Overlay\nPredicted: {predicted_class} ({confidence_pct:.1f}%)", fontsize=10)
    axes[i, 2].axis("off")

    print(f"✅ {true_class:12} → Predicted: {predicted_class:12} | Confidence: {confidence_pct:.1f}%")

# Save the figure
plt.tight_layout()
output_path = "outputs/gradcam/heatmaps.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n🎉 Grad-CAM complete!")
print(f"✅ Heatmap image saved to: {output_path}")
print(f"📂 Open this file to see the AI's focus areas!")
plt.show()