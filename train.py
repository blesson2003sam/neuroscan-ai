import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os

print("🧠 Brain Tumor AI — Pure PyTorch Training")
print("=" * 50)

# ── Step 1: Setup ──
device = torch.device("cpu")
print(f"🖥️  Device: CPU")

# ── Step 2: Data transforms ──
# Training images get random flips/rotations to help learning
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation images just get resized
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Step 3: Load dataset ──
print("\n📂 Loading dataset...")
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
val_data   = datasets.ImageFolder("dataset/val",   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False)

print(f"✅ Training images : {len(train_data)}")
print(f"✅ Validation images: {len(val_data)}")
print(f"✅ Classes: {train_data.classes}")

# ── Step 4: Load model ──
print("\n🤖 Loading EfficientNet-B0 (pretrained)...")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Replace last layer for our 4 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)
model = model.to(device)
print("✅ Model ready!")

# ── Step 5: Training setup ──
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Step 6: Training loop ──
print("\n🚀 Starting training for 20 epochs...")
print("=" * 50)

best_accuracy = 0
os.makedirs("outputs/weights", exist_ok=True)

for epoch in range(1, 21):

    # ── Training phase ──
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Show progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch}/20 | Batch {batch_idx+1}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.3f}")

    train_acc = 100 * correct / total

    # ── Validation phase ──
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"\n{'='*50}")
    print(f"✅ Epoch {epoch}/20 Complete!")
    print(f"   Train Accuracy : {train_acc:.2f}%")
    print(f"   Val Accuracy   : {val_acc:.2f}%")
    print(f"{'='*50}\n")

    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), "outputs/weights/best_model.pth")
        print(f"🏆 Best model saved! Accuracy: {val_acc:.2f}%\n")

print("🎉 Training Complete!")
print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
print("✅ Model saved to: outputs/weights/best_model.pth")