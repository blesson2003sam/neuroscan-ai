# This script reorganises your dataset into the format YOLOv11 expects
import os
import shutil

print("📂 Reorganising dataset folders...")

# Your 4 tumor classes
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Create train and val folders for each class
for cls in classes:
    os.makedirs(f"dataset/train/{cls}", exist_ok=True)
    os.makedirs(f"dataset/val/{cls}", exist_ok=True)
    print(f"✅ Created folders for: {cls}")

# Move Training images → dataset/train
for cls in classes:
    src = f"dataset/Training/{cls}"
    dst = f"dataset/train/{cls}"
    if os.path.exists(src):
        files = os.listdir(src)
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"✅ Copied {len(files)} training images for: {cls}")

# Move Testing images → dataset/val
for cls in classes:
    src = f"dataset/Testing/{cls}"
    dst = f"dataset/val/{cls}"
    if os.path.exists(src):
        files = os.listdir(src)
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"✅ Copied {len(files)} validation images for: {cls}")

print("")
print("🎉 Dataset ready! Your folder structure is now correct.")
print("   Now run: python train.py")