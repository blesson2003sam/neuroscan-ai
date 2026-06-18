import torch
import cv2
import numpy
from ultralytics import YOLO
from fastapi import FastAPI

print("✅ PyTorch version:", torch.__version__)
print("✅ Running on:", "GPU" if torch.cuda.is_available() else "CPU (no GPU — that is fine!)")
print("✅ OpenCV version:", cv2.__version__)
print("✅ All libraries loaded successfully!")
print("")
print("🎉 Your computer is fully ready for the project!")