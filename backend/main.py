import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://neuroscan-ai-eta.vercel.app",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Brain Tumor Classification API",
    description="Research prototype API for MRI image classification and Grad-CAM visualization.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "service": "brain-tumor-classification-api"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "brain-tumor-classification-api"}
