import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router

app = FastAPI(
    title="NeuroScan AI API",
    description="Brain Tumor Detection API using EfficientNet + Grad-CAM",
    version="1.0.0"
)

# Allow Next.js frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "NeuroScan AI is running!"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B0", "accuracy": "95.19%"}