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

# Allow ALL origins so Vercel can talk to Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "NeuroScan AI is running!"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B0", "accuracy": "95.19%"}