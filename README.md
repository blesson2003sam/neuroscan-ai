# Brain Tumor Classification

A research and educational web prototype for four-class MRI image classification with Grad-CAM visualizations. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

## Structure

- `backend/` — FastAPI prediction service and EfficientNet-B0 model loader.
- `frontend/` — Next.js browser interface, local scan history, and PDF research report.
- `outputs/` — locally generated training and visualization outputs; do not treat these as deployment artifacts.

## Local development

1. Place the trained model file at `backend/weights/best_model.pth`.
2. In `backend/`, create `.env` from `.env.example`, install `requirements.txt`, and run `uvicorn main:app --reload`.
3. In `frontend/`, create `.env.local` from `.env.example`, set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`, install dependencies, and run `npm run dev`.

## Deployment settings

- **Render:** set `ALLOWED_ORIGINS` to the exact deployed Vercel URL (and any preview URLs that need access).
- **Vercel:** set `NEXT_PUBLIC_API_BASE_URL` to the Render service URL without a trailing slash, then redeploy. This is a build-time public variable.

The frontend is intentionally configured without a fixed service URL in source code so local, staging, and production environments can use different backends.
