import datetime
from audio_inference import predict_audio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import shutil
import uuid
import os
import base64
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiohttp
from inference import predict_fake_real, predict_from_image
from prometheus_fastapi_instrumentator import Instrumentator
import time
from prometheus_client import Counter, Histogram
from dotenv import load_dotenv
load_dotenv()

class FeedbackRequest(BaseModel):
    prediction_id: str
    user_actual_result: str
    feedback_text: Optional[str] = None
    score: Optional[int] = None

app = FastAPI()
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    body = await request.body()
    if len(body) > 1024 * 1024 * 10:
        raise HTTPException(status_code=413, detail="Payload too large")
    return await call_next(request)

REQUEST_COUNT = Counter("model_requests_total", "Total number of model requests", ["model_name", "status"])
REQUEST_LATENCY = Histogram("model_request_latency_seconds", "Latency per request", ["model_name"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    model_name = "deepfake_detector"
    start_time = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(model_name).observe(duration)
        REQUEST_COUNT.labels(model_name, status).inc()
    return response

class PredictRequest(BaseModel):
    base64_image: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
@limiter.limit("5/second")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    req: Optional[PredictRequest] = None,
):
    try:
        temp_path = None
        original_filename = None

        if file:
            # Original filename save karo
            original_filename = file.filename
            ext = file.filename.split(".")[-1]
            temp_path = f"temp_{uuid.uuid4()}.{ext}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        elif req:
            if req.image_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(str(req.image_url)) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch image URL")
                        img_bytes = await resp.read()
                        temp_path = f"temp_{uuid.uuid4()}.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(img_bytes)

            elif req.video_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(str(req.video_url)) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch video URL")
                        video_bytes = await resp.read()
                        temp_path = f"temp_{uuid.uuid4()}.mp4"
                        with open(temp_path, "wb") as f:
                            f.write(video_bytes)

            elif req.base64_image:
                img_bytes = base64.b64decode(req.base64_image)
                temp_path = f"temp_{uuid.uuid4()}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

        if not temp_path:
            raise HTTPException(status_code=400, detail="No valid input provided.")

        # Route to correct model
        if temp_path.endswith(".mp4"):
            result = predict_fake_real(temp_path)
        elif temp_path.endswith((".wav", ".mp3", ".flac", ".m4a")):
            result = predict_audio(temp_path)
        else:
            # Original filename pass karo image detection mein
            result = predict_from_image(temp_path, original_filename)

        os.remove(temp_path)

        if result is None or result[0] is None:
            return {"error": "No faces detected."}

        prediction_id = str(uuid.uuid4())
        label, confidence = result

        prediction_data = {
            "id": prediction_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_result": label,
            "confidence_score": confidence,
            "input_type": "video" if temp_path.endswith(".mp4") else "image"
        }

        return {
            "result": label,
            "confidence": confidence,
            "prediction_id": prediction_id
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    return {
        "status": "success",
        "message": "Feedback submitted successfully.",
        "feedback_id": str(uuid.uuid4())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)