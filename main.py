from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# Allow all CORS origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("best.pt")  
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    results = model(image)

    # Extract predictions
    predictions = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # class index
        conf = float(box.conf[0].item())  # confidence
        label = model.names[cls_id]
        predictions.append({"class": label, "confidence": round(conf, 3)})

    return {"predictions": predictions}
