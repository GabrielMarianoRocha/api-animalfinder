from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
from megadetector.detection.run_detector_batch import load_and_run_detector_batch
from classificador import classify_animal
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = '/tmp/uploads'
OUTPUT_DIR = '/tmp/output'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPLOAD_DIR = './ImageDetector/uploads'
OUTPUT_DIR = './ImageDetector/output'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

colors = {'animal': 'red', 'pessoa': 'blue', 'veículo': 'green'}

@app.get("/")
def home():
    return {"message": "API do Animal Finder está rodando!"}


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    image_id = str(uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = load_and_run_detector_batch('MDV5A', [input_path])
        detections = results[0].get('detections', [])

        image = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        try:
            font = ImageFont.truetype("arial.ttf", 68)
        except:
            font = ImageFont.load_default()

        for det in detections:
            conf = det['conf']
            if conf < 0.5:
                continue

            bbox = det['bbox']
            category_id = det['category']
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(x1 + bbox[2] * width)
            y2 = int(y1 + bbox[3] * height)

            if category_id == 1:
                crop = image.crop((x1, y1, x2, y2))
                label = classify_animal(crop)
            elif category_id == 2:
                label = 'pessoa'
            elif category_id == 3:
                label = 'veículo'
            else:
                label = f"categoria {category_id}"

            color = colors.get(label, 'black')

            draw.rectangle([x1, y1, x2, y2], outline=color, width=10)

            text = f"{label} ({conf:.2f})"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
            draw.text((x1, y1 - text_height), text, fill="white", font=font)

        output_path = os.path.join(OUTPUT_DIR, f"{image_id}_output.jpg")
        image.save(output_path)

        return FileResponse(output_path, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
