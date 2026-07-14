import gdown
import os
import io
import base64
import hashlib
import time
import tempfile
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ── Download model from Google Drive if not present ───────────────
MODEL_PATH     = "best.pt"
GDRIVE_FILE_ID = "10K8spP0obRmGOCGudCWYnu4soe1ni6e9"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )
    print("Model downloaded successfully")

print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="Hobbiton Investments — AI Damage Inspection API",
    description="AI-powered vehicle damage detection for insurance claims",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

seen_hashes = set()

CLASS_COLORS = {
    "Broken part":  (231, 76,  60),
    "Missing part": (192, 57,  43),
    "Cracked":      (52,  152, 219),
    "Corrosion":    (230, 126, 34),
    "Dent":         (155, 89,  182),
    "Flaking":      (241, 196, 15),
    "Scratch":      (46,  204, 113),
    "Paint chip":   (26,  188, 156),
}

SEVERITY_MAP = {
    "Broken part":  "Severe",
    "Missing part": "Severe",
    "Cracked":      "Severe",
    "Corrosion":    "Moderate",
    "Dent":         "Moderate",
    "Flaking":      "Moderate",
    "Scratch":      "Minor",
    "Paint chip":   "Minor",
}

# ── Location classifier ───────────────────────────────────────────
def classify_location(box: dict, img_width: int, img_height: int) -> str:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rel_x = cx / img_width
    rel_y = cy / img_height
    box_w = (x2 - x1) / img_width
    box_h = (y2 - y1) / img_height
    box_area = box_w * box_h

    if rel_y < 0.40 and box_area > 0.15 and 0.2 < rel_x < 0.8:
        return "Windscreen"
    if rel_y < 0.30:
        if rel_x < 0.35:   return "Hood / Left"
        elif rel_x > 0.65: return "Hood / Right"
        else:               return "Roof" if rel_y < 0.15 else "Hood / Centre"
    if rel_y < 0.50:
        if rel_x < 0.25:              return "Front Left Fender"
        elif rel_x > 0.75:            return "Front Right Fender"
        elif 0.25 <= rel_x <= 0.75:   return "Windscreen / Lower"
    if 0.35 <= rel_y <= 0.70:
        if rel_x < 0.25:              return "Front Left Door"
        elif rel_x > 0.75:            return "Front Right Door"
        elif 0.25 <= rel_x < 0.50:    return "Rear Left Door"
        elif 0.50 <= rel_x <= 0.75:   return "Rear Right Door"
    if rel_y > 0.70:
        if rel_x < 0.30:   return "Front Left Bumper"
        elif rel_x > 0.70: return "Front Right Bumper"
        elif box_area > 0.10: return "Front Bumper"
        else:              return "Lower Body / Sill"
    if rel_y < 0.45 and box_area < 0.05:
        if rel_x < 0.25:   return "Left Headlight"
        elif rel_x > 0.75: return "Right Headlight"
    return "Body Panel"

# ── Helpers ───────────────────────────────────────────────────────
def draw_damage_boxes(image: Image.Image, detections: list) -> str:
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        box   = det["bounding_box"]
        color = CLASS_COLORS.get(det["class"], (255, 255, 255))
        label = f"{det['class']} {int(det['confidence']*100)}%"
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        for t in range(4):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
        lw = len(label) * 9
        draw.rectangle([x1, y1-24, x1+lw, y1], fill=color)
        draw.text((x1+4, y1-20), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def get_overall_severity(detections):
    if not detections: return "None"
    sevs = [SEVERITY_MAP.get(d["class"], "Minor") for d in detections]
    if "Severe"   in sevs: return "Severe"
    if "Moderate" in sevs: return "Moderate"
    return "Minor"

def get_fraud_risk(flags):
    if len(flags) >= 3: return "High"
    if len(flags) >= 1: return "Medium"
    return "Low"

def run_fraud_checks(image: Image.Image, image_bytes: bytes, detections: list):
    flags = []
    img_hash = hashlib.md5(image_bytes).hexdigest()
    if img_hash in seen_hashes:
        flags.append("Duplicate image — this photo was submitted before")
    else:
        seen_hashes.add(img_hash)
    w, h = image.size
    if w < 300 or h < 300:
        flags.append("Image resolution too low — may be a screenshot or thumbnail")
    if len(detections) > 12:
        flags.append(f"Unusually high damage count ({len(detections)} detections) — possible exaggeration")
    preexisting = {"Corrosion", "Flaking", "Paint chip"}
    detected_classes = set(d["class"] for d in detections)
    if detections and detected_classes.issubset(preexisting):
        flags.append("Only pre-existing damage types detected — possible prior damage claim")
    if len(detections) == 0:
        flags.append("No damage detected in submitted image")
    return flags

def parse_detections(results, image):
    detections = []
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        confidence = round(float(box.conf), 3)
        coords     = box.xyxy[0].tolist()
        bbox = {
            "x1": round(coords[0]), "y1": round(coords[1]),
            "x2": round(coords[2]), "y2": round(coords[3])
        }
        detections.append({
            "class":        class_name,
            "severity":     SEVERITY_MAP.get(class_name, "Minor"),
            "confidence":   confidence,
            "location":     classify_location(bbox, image.width, image.height),
            "bounding_box": bbox
        })
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return detections

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Hobbiton Investments AI Damage Inspection API",
            "version": "1.0.0", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/inspect")
async def inspect_damage(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    start_time   = time.time()
    results      = model.predict(image, conf=0.25, verbose=False)
    inference_ms = round((time.time() - start_time) * 1000, 1)
    detections   = parse_detections(results, image)
    fraud_flags  = run_fraud_checks(image, image_bytes, detections)
    fraud_risk   = get_fraud_risk(fraud_flags)
    return JSONResponse(content={
        "claim_id":  f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "image_info": {"filename": file.filename,
                       "dimensions": f"{image.width}x{image.height}",
                       "size_kb": round(len(image_bytes)/1024, 1)},
        "damage_assessment": {"damage_detected": len(detections) > 0,
                              "total_detections": len(detections),
                              "overall_severity": get_overall_severity(detections),
                              "detections": detections,
                              "inference_time_ms": inference_ms},
        "fraud_analysis": {"fraud_risk": fraud_risk,
                           "flags": fraud_flags, "flagged": len(fraud_flags) > 0},
        "recommendation": (
            "APPROVE — No fraud flags, damage assessed" if fraud_risk == "Low" and len(detections) > 0
            else "REVIEW — Fraud flags raised, manual check required" if fraud_risk in ["Medium","High"]
            else "REVIEW — No damage detected, request clearer photo")
    })

@app.post("/inspect/annotated")
async def inspect_damage_annotated(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    max_size = 1280
    if max(image.width, image.height) > max_size:
        ratio = max_size / max(image.width, image.height)
        image = image.resize((int(image.width*ratio), int(image.height*ratio)), Image.LANCZOS)
    start_time    = time.time()
    results       = model.predict(image, conf=0.25, verbose=False)
    inference_ms  = round((time.time() - start_time) * 1000, 1)
    detections    = parse_detections(results, image)
    fraud_flags   = run_fraud_checks(image, image_bytes, detections)
    fraud_risk    = get_fraud_risk(fraud_flags)
    annotated_b64 = draw_damage_boxes(image, detections)
    return JSONResponse(content={
        "claim_id":  f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "image_info": {"filename": file.filename,
                       "dimensions": f"{image.width}x{image.height}",
                       "size_kb": round(len(image_bytes)/1024, 1)},
        "damage_assessment": {"damage_detected": len(detections) > 0,
                              "total_detections": len(detections),
                              "overall_severity": get_overall_severity(detections),
                              "detections": detections,
                              "inference_time_ms": inference_ms},
        "fraud_analysis": {"fraud_risk": fraud_risk,
                           "flags": fraud_flags, "flagged": len(fraud_flags) > 0},
        "recommendation": (
            "APPROVE — No fraud flags, damage assessed" if fraud_risk == "Low" and len(detections) > 0
            else "REVIEW — Fraud flags raised, manual check required" if fraud_risk in ["Medium","High"]
            else "REVIEW — No damage detected, request clearer photo"),
        "annotated_image": {"format": "base64/jpeg", "data": annotated_b64}
    })

@app.post("/inspect/video")
async def inspect_video(file: UploadFile = File(...)):
    allowed = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="File must be a video (MP4, MOV, AVI)")
    video_bytes = await file.read()
    temp_path   = os.path.join(tempfile.gettempdir(),
                               f"vid_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
    try:
        import cv2
        cap          = cv2.VideoCapture(temp_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 1)
        interval     = max(1, int(fps * 0.5))
        frame_num    = 0
        all_dets     = []
        best_frame   = None
        best_conf    = 0
        frames_done  = 0
        start_time   = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_num % interval == 0:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil   = Image.fromarray(rgb)
                ms    = 640
                if max(pil.width, pil.height) > ms:
                    r = ms / max(pil.width, pil.height)
                    pil = pil.resize((int(pil.width*r), int(pil.height*r)), Image.LANCZOS)
                res   = model.predict(pil, conf=0.25, verbose=False)
                frames_done += 1
                for box in res[0].boxes:
                    cn  = res[0].names[int(box.cls)]
                    cf  = round(float(box.conf), 3)
                    co  = box.xyxy[0].tolist()
                    bb  = {"x1":round(co[0]),"y1":round(co[1]),"x2":round(co[2]),"y2":round(co[3])}
                    all_dets.append({"class":cn,"severity":SEVERITY_MAP.get(cn,"Minor"),
                                     "confidence":cf,"location":classify_location(bb,pil.width,pil.height),
                                     "frame":frame_num,"bounding_box":bb})
                    if cf > best_conf:
                        best_conf  = cf
                        best_frame = pil.copy()
            frame_num += 1
        cap.release()
        inference_ms = round((time.time()-start_time)*1000,1)
        if not all_dets:
            return JSONResponse(content={
                "claim_id": f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "video_info": {"filename":file.filename,"duration_sec":duration_sec,
                               "frames_analysed":frames_done,"size_kb":round(len(video_bytes)/1024,1)},
                "damage_assessment": {"damage_detected":False,"total_detections":0,
                                      "overall_severity":"None","detections":[],"inference_time_ms":inference_ms},
                "fraud_analysis": {"fraud_risk":"Medium","flags":["No damage detected in video"],"flagged":True},
                "recommendation": "REVIEW — No damage detected, request clearer video"})
        counts  = Counter(d["class"] for d in all_dets)
        unique  = {}
        for d in all_dets:
            if d["class"] not in unique or d["confidence"] > unique[d["class"]]["confidence"]:
                unique[d["class"]] = d
        summary = sorted(unique.values(), key=lambda x: x["confidence"], reverse=True)
        for d in summary:
            d["occurrences_in_video"] = counts[d["class"]]
        fraud_flags = []
        if best_frame:
            buf = io.BytesIO()
            best_frame.save(buf, format="JPEG")
            fraud_flags = run_fraud_checks(best_frame, buf.getvalue(), summary)
        fraud_risk     = get_fraud_risk(fraud_flags)
        best_frame_b64 = draw_damage_boxes(best_frame, summary) if best_frame else None
        return JSONResponse(content={
            "claim_id":  f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "video_info": {"filename":file.filename,"duration_sec":duration_sec,
                           "total_frames":total_frames,"frames_analysed":frames_done,
                           "size_kb":round(len(video_bytes)/1024,1)},
            "damage_assessment": {"damage_detected":True,"total_detections":len(summary),
                                  "overall_severity":get_overall_severity(summary),
                                  "detections":summary,"inference_time_ms":inference_ms},
            "fraud_analysis": {"fraud_risk":fraud_risk,"flags":fraud_flags,"flagged":len(fraud_flags)>0},
            "recommendation": (
                "APPROVE — No fraud flags, damage assessed" if fraud_risk=="Low" and len(summary)>0
                else "REVIEW — Fraud flags raised, manual check required" if fraud_risk in ["Medium","High"]
                else "REVIEW — No damage detected, request clearer video"),
            "best_frame": {"format":"base64/jpeg","data":best_frame_b64} if best_frame_b64 else None
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
