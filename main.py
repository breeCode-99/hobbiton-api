import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import hashlib
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import os
import time
from datetime import datetime

# ── Download model from Google Drive if not present ───────────────
MODEL_PATH    = "best.pt"
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

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────
MODEL_PATH = "best.pt"
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# Store image hashes to detect duplicates
seen_hashes = set()


# Colour per damage class for drawing boxes
CLASS_COLORS = {
    "Broken part":  (231, 76,  60),   # red
    "Missing part": (192, 57,  43),   # dark red
    "Cracked":      (52,  152, 219),  # blue
    "Corrosion":    (230, 126, 34),   # orange
    "Dent":         (155, 89,  182),  # purple
    "Flaking":      (241, 196, 15),   # yellow
    "Scratch":      (46,  204, 113),  # green
    "Paint chip":   (26,  188, 156),  # teal
}

# ── Location classifier ───────────────────────────────────────────
def classify_location(box: dict, img_width: int, img_height: int) -> str:
    """
    Maps bounding box coordinates to a car part location.
    Uses relative position within the image to determine zone.
    """
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

    # Calculate centre point of the bounding box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Normalise to 0-1 range
    rel_x = cx / img_width
    rel_y = cy / img_height

    # Calculate box size relative to image
    box_w = (x2 - x1) / img_width
    box_h = (y2 - y1) / img_height
    box_area = box_w * box_h

    # ── Vertical zones ──
    # Top 30% = upper body (windscreen, roof, hood)
    # Middle 40% = mid body (doors, fenders)
    # Bottom 30% = lower body (bumpers, sills)

    # ── Horizontal zones ──
    # Left 30% = left side
    # Centre 40% = centre
    # Right 30% = right side

    # Large box covering most of upper area = windscreen
    if rel_y < 0.40 and box_area > 0.15 and 0.2 < rel_x < 0.8:
        return "Windscreen"

    # Top zone
    if rel_y < 0.30:
        if rel_x < 0.35:
            return "Hood / Left"
        elif rel_x > 0.65:
            return "Hood / Right"
        else:
            if rel_y < 0.15:
                return "Roof"
            return "Hood / Centre"

    # Upper middle zone
    if rel_y < 0.50:
        if rel_x < 0.25:
            return "Front Left Fender"
        elif rel_x > 0.75:
            return "Front Right Fender"
        elif 0.25 <= rel_x <= 0.75:
            return "Windscreen / Lower"

    # Middle zone — doors
    if 0.35 <= rel_y <= 0.70:
        if rel_x < 0.25:
            return "Front Left Door"
        elif rel_x > 0.75:
            return "Front Right Door"
        elif 0.25 <= rel_x < 0.50:
            return "Rear Left Door"
        elif 0.50 <= rel_x <= 0.75:
            return "Rear Right Door"

    # Lower zone — bumpers and sills
    if rel_y > 0.70:
        if rel_x < 0.30:
            return "Front Left Bumper"
        elif rel_x > 0.70:
            return "Front Right Bumper"
        elif box_area > 0.10:
            return "Front Bumper"
        else:
            return "Lower Body / Sill"

    # Headlight zone (small boxes, upper corners)
    if rel_y < 0.45 and box_area < 0.05:
        if rel_x < 0.25:
            return "Left Headlight"
        elif rel_x > 0.75:
            return "Right Headlight"

    return "Body Panel"

# ── Severity mapping ──────────────────────────────────────────────
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
def draw_damage_boxes(image: Image.Image, detections: list) -> str:
    """Draw bounding boxes on image and return as base64 string."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        box   = det["bounding_box"]
        color = CLASS_COLORS.get(det["class"], (255, 255, 255))
        label = f"{det['class']} {int(det['confidence']*100)}%"

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Draw box (thick border)
        for thickness in range(4):
            draw.rectangle(
                [x1-thickness, y1-thickness, x2+thickness, y2+thickness],
                outline=color
            )

        # Label background
        label_w = len(label) * 9
        label_h = 24
        draw.rectangle([x1, y1-label_h, x1+label_w, y1], fill=color)

        # Label text
        draw.text((x1+4, y1-label_h+4), label, fill=(255, 255, 255))

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
def get_overall_severity(detections):
    if not detections:
        return "None"
    severities = [SEVERITY_MAP.get(d["class"], "Minor") for d in detections]
    if "Severe"   in severities: return "Severe"
    if "Moderate" in severities: return "Moderate"
    return "Minor"

def get_fraud_risk(flags):
    if len(flags) >= 3: return "High"
    if len(flags) >= 1: return "Medium"
    return "Low"

# ── Fraud detection ───────────────────────────────────────────────
def run_fraud_checks(image: Image.Image, image_bytes: bytes, detections: list):
    flags = []

    # Check 1 — Duplicate image
    img_hash = hashlib.md5(image_bytes).hexdigest()
    if img_hash in seen_hashes:
        flags.append("Duplicate image — this photo was submitted before")
    else:
        seen_hashes.add(img_hash)

    # Check 2 — Image too small (low quality submission)
    w, h = image.size
    if w < 300 or h < 300:
        flags.append("Image resolution too low — may be a screenshot or thumbnail")

    # Check 3 — Extremely high number of damage detections
    if len(detections) > 12:
        flags.append(f"Unusually high damage count ({len(detections)} detections) — possible exaggeration")

    # Check 4 — Only pre-existing damage types detected
    preexisting = {"Corrosion", "Flaking", "Paint chip"}
    detected_classes = set(d["class"] for d in detections)
    if detections and detected_classes.issubset(preexisting):
        flags.append("Only pre-existing damage types detected (corrosion, flaking, paint chip) — possible prior damage claim")

    # Check 5 — No damage found but claim submitted
    if len(detections) == 0:
        flags.append("No damage detected in submitted image — image may not show damaged area")

    return flags

# ── Root endpoint ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Hobbiton Investments AI Damage Inspection API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }

# ── Health check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# ── Main inspection endpoint ──────────────────────────────────────
@app.post("/inspect")
async def inspect_damage(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    # Read image
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image — file may be corrupted")

    # Run YOLOv8 inference
    start_time = time.time()
    results     = model.predict(image, conf=0.25, verbose=False)
    inference_ms = round((time.time() - start_time) * 1000, 1)

    # Parse detections
    detections = []
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        confidence = round(float(box.conf), 3)
        coords     = box.xyxy[0].tolist()

        bbox = {
            "x1": round(coords[0]),
            "y1": round(coords[1]),
            "x2": round(coords[2]),
            "y2": round(coords[3])
        }

        # Classify location based on bounding box position
        location = classify_location(bbox, image.width, image.height)

        detections.append({
            "class":        class_name,
            "severity":     SEVERITY_MAP.get(class_name, "Minor"),
            "confidence":   confidence,
            "location":     location,
            "bounding_box": bbox
        })

    # Sort by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # Run fraud checks
    fraud_flags = run_fraud_checks(image, image_bytes, detections)
    fraud_risk  = get_fraud_risk(fraud_flags)

    # Overall severity
    overall_severity = get_overall_severity(detections)

    # Build response
    response = {
        "claim_id":         f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp":        datetime.now().isoformat(),
        "image_info": {
            "filename":   file.filename,
            "dimensions": f"{image.width}x{image.height}",
            "size_kb":    round(len(image_bytes) / 1024, 1)
        },
        "damage_assessment": {
            "damage_detected":   len(detections) > 0,
            "total_detections":  len(detections),
            "overall_severity":  overall_severity,
            "detections":        detections,
            "inference_time_ms": inference_ms
        },
        "fraud_analysis": {
            "fraud_risk":  fraud_risk,
            "flags":       fraud_flags,
            "flagged":     len(fraud_flags) > 0
        },
        "recommendation": (
            "APPROVE — No fraud flags, damage assessed"       if fraud_risk == "Low"    and len(detections) > 0
            else "REVIEW — Fraud flags raised, manual check required" if fraud_risk in ["Medium", "High"]
            else "REVIEW — No damage detected, request clearer photo"
        )
    }

    return JSONResponse(content=response)

# ── Annotated image endpoint ──────────────────────────────────────
@app.post("/inspect/annotated")
async def inspect_damage_annotated(file: UploadFile = File(...)):
    """Returns damage assessment + annotated image with boxes drawn."""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    # Resize large images for faster processing
    max_size = 1280
    if max(image.width, image.height) > max_size:
        ratio = max_size / max(image.width, image.height)
        new_w = int(image.width  * ratio)
        new_h = int(image.height * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Run inference
    start_time   = time.time()
    results      = model.predict(image, conf=0.25, verbose=False)
    inference_ms = round((time.time() - start_time) * 1000, 1)

 # Parse detections
    detections = []
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        confidence = round(float(box.conf), 3)
        coords     = box.xyxy[0].tolist()

        bbox = {
            "x1": round(coords[0]),
            "y1": round(coords[1]),
            "x2": round(coords[2]),
            "y2": round(coords[3])
        }

        location = classify_location(bbox, image.width, image.height)

        detections.append({
            "class":        class_name,
            "severity":     SEVERITY_MAP.get(class_name, "Minor"),
            "confidence":   confidence,
            "location":     location,
            "bounding_box": bbox
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # Fraud checks
    fraud_flags = run_fraud_checks(image, image_bytes, detections)
    fraud_risk  = get_fraud_risk(fraud_flags)

    # Draw boxes on image
    annotated_b64 = draw_damage_boxes(image, detections)

    return JSONResponse(content={
        "claim_id":   f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp":  datetime.now().isoformat(),
        "image_info": {
            "filename":   file.filename,
            "dimensions": f"{image.width}x{image.height}",
            "size_kb":    round(len(image_bytes) / 1024, 1)
        },
        "damage_assessment": {
            "damage_detected":   len(detections) > 0,
            "total_detections":  len(detections),
            "overall_severity":  get_overall_severity(detections),
            "detections":        detections,
            "inference_time_ms": inference_ms
        },
        "fraud_analysis": {
            "fraud_risk": fraud_risk,
            "flags":      fraud_flags,
            "flagged":    len(fraud_flags) > 0
        },
        "recommendation": (
            "APPROVE — No fraud flags, damage assessed"
            if fraud_risk == "Low" and len(detections) > 0
            else "REVIEW — Fraud flags raised, manual check required"
            if fraud_risk in ["Medium", "High"]
            else "REVIEW — No damage detected, request clearer photo"
        ),
        "annotated_image": {
            "format":  "base64/jpeg",
            "data":    annotated_b64,
            "preview": f"data:image/jpeg;base64,{annotated_b64[:50]}..."
        }
    })

# ── Video inspection endpoint ─────────────────────────────────────
@app.post("/inspect/video")
async def inspect_video(file: UploadFile = File(...)):
    """Accepts a video file, extracts frames, runs damage detection on each frame,
    and returns an aggregated damage report."""

    # Validate file type
    allowed = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="File must be a video (MP4, MOV, AVI)")

    # Save video to temp file
    video_bytes = await file.read()
    temp_path = os.path.join(os.getcwd(), f"temp_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    try:
        import cv2
        from collections import Counter

        cap     = cv2.VideoCapture(temp_path)
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 1)

        # Sample one frame every 0.5 seconds
        interval      = max(1, int(fps * 0.5))
        frame_num     = 0
        all_detections = []
        best_frame     = None
        best_conf      = 0
        frames_analysed = 0

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # Resize if needed
                max_size = 640
                if max(pil_frame.width, pil_frame.height) > max_size:
                    ratio = max_size / max(pil_frame.width, pil_frame.height)
                    pil_frame = pil_frame.resize(
                        (int(pil_frame.width * ratio), int(pil_frame.height * ratio)),
                        Image.LANCZOS
                    )

                results = model.predict(pil_frame, conf=0.25, verbose=False)
                frames_analysed += 1

                for box in results[0].boxes:
                    class_name = results[0].names[int(box.cls)]
                    confidence = round(float(box.conf), 3)
                    coords     = box.xyxy[0].tolist()

                    bbox = {
                        "x1": round(coords[0]),
                        "y1": round(coords[1]),
                        "x2": round(coords[2]),
                        "y2": round(coords[3])
                    }

                    location = classify_location(bbox, pil_frame.width, pil_frame.height)

                    all_detections.append({
                        "class":      class_name,
                        "severity":   SEVERITY_MAP.get(class_name, "Minor"),
                        "confidence": confidence,
                        "location":   location,
                        "frame":      frame_num,
                        "bounding_box": bbox
                    })

                    # Track best frame (highest confidence detection)
                    if confidence > best_conf:
                        best_conf  = confidence
                        best_frame = pil_frame.copy()

            frame_num += 1

        cap.release()
        inference_ms = round((time.time() - start_time) * 1000, 1)

        # ── Aggregate results ──────────────────────────────────────
        if not all_detections:
            return JSONResponse(content={
                "claim_id":   f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp":  datetime.now().isoformat(),
                "video_info": {
                    "filename":       file.filename,
                    "duration_sec":   duration_sec,
                    "frames_analysed": frames_analysed,
                    "size_kb":        round(len(video_bytes) / 1024, 1)
                },
                "damage_assessment": {
                    "damage_detected":  False,
                    "total_detections": 0,
                    "overall_severity": "None",
                    "detections":       [],
                    "inference_time_ms": inference_ms
                },
                "fraud_analysis": {
                    "fraud_risk": "Medium",
                    "flags":      ["No damage detected in video — ensure camera focuses on damaged area"],
                    "flagged":    True
                },
                "recommendation": "REVIEW — No damage detected, request clearer video"
            })

        # Count occurrences of each damage type
        class_counts = Counter(d["class"] for d in all_detections)

        # Get unique damage types with their best confidence score
        unique_damages = {}
        for det in all_detections:
            cls = det["class"]
            if cls not in unique_damages or det["confidence"] > unique_damages[cls]["confidence"]:
                unique_damages[cls] = det

        # Sort by confidence
        summary_detections = sorted(
            unique_damages.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )

        # Add occurrence count to each
        for det in summary_detections:
            det["occurrences_in_video"] = class_counts[det["class"]]

        # Overall severity
        overall_severity = get_overall_severity(summary_detections)

        # Fraud checks on best frame
        fraud_flags = []
        if best_frame:
            buffer = io.BytesIO()
            best_frame.save(buffer, format="JPEG")
            frame_bytes = buffer.getvalue()
            fraud_flags = run_fraud_checks(best_frame, frame_bytes, summary_detections)

        fraud_risk = get_fraud_risk(fraud_flags)

        # Annotate best frame with detections
        best_frame_b64 = None
        if best_frame:
            best_frame_b64 = draw_damage_boxes(best_frame, summary_detections)

        response = {
            "claim_id":  f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "filename":        file.filename,
                "duration_sec":    duration_sec,
                "total_frames":    total_frames,
                "frames_analysed": frames_analysed,
                "size_kb":         round(len(video_bytes) / 1024, 1)
            },
            "damage_assessment": {
                "damage_detected":   True,
                "total_detections":  len(summary_detections),
                "overall_severity":  overall_severity,
                "detections":        summary_detections,
                "inference_time_ms": inference_ms
            },
            "fraud_analysis": {
                "fraud_risk": fraud_risk,
                "flags":      fraud_flags,
                "flagged":    len(fraud_flags) > 0
            },
            "recommendation": (
                "APPROVE — No fraud flags, damage assessed"
                if fraud_risk == "Low" and len(summary_detections) > 0
                else "REVIEW — Fraud flags raised, manual check required"
                if fraud_risk in ["Medium", "High"]
                else "REVIEW — No damage detected, request clearer video"
            ),
            "best_frame": {
                "format": "base64/jpeg",
                "data":   best_frame_b64
            } if best_frame_b64 else None
        }

        return JSONResponse(content=response)

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
# ── Run server ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)