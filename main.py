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

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.utils import ImageReader
from fastapi.responses import StreamingResponse
import io as _io

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
async def inspect_damage_annotated(file: UploadFile = File(...), skip_fraud: bool = False):
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
    fraud_flags   = [] if skip_fraud else run_fraud_checks(image, image_bytes, detections)
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

# ── PDF Report Generation ─────────────────────────────────────────

def generate_pdf_report(data: dict) -> bytes:
    """Generate a professional PDF report from inspection data."""
    buf = _io.BytesIO()
    W, H = A4  # 595 x 842

    c = canvas.Canvas(buf, pagesize=A4)

    # ── Colours ──────────────────────────────────────────
    GREEN      = HexColor("#1A1A1A")   # header bg — near black
    GREEN_LIGHT= HexColor("#F5F5F5")   # light bg areas
    RED        = HexColor("#1A1A1A")   # flag headers — black
    RED_LIGHT  = HexColor("#F5F5F5")   # flag bg — light grey
    AMBER      = HexColor("#1A1A1A")   # moderate — black
    AMBER_LIGHT= HexColor("#F5F5F5")   # moderate bg — light grey
    GREY_LIGHT = HexColor("#F9F9F9")
    GREY_MID   = HexColor("#D1D5DB")
    GREY_DARK  = HexColor("#6B7280")
    TEXT_DARK  = HexColor("#111827")
    TEXT_MID   = HexColor("#374151")

    # ── Helper functions ──────────────────────────────────
    def draw_rect(x, y, w, h, fill, stroke=None, radius=4):
        c.setFillColor(fill)
        if stroke:
            c.setStrokeColor(stroke)
            c.setLineWidth(1)
            c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
        else:
            c.setStrokeColor(fill)
            c.roundRect(x, y, w, h, radius, fill=1, stroke=0)

    def text(txt, x, y, font="Helvetica", size=10, color=TEXT_DARK, align="left"):
        c.setFont(font, size)
        c.setFillColor(color)
        if align == "center":
            tw = c.stringWidth(txt, font, size)
            c.drawString(x - tw/2, y, txt)
        elif align == "right":
            tw = c.stringWidth(txt, font, size)
            c.drawString(x - tw, y, txt)
        else:
            c.drawString(x, y, txt)

    # ── HEADER ────────────────────────────────────────────
    draw_rect(0, H-80, W, 80, GREEN)

    

    # Title
    text("HOBBITON INVESTMENTS", 84, H-36, "Helvetica-Bold", 14, white)
    text("Vehicle Damage Inspection Report", 84, H-52, "Helvetica", 10, HexColor("#C3DEC9"))

    # Claim ID top right
    claim_id = data.get("claim_id", "—")
    text(claim_id, W-32, H-36, "Helvetica-Bold", 10, white, "right")
    ts = data.get("timestamp", "")[:19].replace("T", " ")
    text(ts, W-32, H-52, "Helvetica", 9, HexColor("#C3DEC9"), "right")

    y = H - 100

    # ── ANNOTATED IMAGE ───────────────────────────────────
    ann = data.get("annotated_image", {})
    if ann and ann.get("data"):
        try:
            import base64
            img_bytes = base64.b64decode(ann["data"])
            img_buf   = _io.BytesIO(img_bytes)
            img_reader = ImageReader(img_buf)
            img_w, img_h = 531, 220
            draw_rect(32, y - img_h - 4, img_w + 4, img_h + 4, GREY_MID, radius=6)
            c.drawImage(img_reader, 34, y - img_h - 2, width=img_w, height=img_h,
                       preserveAspectRatio=True, mask="auto")
            y -= (img_h + 20)
        except Exception:
            y -= 10
    else:
        y -= 10

    # ── DAMAGE ASSESSMENT SECTION ─────────────────────────
    draw_rect(32, y-28, W-64, 28, GREEN, radius=6)
    text("DAMAGE ASSESSMENT", 44, y-18, "Helvetica-Bold", 11, white)

    # Overall severity badge
    dam   = data.get("damage_assessment", {})
    sev   = dam.get("overall_severity", "None")
    sev_color = HexColor("#1A1A1A")
    sev_bg    = HexColor("#F5F5F5")
    sev_w = c.stringWidth(f"  {sev}  ", "Helvetica-Bold", 10) + 8
    draw_rect(W-32-sev_w, y-24, sev_w, 20, sev_bg, sev_color, radius=4)
    text(sev.upper(), W-32-sev_w/2, y-16, "Helvetica-Bold", 9, sev_color, "center")

    y -= 40

    # Summary row
    total = dam.get("total_detections", 0)
    inf   = dam.get("inference_time_ms", 0)
    img_info = data.get("image_info", {})

    for i, (label, val) in enumerate([
        ("Total Detections", str(total)),
        ("Inference Time", f"{inf}ms"),
        ("Image Size", img_info.get("dimensions","—")),
        ("File", img_info.get("filename","—")[:20]),
    ]):
        bx = 32 + i * 132
        draw_rect(bx, y-36, 126, 36, GREY_LIGHT, GREY_MID, radius=4)
        text(label, bx+8, y-14, "Helvetica", 7, GREY_DARK)
        text(val, bx+8, y-28, "Helvetica-Bold", 9, TEXT_DARK)

    y -= 52

    # Detections table
    detections = dam.get("detections", [])
    if detections:
        # Table header
        headers = ["#", "Damage Type", "Location", "Severity", "Confidence"]
        widths  = [24, 150, 160, 90, 90]
        cols    = [32, 56, 206, 366, 456]

        draw_rect(32, y-22, W-64, 22, HexColor("#F3F4F6"), radius=0)
        c.setStrokeColor(GREY_MID)
        c.setLineWidth(0.5)
        c.rect(32, y-22, W-64, 22, fill=0, stroke=1)

        for i, h in enumerate(headers):
            text(h, cols[i]+4, y-14, "Helvetica-Bold", 8, GREY_DARK)
        y -= 22

        # Table rows
        for idx, det in enumerate(detections):
            row_bg = white if idx % 2 == 0 else GREY_LIGHT
            draw_rect(32, y-22, W-64, 22, row_bg, radius=0)
            c.setStrokeColor(GREY_MID)
            c.setLineWidth(0.3)
            c.rect(32, y-22, W-64, 22, fill=0, stroke=1)

            det_sev   = det.get("severity","")
            det_color = TEXT_DARK

            text(str(idx+1),            cols[0]+4, y-14, "Helvetica", 8, GREY_DARK)
            text(det.get("class",""),   cols[1]+4, y-14, "Helvetica-Bold", 8, TEXT_DARK)
            text(det.get("location",""),cols[2]+4, y-14, "Helvetica", 8, TEXT_MID)
            text(det_sev,               cols[3]+4, y-14, "Helvetica-Bold", 8, det_color)
            text(f"{int(det.get('confidence',0)*100)}%", cols[4]+4, y-14, "Helvetica", 8, TEXT_MID)

            y -= 22
            if y < 180:
                c.showPage()
                y = H - 60
    else:
        draw_rect(32, y-36, W-64, 36, GREY_LIGHT, GREY_MID, radius=4)
        text("No damage detected in submitted image", W/2, y-20, "Helvetica", 10, GREY_DARK, "center")
        y -= 52

    y -= 16

    # ── FRAUD ANALYSIS SECTION ────────────────────────────
    fraud     = data.get("fraud_analysis", {})
    risk      = fraud.get("fraud_risk", "Low")
    risk_color= RED if risk=="High" else AMBER if risk=="Medium" else GREEN
    risk_bg   = RED_LIGHT if risk=="High" else AMBER_LIGHT if risk=="Medium" else GREEN_LIGHT

    draw_rect(32, y-28, W-64, 28, risk_color, radius=6)
    text("FRAUD ANALYSIS", 44, y-18, "Helvetica-Bold", 11, white)
    risk_label = f"{risk.upper()} RISK"
    rw = c.stringWidth(f"  {risk_label}  ", "Helvetica-Bold", 9) + 8
    draw_rect(W-32-rw, y-24, rw, 20, risk_bg, risk_color, radius=4)
    text(risk_label, W-32-rw/2, y-16, "Helvetica-Bold", 9, risk_color, "center")

    y -= 40

    flags = fraud.get("flags", [])
    if flags:
        for flag in flags:
            draw_rect(32, y-28, W-64, 28, RED_LIGHT, RED, radius=4)
            text("⚠", 44, y-16, "Helvetica", 10, RED)
            flag_text = flag if isinstance(flag, str) else flag.get("message", str(flag))
            if len(flag_text) > 85:
                flag_text = flag_text[:82] + "..."
            text(flag_text, 60, y-16, "Helvetica", 8, RED)
            y -= 34
    else:
        draw_rect(32, y-28, W-64, 28, GREEN_LIGHT, GREEN, radius=4)
        text("✓  No fraud indicators detected on this submission", 44, y-16, "Helvetica", 9, GREEN)
        y -= 34

    y -= 16

    # ── RECOMMENDATION ────────────────────────────────────
    rec       = data.get("recommendation", "")
    is_approve= rec.startswith("APPROVE")
    rec_color = HexColor("#1A1A1A")
    rec_bg    = HexColor("#F5F5F5")
    rec_icon  = "✓" if is_approve else "!"
    rec_title = "APPROVED FOR PROCESSING" if is_approve else "FLAGGED FOR MANUAL REVIEW"

    draw_rect(32, y-52, W-64, 52, rec_bg, rec_color, radius=6)
    c.setFillColor(rec_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(44, y-26, rec_icon)
    text(rec_title, 68, y-22, "Helvetica-Bold", 12, rec_color)
    text(rec, 68, y-38, "Helvetica", 8, rec_color)

    y -= 68

    # ── FOOTER ────────────────────────────────────────────
    draw_rect(0, 0, W, 40, HexColor("#1A1A1A"))
    text("Generated by Hobbiton Investments AI Inspection System · YOLOv8m · v1.0",
         W/2, 24, "Helvetica", 8, HexColor("#AAAAAA"), "center")
    text("Final decisions rest with licensed claim handlers.",
         W/2, 12, "Helvetica", 7, HexColor("#888888"), "center")

    c.save()
    buf.seek(0)
    return buf.read()


@app.post("/inspect/report")
async def inspect_and_report(file: UploadFile = File(...), skip_fraud: bool = False):
    """Runs damage inspection and returns a downloadable PDF report."""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    max_size = 1280
    if max(image.width, image.height) > max_size:
        ratio = max_size / max(image.width, image.height)
        image = image.resize(
            (int(image.width*ratio), int(image.height*ratio)), Image.LANCZOS
        )

    start_time   = time.time()
    results      = model.predict(image, conf=0.25, verbose=False)
    inference_ms = round((time.time()-start_time)*1000, 1)
    detections   = parse_detections(results, image)

    fraud_flags  = [] if skip_fraud else run_fraud_checks(image, image_bytes, detections)
    fraud_risk   = get_fraud_risk(fraud_flags)
    ann_b64      = draw_damage_boxes(image, detections)

    report_data = {
        "claim_id":  f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "filename":   file.filename,
            "dimensions": f"{image.width}x{image.height}",
            "size_kb":    round(len(image_bytes)/1024, 1)
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
        "annotated_image": {"data": ann_b64}
    }

    pdf_bytes = generate_pdf_report(report_data)

    claim_id = report_data["claim_id"]
    return StreamingResponse(
        _io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={claim_id}_report.pdf"}
    )
