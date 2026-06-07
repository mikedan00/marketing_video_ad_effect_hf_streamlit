import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def save_uploaded_file(uploaded_file, suffix: str = ".mp4") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def extract_keyframes(video_path: str, n_frames: int = 8, max_size=(1024, 576), jpeg_quality: int = 82) -> Tuple[List[Dict], Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. 파일 형식 또는 코덱을 확인하세요.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    if total <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 읽을 수 없습니다.")

    n = max(1, min(int(n_frames), total))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail(max_size)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        frames.append({
            "frame_index": int(idx),
            "timestamp_sec": round(idx / fps, 2),
            "b64": b64,
        })

    cap.release()

    return frames, {
        "total_frames": total,
        "fps": round(fps, 2),
        "duration_sec": round(total / fps, 2),
        "extracted_frames": len(frames),
    }


def compute_video_stats(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_count = min(30, max(total, 1))
    indices = np.linspace(0, max(total - 1, 0), sample_count, dtype=int)

    brightness, saturation, motion = [], [], []
    prev_gray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        brightness.append(float(np.mean(gray)))
        saturation.append(float(np.mean(hsv[:, :, 1])))

        if prev_gray is not None:
            motion.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
        prev_gray = gray

    cap.release()

    ratio = round(w / h, 3) if h else 0
    fmt = "vertical(9:16)" if ratio < 0.7 else ("square(1:1)" if ratio < 1.1 else "horizontal(16:9)")

    return {
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "aspect_ratio": ratio,
        "format_tag": fmt,
        "duration_sec": round(total / fps, 2) if fps else 0,
        "fps": round(fps, 2),
        "avg_brightness": round(float(np.mean(brightness)), 1) if brightness else 0,
        "avg_saturation": round(float(np.mean(saturation)), 1) if saturation else 0,
        "avg_motion_score": round(float(np.mean(motion)), 2) if motion else 0,
    }
