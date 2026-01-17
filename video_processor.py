"""
Video processing module that can be used by both test scripts and web application.
This module provides a unified function to process videos with the YOLO model.
"""
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model (will be loaded once)
_model = None
_model_loaded = False

# Constants (same as main_v4.py)
MIN_CONFIDENCE_THRESHOLD = 0.4
INFERENCE_SIZE = 1280
ENABLE_TILING = False

def load_model():
    """Load the YOLO model"""
    global _model, _model_loaded
    if _model_loaded and _model is not None:
        return _model
    
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    print(f"Loading model from: {os.path.abspath(model_path)}")
    _model = YOLO(model_path)
    _model_loaded = True
    print("Model loaded successfully")
    return _model

def detect_brightness(img):
    """Detect average brightness of a frame"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def preprocess_image(img, clip_limit=3.0, gamma=1.0, brightness_boost=0):
    """Image preprocessing (same as main_v4.py)"""
    if brightness_boost > 0:
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness_boost)
    
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)

    if clip_limit > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    
    return img

def run_tiled_inference(model, frame, conf=0.25):
    """Run inference on frame (same as main_v4.py)"""
    img_h, img_w = frame.shape[:2]
    all_boxes = []
    
    results = model(frame, conf=conf, iou=0.5, imgsz=INFERENCE_SIZE, agnostic_nms=True, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bconf = float(box.conf[0].cpu().numpy())
        bcls = int(box.cls[0].cpu().numpy())
        all_boxes.append([x1, y1, x2, y2, bconf, bcls])
    
    if not ENABLE_TILING:
        return all_boxes
    
    # Tiling code (same as main_v4.py)
    overlap = 100
    tile_h = img_h // 2 + overlap
    tile_w = img_w // 2 + overlap
    
    tiles = [
        (0, 0, tile_w, tile_h),
        (img_w - tile_w, 0, img_w, tile_h),
        (0, img_h - tile_h, tile_w, img_h),
        (img_w - tile_w, img_h - tile_h, img_w, img_h),
    ]
    
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = frame[ty1:ty2, tx1:tx2]
        if tile.size == 0:
            continue
        
        tile_results = model(tile, conf=conf, iou=0.5, imgsz=INFERENCE_SIZE, agnostic_nms=True, verbose=False)
        
        for box in tile_results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            bconf = float(box.conf[0].cpu().numpy())
            bcls = int(box.cls[0].cpu().numpy())
            
            abs_x1 = bx1 + tx1
            abs_y1 = by1 + ty1
            abs_x2 = bx2 + tx1
            abs_y2 = by2 + ty1
            
            all_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2, bconf, bcls])
    
    if not all_boxes:
        return []
    
    nms_boxes = []
    nms_scores = []
    nms_classes = []
    
    for b in all_boxes:
        x1, y1, x2, y2, s, c = b
        nms_boxes.append([x1, y1, x2-x1, y2-y1])
        nms_scores.append(float(s))
        nms_classes.append(int(c))
    
    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=conf, nms_threshold=0.5)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = nms_boxes[i]
            final_boxes.append([x, y, x+w, y+h, nms_scores[i], nms_classes[i]])
    
    return final_boxes

def double_check_and_annotate(model, frame, all_boxes):
    """Double-check detections and annotate frame (same as main_v4.py)"""
    img_h, img_w = frame.shape[:2]
    final_detections = []

    for box_data in all_boxes:
        x1, y1, x2, y2, conf, cls_id = box_data
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        if conf < MIN_CONFIDENCE_THRESHOLD:
            continue
        
        new_cls_id = cls_id
        
        box_center_y = (y1 + y2) / 2
        is_upper_half = box_center_y < (img_h * 0.65)
        
        if cls_id == 1 and is_upper_half:
            pad = 30
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(img_w, int(x2) + pad)
            cy2 = min(img_h, int(y2) + pad)
            
            crop = frame[cy1:cy2, cx1:cx2]
            
            if crop.size > 0:
                processed_crop = preprocess_image(crop, clip_limit=4.0)
                crop_results = model(processed_crop, conf=0.30, iou=0.5, agnostic_nms=True, verbose=False)
                
                for c_box in crop_results[0].boxes:
                    c_cls = int(c_box.cls[0].cpu().numpy())
                    c_conf = float(c_box.conf[0].cpu().numpy())
                    if c_cls == 0 and c_conf >= MIN_CONFIDENCE_THRESHOLD:
                        new_cls_id = 0
                        print(f"Correction made: no-threat -> threat (Conf: {c_conf:.2f})")
                        break
        
        final_detections.append([int(x1), int(y1), int(x2), int(y2), conf, new_cls_id])

    annotated_frame = frame.copy()
    for x1, y1, x2, y2, conf, cls_id in final_detections:
        color = (0, 0, 255) if cls_id == 0 else (0, 255, 0) 
        label = "threat" if cls_id == 0 else "no-threat"
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    return annotated_frame, final_detections

def process_video(video_path, output_path=None, show_preview=False):
    """
    Process video with YOLO model (same logic as main_v4.py test_video function).
    
    Args:
        video_path: Path to input video file
        output_path: Path to save processed video (if None, only shows preview)
        show_preview: If True, shows video in window (like test_video)
    
    Returns:
        dict with processing results:
        {
            'status': 'success' or 'error',
            'total_frames': int,
            'detections': list of detections,
            'output_path': str (if output_path provided),
            'error': str (if error occurred)
        }
    """
    # Load model
    model = load_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"Could not open video file: {video_path}"
        print(f"Error: {error_msg}")
        return {'status': 'error', 'error': error_msg}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video test started: {video_path}")
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            error_msg = f"Could not create output video: {output_path}"
            print(f"Error: {error_msg}")
            cap.release()
            return {'status': 'error', 'error': error_msg}
    
    # Settings (same as main_v4.py)
    settings_normal = [
        (0.25, 3.0, 1.0, 0, "Normal"),
        (0.25, 6.0, 1.0, 0, "High Contrast"),
        (0.20, 4.0, 0.9, 0, "Mixed"),
    ]
    
    settings_dark = [
        (0.20, 8.0, 0.5, 50, "Dark - Brighten"),
        (0.20, 10.0, 0.4, 80, "Dark - Max"),
        (0.15, 6.0, 0.6, 30, "Dark - Sensitive"),
    ]
    
    frame_count = 0
    all_detections = []
    
    # Process video frame by frame (EXACTLY same as main_v4.py test_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Adaptive preprocessing (same as main_v4.py lines 249-256)
        brightness = detect_brightness(frame)
        is_dark = brightness < 80
        
        settings = settings_dark if is_dark else settings_normal
        conf, clip, gamma, bright, desc = settings[frame_count % len(settings)]
        frame_count += 1
        
        processed_frame = preprocess_image(frame, clip_limit=clip, gamma=gamma, brightness_boost=bright)
        
        # Tiled inference (same as main_v4.py line 259)
        all_boxes = run_tiled_inference(model, processed_frame, conf=conf)
        annotated_frame, final_detections = double_check_and_annotate(model, processed_frame, all_boxes)
        
        # Add status text overlay (same as main_v4.py lines 262-265)
        dark_status = "DARK" if is_dark else "NORMAL"
        tiling_status = "TILED" if ENABLE_TILING else "NORMAL"
        cv2.putText(annotated_frame, f"{desc} | {dark_status} | {tiling_status} | imgsz={INFERENCE_SIZE}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Save frame if output path provided
        if out is not None:
            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))
            out.write(annotated_frame)
        
        # Show preview if requested
        if show_preview:
            cv2.imshow('YOLO Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Collect detections with timestamp
        timestamp = frame_count / fps
        for det in final_detections:
            x1, y1, x2, y2, conf, cls_id = det
            all_detections.append({
                'class': 'threat' if cls_id == 0 else 'no-threat',
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'timestamp': float(timestamp),
                'frame': frame_count
            })
        
        # Progress logging
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    if out is not None:
        out.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    result = {
        'status': 'success',
        'total_frames': frame_count,
        'detections': all_detections
    }
    
    if output_path:
        result['output_path'] = output_path
        print(f"Processed video saved to: {output_path}")
    
    print(f"Video processing completed! Total frames: {frame_count}, Detections: {len(all_detections)}")
    return result

