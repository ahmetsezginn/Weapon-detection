from flask import Flask, render_template, request, jsonify, send_from_directory, Response, url_for
import os
import io
import subprocess
import numpy as np
from PIL import Image
import cv2
import threading
import time

def convert_video_for_browser(input_path, output_path):
    """
    Convert video to browser-compatible format using FFmpeg.
    Uses H.264 codec which is supported by all modern browsers.
    Uses imageio-ffmpeg package which includes FFmpeg binary.
    Returns True if successful, False otherwise.
    """
    try:
        # Get FFmpeg path from imageio-ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"Using FFmpeg from imageio-ffmpeg: {ffmpeg_path}")
        except ImportError:
            print("WARNING: imageio-ffmpeg not installed. Trying system FFmpeg...")
            ffmpeg_path = 'ffmpeg'
        except Exception as e:
            print(f"WARNING: Could not get FFmpeg from imageio-ffmpeg: {e}")
            ffmpeg_path = 'ffmpeg'
        
        # FFmpeg command to convert to browser-compatible H.264/AAC format
        cmd = [
            ffmpeg_path,
            '-y',  # Overwrite output file
            '-i', input_path,  # Input file
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'fast',  # Encoding speed/quality tradeoff
            '-crf', '23',  # Quality (lower = better, 18-28 is good range)
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '128k',  # Audio bitrate
            '-movflags', '+faststart',  # Enable fast start for web playback
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            output_path
        ]
        
        print(f"Converting video for browser compatibility...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            # If conversion fails, try to copy original
            import shutil
            shutil.copy(input_path, output_path)
            return True
        
        print(f"Video converted successfully!")
        return True
        
    except Exception as e:
        print(f"Error converting video: {e}")
        import traceback
        traceback.print_exc()
        # If anything fails, try to copy the original
        try:
            import shutil
            shutil.copy(input_path, output_path)
            return True
        except:
            return False

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
model_loaded = False

# Constants for real-time detection
TEMPORAL_SMOOTHING_FRAMES = 5
MIN_DETECTION_FRAMES = 2
REALTIME_CONFIDENCE_THRESHOLD = 0.25

# Advanced real-time detection settings (compatible with main.py)
MIN_CONFIDENCE_THRESHOLD = 0.4          # Detections below this value will not be reported
INFERENCE_SIZE = 1280                   # High resolution for small objects
ENABLE_TILING = False                   # Tiling is optional

# Global state for temporal smoothing (real-time detection)
realtime_detection_history = {}  # Track detections across frames
frame_counter = 0

# Global state for video streaming
video_camera = None
video_streaming = False
video_lock = threading.Lock()

def load_model():
    """Load the YOLO model"""
    global model, model_loaded
    if model_loaded and model is not None:
        return model
    
    try:
        from ultralytics import YOLO
        model_path = 'best.pt'
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model file '{model_path}' not found!")
            return None
        
        print(f"Loading model from: {os.path.abspath(model_path)}")
        model = YOLO(model_path)
        
        # Test the model
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_result = model.predict(test_img, verbose=False)
        
        print("Model loaded successfully")
        if hasattr(model, 'names'):
            print(f"Model classes: {model.names}")
        
        model_loaded = True
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_brightness(frame):
    """Detect average brightness of a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def preprocess_image(img, clip_limit=3.0, gamma=1.0, brightness_boost=0):
    """
    Image enhancement operations (from main_v4.py).
    """
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

def double_check_and_annotate(model, frame, all_boxes):
    """
    Draw results and double check no-threat detections (from main_v4.py).
    Returns: (annotated_frame, final_detections)
    """
    img_h, img_w = frame.shape[:2]
    
    final_detections = []

    for box_data in all_boxes:
        try:
            if len(box_data) < 6:
                print(f"Invalid box_data format: {box_data}")
                continue
            x1, y1, x2, y2, conf, cls_id = box_data[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
        except Exception as e:
            print(f"Error unpacking box_data: {e}, box_data: {box_data}")
            continue
        
        # Minimum confidence threshold check
        if conf < MIN_CONFIDENCE_THRESHOLD:
            continue
        
        new_cls_id = cls_id
        
        # If 'no-threat' (1) and in UPPER region, re-check
        # Don't double check detections in lower region (feet)
        box_center_y = (y1 + y2) / 2
        is_upper_half = box_center_y < (img_h * 0.65)  # Upper 65%
        
        if cls_id == 1 and is_upper_half:
            pad = 30
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(img_w, int(x2) + pad)
            cy2 = min(img_h, int(y2) + pad)
            
            crop = frame[cy1:cy2, cx1:cx2]
            
            if crop.size > 0:
                try:
                    processed_crop = preprocess_image(crop, clip_limit=4.0)
                    crop_results = model(processed_crop, conf=0.30, iou=0.5, agnostic_nms=True, verbose=False)
                    
                    if crop_results and len(crop_results) > 0 and hasattr(crop_results[0], 'boxes') and len(crop_results[0].boxes) > 0:
                        for c_box in crop_results[0].boxes:
                            try:
                                c_cls = int(c_box.cls[0].cpu().numpy())
                                c_conf = float(c_box.conf[0].cpu().numpy())
                                if c_cls == 0 and c_conf >= MIN_CONFIDENCE_THRESHOLD:
                                    new_cls_id = 0
                                    print(f"Correction made: no-threat -> threat (Conf: {c_conf:.2f})")
                                    break
                            except Exception as box_error:
                                print(f"Error processing crop box: {box_error}")
                                continue
                except Exception as e:
                    print(f"Error in double-check crop processing: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with original classification if crop processing fails
        
        try:
            final_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(new_cls_id)])
        except Exception as e:
            print(f"Error appending detection: {e}")
            continue

    # Draw annotations - only show threat (cls_id == 0) detections
    annotated_frame = frame.copy()
    try:
        for det in final_detections:
            try:
                if len(det) < 6:
                    print(f"Invalid detection format in drawing: {det}")
                    continue
                x1, y1, x2, y2, conf, cls_id = det[:6]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)
                
                # Only draw threat detections on screen
                if cls_id != 0:
                    continue
                
                color = (0, 0, 255)  # Red - threat
                label = "threat"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as draw_error:
                print(f"Error drawing detection: {draw_error}, det: {det}")
                continue
    except Exception as e:
        print(f"Error in drawing annotations: {e}")
        import traceback
        traceback.print_exc()
                    
    return annotated_frame, final_detections

def run_tiled_inference(model, frame, conf=0.25, fast_mode=False, ultra_fast=False):
    """
    Advanced real-time inference:
    - Full frame scan with high resolution (INFERENCE_SIZE)
    - Optional tiling to capture small objects
    - Box merging with OpenCV NMS
    Return format: [x1, y1, x2, y2, conf, cls_id]
    """
    try:
        img_h, img_w = frame.shape[:2]
        all_boxes = []  # [x1, y1, x2, y2, conf, cls]

        # 1) Full frame scan (high resolution)
        results = model(
            frame,
            conf=conf,
            iou=0.5,
            imgsz=INFERENCE_SIZE,
            agnostic_nms=True,
            verbose=False
        )
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bconf = float(box.conf[0].cpu().numpy())
                    bcls = int(box.cls[0].cpu().numpy())
                    all_boxes.append([x1, y1, x2, y2, bconf, bcls])
                except Exception as e:
                    print(f"Error processing box: {e}")
                    continue

        # If tiling is disabled, return results directly
        if not ENABLE_TILING:
            return all_boxes

        # 2) Tiles (2x2 grid + overlap)
        overlap = 100  # pixels
        tile_h = img_h // 2 + overlap
        tile_w = img_w // 2 + overlap

        tiles = [
            (0, 0, tile_w, tile_h),                          # Top left
            (img_w - tile_w, 0, img_w, tile_h),              # Top right
            (0, img_h - tile_h, tile_w, img_h),              # Bottom left
            (img_w - tile_w, img_h - tile_h, img_w, img_h),  # Bottom right
        ]

        for (tx1, ty1, tx2, ty2) in tiles:
            tile = frame[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            tile_results = model(
                tile,
                conf=conf,
                iou=0.5,
                imgsz=INFERENCE_SIZE,
                agnostic_nms=True,
                verbose=False
            )

            if tile_results and len(tile_results) > 0 and hasattr(tile_results[0], 'boxes') and len(tile_results[0].boxes) > 0:
                for box in tile_results[0].boxes:
                    try:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        bconf = float(box.conf[0].cpu().numpy())
                        bcls = int(box.cls[0].cpu().numpy())

                        # Adjust coordinates relative to the main image
                        abs_x1 = bx1 + tx1
                        abs_y1 = by1 + ty1
                        abs_x2 = bx2 + tx1
                        abs_y2 = by2 + ty1

                        all_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2, bconf, bcls])
                    except Exception as e:
                        print(f"Error processing tile box: {e}")
                        continue

        # If no boxes
        if not all_boxes:
            return []

        # NMS - Remove overlapping boxes
        nms_boxes = []
        nms_scores = []
        for b in all_boxes:
            x1, y1, x2, y2, s, _ = b
            nms_boxes.append([x1, y1, x2 - x1, y2 - y1])
            nms_scores.append(float(s))

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=conf, nms_threshold=0.5)

        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = nms_boxes[i]
                x1, y1, x2, y2, s, c = all_boxes[i]
                final_boxes.append([x, y, x + w, y + h, s, c])

        return final_boxes
    except Exception as e:
        print(f"Error in inference: {e}")
        return []


def double_check_detections(model, frame, detections, enable_verification=True):
    """
    According to the real-time logic in main.py:
    - Eliminates low confidence scores
    - Re-checks 'no-threat' (1) boxes in the upper region,
      corrects to 'threat' (0) if necessary.
    Return format: [x1, y1, x2, y2, conf, cls_id]
    """
    if not detections:
        return []

    img_h, img_w = frame.shape[:2]
    final_detections = []

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)

        # Minimum confidence threshold
        if conf < MIN_CONFIDENCE_THRESHOLD:
            continue

        new_cls_id = cls_id

        if enable_verification:
            # If 'no-threat' (1) and in upper region, re-check
            box_center_y = (y1 + y2) / 2
            is_upper_half = box_center_y < (img_h * 0.65)  # upper 65%

            if cls_id == 1 and is_upper_half:
                pad = 30
                cx1 = max(0, int(x1) - pad)
                cy1 = max(0, int(y1) - pad)
                cx2 = min(img_w, int(x2) + pad)
                cy2 = min(img_h, int(y2) + pad)

                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size > 0:
                    # Enhance cropped area
                    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    processed_crop = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                    crop_results = model(
                        processed_crop,
                        conf=0.30,
                        iou=0.5,
                        agnostic_nms=True,
                        verbose=False
                    )

            if crop_results and len(crop_results) > 0 and hasattr(crop_results[0], 'boxes') and len(crop_results[0].boxes) > 0:
                for c_box in crop_results[0].boxes:
                    try:
                        c_cls = int(c_box.cls[0].cpu().numpy())
                        c_conf = float(c_box.conf[0].cpu().numpy())
                        if c_cls == 0 and c_conf >= MIN_CONFIDENCE_THRESHOLD:
                            new_cls_id = 0
                            print(f"Correction made: no-threat -> threat (Conf: {c_conf:.2f})")
                            break
                    except Exception as e:
                        print(f"Error processing crop box: {e}")
                        continue

        final_detections.append([x1, y1, x2, y2, conf, new_cls_id])

    return final_detections

def update_temporal_smoothing(detections, frame_id):
    """
    Temporal smoothing algorithm: track detections across frames to reduce false positives
    and stabilize results. Only reports detections that appear consistently.
    """
    global realtime_detection_history, frame_counter
    frame_counter += 1
    
    # Clean old detections (older than TEMPORAL_SMOOTHING_FRAMES)
    if frame_counter > TEMPORAL_SMOOTHING_FRAMES:
        old_frame_id = frame_counter - TEMPORAL_SMOOTHING_FRAMES
        realtime_detection_history = {k: v for k, v in realtime_detection_history.items() 
                                     if v['last_seen'] >= old_frame_id}
    
    smoothed_detections = []
    
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        conf = det[4]
        cls_id = det[5]
        
        # Create a unique key based on position and class (simple tracking)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Quantize position to handle small movements
        quantized_x = int(center_x / 20) * 20
        quantized_y = int(center_y / 20) * 20
        detection_key = f"{cls_id}_{quantized_x}_{quantized_y}"
        
        if detection_key in realtime_detection_history:
            # Update existing detection
            history = realtime_detection_history[detection_key]
            history['count'] += 1
            history['last_seen'] = frame_counter
            # Exponential moving average for confidence
            history['confidence'] = 0.7 * history['confidence'] + 0.3 * conf
            history['bbox'] = det[:4]  # Update bbox
            
            # Only include if seen in minimum frames
            if history['count'] >= MIN_DETECTION_FRAMES:
                smoothed_detections.append([
                    history['bbox'][0], history['bbox'][1], 
                    history['bbox'][2], history['bbox'][3],
                    history['confidence'], cls_id
                ])
        else:
            # New detection - add to history
            realtime_detection_history[detection_key] = {
                'count': 1,
                'last_seen': frame_counter,
                'confidence': conf,
                'bbox': det[:4]
            }
    
    return smoothed_detections

# Flask Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/overview')
def overview():
    """Project overview page"""
    return render_template('overview.html')

@app.route('/architecture')
def architecture():
    """System architecture page"""
    return render_template('architecture.html')

@app.route('/demo')
def demo():
    """Live demo page"""
    return render_template('demo.html')

@app.route('/team')
def team():
    """Team information page"""
    return render_template('team.html')

@app.route('/contact')
def contact():
    """Contact and references page"""
    return render_template('contact.html')

def generate_frames():
    """
    Generator function for video streaming.
    Uses the same logic as main_v4.py test_webcam() function.
    Captures frames from webcam, processes them through YOLO model,
    draws bounding boxes, and yields JPEG frames.
    """
    global video_camera, video_streaming, model, frame_counter
    
    # Initialize camera if not already done (use CAP_DSHOW for Windows)
    with video_lock:
        if video_camera is None:
            try:
                video_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except:
                video_camera = cv2.VideoCapture(0)
            
            if not video_camera.isOpened():
                print("Error: Could not open webcam")
                return
        
        video_streaming = True
    
    # Load model if not loaded
    if model is None:
        model = load_model()
        if model is None:
            print("Error: Model not loaded")
            return
    
    # Preprocessing settings (same as main_v4.py)
    settings = [
        (0.25, 3.0, 1.0, 0, "Normal"),
        (0.25, 6.0, 1.0, 0, "High Contrast"),
        (0.20, 4.0, 0.8, 0, "Mixed"),
        (0.20, 8.0, 0.5, 30, "Dark - Brighten"),
        (0.20, 10.0, 0.4, 50, "Dark - Max"),
        (0.15, 6.0, 0.6, 20, "Dark - Sensitive"),
    ]
    
    frame_counter = 0
    
    try:
        while video_streaming:
            ret, frame = video_camera.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # Select preprocessing settings (cycle through settings)
            conf, clip, gamma, bright, desc = settings[frame_counter % len(settings)]
            
            # Preprocess frame using main_v4.py logic
            processed_frame = preprocess_image(frame, clip_limit=clip, gamma=gamma, brightness_boost=bright)
            
            # Run tiled inference (same as main_v4.py)
            all_boxes = run_tiled_inference(model, processed_frame, conf=conf)
            
            # Apply double-check and annotate (same as main_v4.py)
            annotated_frame, final_detections = double_check_and_annotate(model, processed_frame, all_boxes)
            
            # Add status text (optional, can be removed)
            cv2.putText(annotated_frame, f"Mode: {desc} | imgsz={INFERENCE_SIZE}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control FPS
            time.sleep(0.05)  # ~20 FPS max
            
    except Exception as e:
        print(f"Error in video streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with video_lock:
            video_streaming = False

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route.
    Returns a multipart response with JPEG frames.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_video_stream', methods=['POST'])
def start_video_stream():
    """Start video streaming"""
    global video_streaming, video_camera
    
    with video_lock:
        if video_camera is None:
            try:
                video_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except:
                video_camera = cv2.VideoCapture(0)
            
            if not video_camera.isOpened():
                return jsonify({'error': 'Could not open webcam'}), 500
        
        video_streaming = True
    
    return jsonify({'status': 'started', 'message': 'Video stream started'})

@app.route('/stop_video_stream', methods=['POST'])
def stop_video_stream():
    """Stop video streaming"""
    global video_streaming, video_camera
    
    with video_lock:
        video_streaming = False
        if video_camera is not None:
            video_camera.release()
            video_camera = None
    
    return jsonify({'status': 'stopped', 'message': 'Video stream stopped'})

# API endpoint for real-time camera detection (optimized)
@app.route('/detect_realtime', methods=['POST'])
def detect_realtime():
    """
    Optimized endpoint specifically for real-time camera detection.
    Uses ultra-fast inference, minimal preprocessing, and temporal smoothing.
    """
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format
        img_array = np.array(image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Ultra-fast preprocessing: only resize if needed, skip expensive operations
        # Resize to smaller size for faster inference
        original_h, original_w = frame.shape[:2]
        max_dimension = 640  # Maximum dimension for real-time
        if max(original_h, original_w) > max_dimension:
            scale = max_dimension / max(original_h, original_w)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Minimal preprocessing - only brightness boost if very dark
        brightness = detect_brightness(frame)
        if brightness < 60:  # Very dark
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
        
        # Run ultra-fast inference
        all_boxes = run_tiled_inference(
            model, frame, 
            conf=REALTIME_CONFIDENCE_THRESHOLD, 
            fast_mode=True, 
            ultra_fast=True
        )
        
        # Fast classification (skip verification)
        final_detections = double_check_detections(
            model, frame, all_boxes, 
            enable_verification=False
        )
        
        # Apply temporal smoothing
        global frame_counter
        smoothed_detections = update_temporal_smoothing(final_detections, frame_counter)
        
        # Convert to JSON format
        detections = []
        for det in smoothed_detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            # Only return threat detections
            if cls_id != 0:
                continue
            
            # Scale bbox back to original size if resized
            if max(original_h, original_w) > max_dimension:
                scale = max(original_h, original_w) / max_dimension
                x1 = int(x1 * scale)
                y1 = int(y1 * scale)
                x2 = int(x2 * scale)
                y2 = int(y2 * scale)
            
            detection = {
                'class': 'threat',
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }
            detections.append(detection)
        
        return jsonify({
            'detections': detections,
            'status': 'success',
            'frame_id': frame_counter
        })
    
    except Exception as e:
        print(f"Error during real-time detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# API endpoint for still image detection
@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint for still image detection.
    Uses the same logic as main_v4.py with adaptive preprocessing.
    """
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format
        img_array = np.array(image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Store original dimensions
        original_h, original_w = frame.shape[:2]
        
        # Adaptive preprocessing based on brightness (same as main_v4.py)
        brightness = detect_brightness(frame)
        
        # Select preprocessing settings based on brightness
        if brightness < 80:  # Dark image
            settings_dark = [
                (0.20, 8.0, 0.5, 50, "Dark - Brighten"),
                (0.20, 10.0, 0.4, 80, "Dark - Max"),
                (0.15, 6.0, 0.6, 30, "Dark - Sensitive"),
            ]
            conf, clip, gamma, bright, desc = settings_dark[0]  # Use first dark setting
        else:  # Normal/bright image
            settings_normal = [
                (0.25, 3.0, 1.0, 0, "Normal"),
                (0.25, 6.0, 1.0, 0, "High Contrast"),
                (0.20, 4.0, 0.9, 0, "Mixed"),
            ]
            conf, clip, gamma, bright, desc = settings_normal[0]  # Use first normal setting
        
        # Preprocess frame using main_v4.py logic
        try:
            processed_frame = preprocess_image(frame, clip_limit=clip, gamma=gamma, brightness_boost=bright)
        except Exception as e:
            print(f"Error in preprocess_image: {e}")
            processed_frame = frame  # Use original frame if preprocessing fails
        
        # Run tiled inference (same as main_v4.py)
        try:
            all_boxes = run_tiled_inference(model, processed_frame, conf=conf)
        except Exception as e:
            print(f"Error in run_tiled_inference: {e}")
            import traceback
            traceback.print_exc()
            all_boxes = []
        
        if all_boxes is None:
            all_boxes = []
        
        # Apply double-check and annotate (same as main_v4.py)
        # Note: double_check_and_annotate returns (annotated_frame, final_detections)
        try:
            annotated_frame, final_detections = double_check_and_annotate(model, processed_frame, all_boxes)
        except Exception as e:
            print(f"Error in double_check_and_annotate: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use simple detection without annotation
            final_detections = []
            for box in all_boxes:
                try:
                    if len(box) >= 6:
                        x1, y1, x2, y2, conf, cls_id = box[:6]
                        if conf >= MIN_CONFIDENCE_THRESHOLD:
                            final_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)])
                except Exception as box_error:
                    print(f"Error processing box in fallback: {box_error}")
                    continue
            annotated_frame = processed_frame
        
        # Convert to JSON format
        detections = []
        try:
            for det in final_detections:
                try:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                    else:
                        print(f"Invalid detection format: {det}")
                        continue
                    
                    # Only return threat detections
                    if int(cls_id) != 0:
                        continue
            
                    detection = {
                        'class': 'threat',
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
                except Exception as det_error:
                    print(f"Error processing detection: {det_error}, det: {det}")
                    continue
        except Exception as e:
            print(f"Error converting detections to JSON: {e}")
            detections = []
        
        return jsonify({
            'detections': detections,
            'status': 'success',
            'processing_mode': desc
        })
    
    except Exception as e:
        print(f"Error during image detection: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({
            'error': f"Image processing failed: {str(e)}",
            'status': 'error',
            'details': error_trace.split('\n')[-3] if error_trace else None
        }), 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded/processed files with proper MIME type for videos."""
    # Get the file extension
    ext = os.path.splitext(filename)[1].lower()
    
    # Set proper MIME type for video files
    if ext == '.mp4':
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            mimetype='video/mp4'
        )
    elif ext == '.webm':
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            mimetype='video/webm'
        )
    elif ext == '.avi':
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            mimetype='video/x-msvideo'
        )
    else:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API endpoint for video detection
@app.route('/detect_video', methods=['POST'])
def detect_video():
    """
    Endpoint for video detection.
    Uses the EXACT SAME function as test_video_file.py (video_processor.process_video).
    When user uploads video through web, it processes it the same way as test_video_file.py
    and displays results on the web.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    try:
        # Import video processor (same module used by test_video_file.py)
        from video_processor import process_video
        
        # Save video temporarily (this is the path that will be passed to process_video)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        
        video_abs_path = os.path.abspath(video_path)
        print(f"=" * 60)
        print(f"Video Detection - Web Upload")
        print(f"Video uploaded and saved to: {video_abs_path}")
        print(f"This path will be used by process_video() (same as test_video_file.py)")
        
        # Prepare output video path
        processed_name = f"processed_{os.path.splitext(file.filename)[0]}.mp4"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_name)
        
        # Process video using video_processor.process_video()
        # This is the EXACT SAME function called by test_video_file.py
        # test_video_file.py calls: process_video(video_path, show_preview=True)
        # Web app calls: process_video(video_path, output_path=processed_path, show_preview=False)
        # Both use the same processing logic, just different output options
        print(f"Processing video with model (using video_processor.process_video - same as test_video_file.py)...")
        result = process_video(video_path, output_path=processed_path, show_preview=False)
        
        # Check if processing was successful
        if result['status'] != 'success':
            error_msg = result.get('error', 'Unknown error during video processing')
            print(f"Error: {error_msg}")
            print(f"=" * 60)
            return jsonify({'error': error_msg, 'status': 'error'}), 500
        
        # Processing successful - same result format as test_video_file.py
        print(f"Video processing completed successfully!")
        print(f"Total frames processed: {result['total_frames']}")
        print(f"Total detections: {len(result['detections'])}")
        print(f"Processed video saved to: {os.path.abspath(processed_path)}")
        
        # Convert video to browser-compatible format (H.264)
        # OpenCV's mp4v codec is not supported by browsers, so we need to convert
        browser_video_name = f"browser_{os.path.splitext(file.filename)[0]}.mp4"
        browser_video_path = os.path.join(app.config['UPLOAD_FOLDER'], browser_video_name)
        
        if convert_video_for_browser(processed_path, browser_video_path):
            # Use the browser-compatible video
            final_video_name = browser_video_name
            final_video_path = browser_video_path
            # Remove the original processed video (mp4v format)
            try:
                if os.path.exists(processed_path) and processed_path != browser_video_path:
                    os.remove(processed_path)
                    print(f"Removed original processed video: {processed_path}")
            except Exception as e:
                print(f"Warning: Could not remove original processed video: {e}")
        else:
            # Fallback to original if conversion fails
            final_video_name = processed_name
            final_video_path = processed_path
        
        print(f"Final video for browser: {os.path.abspath(final_video_path)}")
        print(f"=" * 60)
        
        # Clean up temporary input file
        try:
            os.remove(video_path)
            print(f"Temporary video file removed: {video_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
        
        # Verify output video was created
        if not os.path.exists(final_video_path):
            return jsonify({'error': 'Processed video file was not created', 'status': 'error'}), 500
        
        # Build absolute URL for the processed video (for web display)
        processed_url = url_for('serve_upload', filename=final_video_name, _external=True)
        
        # Return results to web (same data structure that test_video_file.py would show)
        # This data will be displayed on the web interface
        response_data = {
            'detections': result['detections'],
            'status': 'success',
            'total_frames': result['total_frames'],
            'processed_video_url': processed_url,
            'message': f'Video processed successfully. Model analyzed {result["total_frames"]} frames. Found {len(result["detections"])} detections.'
        }
        
        print(f"Returning results to web interface:")
        print(f"  - Total frames: {response_data['total_frames']}")
        print(f"  - Detections: {len(response_data['detections'])}")
        print(f"  - Video URL: {processed_url}")
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error during video detection: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary files if they exist
        try:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.remove(processed_path)
        except:
            pass
        
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Flask Threat Detection System...")
    print("="*50 + "\n")
    
    # Try to load model on startup
    model = load_model()
    if model is None:
        print("WARNING: Model not loaded. Some features may not work.")
    else:
        print("Model loaded successfully!")
    
    print("\nServer starting on http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
