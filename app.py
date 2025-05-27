import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from datetime import datetime, timedelta, timezone
import uuid
import logging
import shutil
import requests
import json
import cv2
import io
import base64
import traceback
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase Admin SDK
db = None
try:
    app_name = "logo_adder_app"
    existing_apps = firebase_admin._apps
    if app_name not in existing_apps:
        try:
            firebase_credentials = st.secrets["firebase"]["credential"]
            cred_dict = json.loads(firebase_credentials)
            cred = credentials.Certificate(cred_dict)
            logging.info("Loaded Firebase credentials from st.secrets")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load credentials from st.secrets: {str(e)}. Falling back to local file.")
            try:
                cred = credentials.Certificate("logoadder-d22b5-firebase-adminsdk.json")
                logging.info("Loaded Firebase credentials from local file")
            except Exception as e:
                logging.error(f"Failed to load local credentials: {str(e)}")
                cred = None
        if cred:
            firebase_admin.initialize_app(cred, name=app_name)
            db = firestore.client(app=firebase_admin.get_app(app_name))
            logging.info("Firebase Admin SDK and Firestore initialized successfully")
        else:
            logging.error("No valid Firebase credentials provided.")
            st.error("Firebase credentials missing. Contact support.")
    else:
        db = firestore.client(app=firebase_admin.get_app(app_name))
        logging.info("Firebase Admin SDK already initialized, using existing Firestore client")
except Exception as e:
    logging.error(f"Unexpected error initializing Firebase: {str(e)}\n{traceback.format_exc()}")
    st.error("Failed to initialize Firebase. Contact support.")

# Firebase Web API Key
try:
    FIREBASE_API_KEY = st.secrets["firebase"]["api_key"]
    logging.info("Loaded Firebase API key from st.secrets")
except KeyError:
    FIREBASE_API_KEY = "AIzaSyD5DufwXe2cOPZniy-3K-LTRA-csWcbWEg"
    logging.warning("Using fallback Firebase API key")
    

# Load Firebase Web API key from secrets
try:
    api_key = st.secrets["firebase"]["api_key"]
    logging.info("Loaded Firebase API key from st.secrets")
except KeyError as e:
    logging.error("Failed to load Firebase API key from st.secrets")
    raise KeyError("Firebase API key not found in st.secrets['firebase']['api_key']")


# Configuration
class Config:
    LOGO_SIZE_PERCENT = 0.5  # Base scale for preset logo positions (50%)
    LOGO_TRANSPARENCY = 0.3  # Default transparency
    LOGO_OFFSET_PERCENT = 0.05  # Offset from edges for preset positions (5%)
    DEFAULT_MAX_EXECUTIONS = 27
    EXECUTION_COLLECTION = "executions"
    LICENSE_COLLECTION = "licenses"
    FOLDERS = ["Logos", "Media", "Logoed_Media", "Blur_Preview"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIDENCE_THRESHOLD = 0.2
    BLUR_KERNEL_FACTOR = 0.3
    YOLO_CONFIDENCE = 0.6
    USE_JAVASCRIPT_DOWNLOAD = False
    ADMIN_USER_ID = "CO9n9TnhWoclEtyuH8jfzsXs7tt2"
    DNN_PROTO_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt")
    DNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

# State management
class State:
    execution_count = 0
    max_executions = Config.DEFAULT_MAX_EXECUTIONS
    license_expiry = None
    subscription_expiry = None
    blur_enabled = True
    face_detector = None
    face_mesh = None
    yolo_model = None
    tracker = None
    dnn_net = None
    infinite_count = False

# Ensure directories exist
def ensure_directories(base_path):
    for folder in Config.FOLDERS:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# Load DNN model for face detection
def load_dnn_model():
    logging.info("Checking for DNN model files at: %s and %s", Config.DNN_PROTO_PATH, Config.DNN_MODEL_PATH)
    if not (os.path.exists(Config.DNN_PROTO_PATH) and os.path.exists(Config.DNN_MODEL_PATH)):
        logging.error("DNN model files not found.")
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(Config.DNN_PROTO_PATH, Config.DNN_MODEL_PATH)
        logging.info("DNN model loaded successfully.")
        return net
    except Exception as e:
        logging.error("Error loading DNN model: %s", str(e))
        return None

# Initialize AI models
def initialize_ai_models():
    logging.info("Initializing AI models for face detection, landmarks, and body detection")
    try:
        mp_face_detection = mp.solutions.face_detection
        State.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=Config.CONFIDENCE_THRESHOLD
        )
        mp_face_mesh = mp.solutions.face_mesh
        State.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=Config.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.5
        )
        State.yolo_model = YOLO("yolov8n.pt")
        State.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        State.dnn_net = load_dnn_model()
        logging.info("AI models initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing AI models: {str(e)}")
        State.face_detector = None
        State.face_mesh = None
        State.yolo_model = None
        State.tracker = None
        State.dnn_net = None
        st.warning("AI models failed to load. Blurring functionality disabled.")

# Detect bodies using YOLO
def detect_bodies(image, yolo_model):
    if yolo_model is None:
        logging.warning("YOLO model not loaded. Skipping body detection.")
        return []
    try:
        results = yolo_model(image, classes=[0], conf=Config.YOLO_CONFIDENCE)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        logging.info(f"Detected {len(boxes)} bodies")
        return boxes
    except Exception as e:
        logging.error(f"Error in body detection: {str(e)}")
        return []

# Check if face is within or near a body
def is_face_near_body(face_box, body_boxes, margin=0.2):
    if not body_boxes:
        logging.info("No bodies detected, allowing all faces")
        return True
    fx1, fy1, fx2, fy2 = face_box
    face_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
    for bx1, by1, bx2, by2 in body_boxes:
        bw, bh = bx2 - bx1, by2 - by1
        bx1_m, by1_m = bx1 - bw * margin, by1 - bh * margin
        bx2_m, by2_m = bx2 + bw * margin, by2 + bh * margin
        if (bx1_m <= face_center[0] <= bx2_m and by1_m <= face_center[1] <= by2_m):
            return True
    return False

# Get nose tip landmark
def get_nose_tip_landmark(image, face_landmarks):
    if not face_landmarks:
        return None
    nose_tip_idx = 1
    landmark = face_landmarks.landmark[nose_tip_idx]
    h, w = image.shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

# Process frame for blurring faces
def process_frame(frame, face_detector, face_mesh, yolo_model, tracker, blur_enabled, review_mode=False):
    if not blur_enabled or any(model is None for model in [face_detector, face_mesh, yolo_model, tracker]):
        logging.info(f"Blur skipped: blur_enabled={blur_enabled}, models_loaded={all(model is not None for model in [face_detector, face_mesh, yolo_model, tracker])}")
        return frame, []
    logging.info("Processing frame for face blurring")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    output_frame = frame.copy()
    blurred_regions = []

    body_boxes = detect_bodies(rgb_frame, yolo_model)
    face_results = face_detector.process(rgb_frame)
    detections = []
    if face_results.detections:
        logging.info(f"Detected {len(face_results.detections)} faces")
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            x2, y2 = x1 + w, y1 + h
            if is_face_near_body((x1, y1, x2, y2), body_boxes):
                conf = detection.score[0]
                detections.append([[x1, y1, w, h], conf, 0])
            else:
                logging.info(f"Face at ({x1}, {y1}, {x2}, {y2}) filtered out (no body nearby)")
    else:
        logging.warning("No faces detected by MediaPipe FaceDetection")

    tracks = tracker.update_tracks(detections, frame=rgb_frame)
    valid_faces = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr().astype(int)
        x1, y1, x2, y2 = bbox
        valid_faces.append((x1, y1, x2, y2, track.track_id))
    logging.info(f"Valid tracked faces: {len(valid_faces)}")

    for x1, y1, x2, y2, track_id in valid_faces:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            logging.warning(f"Invalid face bbox: ({x1}, {y1}, {x2}, {y2})")
            continue

        face_roi = rgb_frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            logging.warning(f"Empty face ROI at ({x1}, {y1}, {x2}, {y2})")
            continue
        mesh_results = face_mesh.process(face_roi)
        nose_tip = None
        if mesh_results.multi_face_landmarks:
            for landmarks in mesh_results.multi_face_landmarks:
                nose_tip = get_nose_tip_landmark(frame, landmarks)
                if nose_tip:
                    nx, ny = nose_tip
                    nx = nx + x1
                    ny = ny + y1
                    nose_tip = (nx, ny)
                    break
        else:
            logging.warning(f"No face landmarks detected for face at ({x1}, {y1}, {x2}, {y2})")

        blur_y2 = nose_tip[1] if nose_tip else y1 + int((y2 - y1) * 0.75)
        blur_y2 = min(blur_y2, y2)
        if blur_y2 <= y1:
            logging.warning(f"Invalid blur region: y1={y1}, blur_y2={blur_y2}")
            continue

        face_width = x2 - x1
        kernel_size = int(face_width * Config.BLUR_KERNEL_FACTOR)
        kernel_size = max(5, kernel_size // 2 * 2 + 1)

        roi = output_frame[y1:blur_y2, x1:x2]
        if roi.size == 0:
            logging.warning(f"Empty blur ROI at ({x1}, {y1}, {x2}, {blur_y2})")
            continue
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        mask = np.zeros_like(roi, dtype=np.uint8)
        mask_height = blur_y2 - y1
        gradient = np.linspace(1, 0, int(mask_height * 0.2)).reshape(-1, 1, 1)
        mask[:int(mask_height * 0.2)] = (gradient * 255).astype(np.uint8)
        mask[int(mask_height * 0.2):] = 255
        try:
            blurred_roi = cv2.seamlessClone(blurred_roi, roi, mask, (roi.shape[1] // 2, roi.shape[0] // 2), cv2.NORMAL_CLONE)
            output_frame[y1:blur_y2, x1:x2] = blurred_roi
        except Exception as e:
            logging.error(f"Error in seamlessClone: {str(e)}")
            continue

        blurred_regions.append({
            "bbox": (x1, y1, x2, blur_y2),
            "track_id": track_id,
            "frame": output_frame.copy()
        })

    logging.info(f"Blur applied to {len(blurred_regions)} regions")
    return output_frame, blurred_regions

# Process image for blurring
def process_image(image, dnn_net, blur_enabled):
    if not blur_enabled or not State.blur_enabled or dnn_net is None:
        logging.info(f"Blur skipped for image: blur_enabled={blur_enabled}, State.blur_enabled={State.blur_enabled}, dnn_net={dnn_net is not None}")
        return image, []
    logging.info("Processing image for face blurring with DNN")
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_array.shape[:2]
    output_frame = img_array.copy()
    blurred_regions = []

    blob = cv2.dnn.blobFromImage(output_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > Config.CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                logging.warning(f"Invalid face bbox: ({x1}, {y1}, {x2}, {y2})")
                continue
            face_width = x2 - x1
            kernel_size = int(face_width * Config.BLUR_KERNEL_FACTOR)
            kernel_size = max(5, kernel_size // 2 * 2 + 1)
            blur_y2 = y1 + int((y2 - y1) * 0.75)
            blur_y2 = min(blur_y2, y2)
            if blur_y2 <= y1:
                logging.warning(f"Invalid blur region: y1={y1}, blur_y2={blur_y2}")
                continue
            roi = output_frame[y1:blur_y2, x1:x2]
            if roi.size == 0:
                logging.warning(f"Empty blur ROI at ({x1}, {y1}, {x2}, {blur_y2})")
                continue
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            mask = np.zeros_like(roi, dtype=np.uint8)
            mask_height = blur_y2 - y1
            gradient = np.linspace(1, 0, int(mask_height * 0.2)).reshape(-1, 1, 1)
            mask[:int(mask_height * 0.2)] = (gradient * 255).astype(np.uint8)
            mask[int(mask_height * 0.2):] = 255
            try:
                blurred_roi = cv2.seamlessClone(blurred_roi, roi, mask, (roi.shape[1] // 2, roi.shape[0] // 2), cv2.NORMAL_CLONE)
                output_frame[y1:blur_y2, x1:x2] = blurred_roi
            except Exception as e:
                logging.error(f"Error in seamlessClone: {str(e)}")
                continue
            blurred_regions.append({
                "bbox": (x1, y1, x2, blur_y2),
                "track_id": f"image_{i}",
                "frame": output_frame.copy()
            })

    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    logging.info(f"Blur applied to image: {len(blurred_regions)} faces detected")
    return Image.fromarray(output_frame).convert('RGBA'), blurred_regions

# Process video for blurring
def process_video(video_path, output_path, face_detector, face_mesh, yolo_model, tracker, blur_enabled):
    if not blur_enabled or not State.blur_enabled or any(model is None for model in [face_detector, face_mesh, yolo_model, tracker]):
        logging.info(f"Blur skipped for video: blur_enabled={blur_enabled}, State.blur_enabled={State.blur_enabled}, models_loaded={all(model is not None for model in [face_detector, face_mesh, yolo_model, tracker])}")
        shutil.copy(video_path, output_path)
        return []
    logging.info(f"Applying blur to video: input={video_path}, output={output_path}")
    try:
        original_clip = VideoFileClip(video_path)
        audio = original_clip.audio  # Preserve original audio
        has_audio = audio is not None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            original_clip.close()
            return []
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        temp_output_path = output_path + "_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            original_clip.close()
            logging.error(f"Failed to initialize video writer: {temp_output_path}")
            return []

        blurred_regions = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            processed_frame, regions = process_frame(frame, face_detector, face_mesh, yolo_model, tracker, blur_enabled)
            out.write(processed_frame)
            for region in regions:
                region["frame_idx"] = frame_idx
            blurred_regions.extend(regions)
            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        video_clip = VideoFileClip(temp_output_path)
        if has_audio:
            video_clip = video_clip.set_audio(audio)
        video_clip.write_videofile(
            output_path,
            codec='libx264',
            audio=has_audio,
            audio_codec='aac' if has_audio else None,
            fps=fps,
            preset='fast',
            bitrate='5000k',
            temp_audiofile=os.path.join(Config.BASE_DIR, f"temp_audio_{uuid.uuid4()}.m4a")
        )
        video_clip.close()
        original_clip.close()
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logging.info(f"Removed temporary file: {temp_output_path}")
        logging.info(f"Video saved with blur: {output_path}, {len(blurred_regions)} regions blurred, audio={'preserved' if has_audio else 'none'}")
        return blurred_regions
    except Exception as e:
        logging.error(f"Error processing video blur: {str(e)}")
        shutil.copy(video_path, output_path)
        return []
    
# Review blurred regions
def review_blurred_regions(blurred_regions, media_type, base_path, media_name):
    st.subheader("Review Blurred Regions")
    approved = True
    preview_path = os.path.join(base_path, "Blur_Preview", f"preview_{media_name}")
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)

    if not blurred_regions:
        st.warning("No faces detected for blurring. The image will be processed with the logo only.")
        return True

    if media_type == "image":
        frame = blurred_regions[0]["frame"]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Blurred Image Preview", use_container_width=True)
        for region in blurred_regions:
            x1, y1, x2, y2 = region["bbox"]
            st.write(f"Face at ({x1}, {y1}, {x2}, {y2})")
            approve = st.checkbox("Approve this blur", value=True, key=f"approve_image_{region['track_id']}")
            if not approve:
                approved = False
        cv2.imwrite(preview_path, frame)
    else:
        frame_indices = sorted(set(r["frame_idx"] for r in blurred_regions))
        sample_indices = frame_indices[::max(1, len(frame_indices) // 5)]
        for idx in sample_indices:
            regions = [r for r in blurred_regions if r["frame_idx"] == idx]
            if not regions:
                continue
            frame = regions[0]["frame"]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {idx} Preview", use_column_width=True)
            for region in regions:
                x1, y1, x2, y2 = region["bbox"]
                track_id = region["track_id"]
                st.write(f"Frame {idx}, Face ID {track_id} at ({x1}, {y1}, {x2}, {y2})")
                approve = st.checkbox("Approve this blur", value=True, key=f"approve_video_{idx}_{track_id}")
                if not approve:
                    approved = False
        if sample_indices:
            frame = blurred_regions[0]["frame"]
            cv2.imwrite(preview_path, frame)

    if not approved:
        st.warning("Some blurred regions were not approved. Please adjust or reprocess.")
    return approved

# Logo position component
def logo_position_component(media_file, key_prefix="logo_position"):
    if not media_file:
        return None
    try:
        image = Image.open(media_file).convert("RGBA")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_src = f"data:image/png;base64,{img_base64}"

        # Initialize transparency in session state if not set
        if 'logo_transparency' not in st.session_state:
            st.session_state.logo_transparency = Config.LOGO_TRANSPARENCY

        # Slider for transparency
        transparency = st.slider(
            "Logo Transparency (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.logo_transparency * 100),
            step=1,
            key=f"{key_prefix}_transparency"
        )
        st.session_state.logo_transparency = transparency / 100.0
        logging.info(f"Transparency set to {transparency}%")

        component_js = """
        <script>
            function updatePosition(event, keyPrefix) {
                const img = event.target;
                const rect = img.getBoundingClientRect();
                const scaleX = img.naturalWidth / rect.width;
                const scaleY = img.naturalHeight / rect.height;
                const scaledX = Math.round((event.clientX - rect.left) * scaleX);
                const scaledY = Math.round((event.clientY - rect.top) * scaleY);
                const input = document.createElement('input');
                input.type = 'hidden';
                input.id = keyPrefix + '_coords';
                input.value = JSON.stringify({x: scaledX, y: scaledY});
                document.body.appendChild(input);
                window.Streamlit.setComponentValue({x: scaledX, y: scaledY});
                const display = document.getElementById(keyPrefix + '_display');
                if (display) {
                    display.textContent = `Selected Position: (${scaledX}, ${scaledY})`;
                }
            }
        </script>
        <div>
            <img src="%s" id="%s_image" onclick="updatePosition(event, '%s')" style="cursor: crosshair; max-width: 100%%;" />
            <p id="%s_display">Click image to set logo position</p>
        </div>
        """ % (img_src, key_prefix, key_prefix, key_prefix)
        coords = st.components.v1.html(component_js, height=600, scrolling=True)
        return coords
    except Exception as e:
        logging.error(f"Error in logo_position_component: {str(e)}")
        st.error(f"Failed to load image for positioning: {str(e)}")
        return None

# Overlay logo on image
def overlay_logo_on_image(image, logo_path, position="center", media_key=None):
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            logging.info("Image converted to RGBA")
        logo = Image.open(logo_path).convert("RGBA")
        img_width, img_height = image.size
        
        # Get manual positioning parameters if available
        if media_key and st.session_state.manual_positioning and media_key in st.session_state.logo_positions:
            params = st.session_state.logo_positions[media_key]
            x_pos = params.get("x_pos", 500)
            y_pos = params.get("y_pos", 500)
            scale = params.get("scale", 1.0)
            opacity = params.get("opacity", 1.0)
            rotation = params.get("rotation", 0)
            base_scale = 0.1  # Base scale for manual positioning
            logging.info(f"Using manual position for {media_key}: x={x_pos}, y={y_pos}, scale={scale}, opacity={opacity}, rotation={rotation}")
        else:
            x_pos, y_pos = None, None
            scale = 1.0
            opacity = Config.LOGO_TRANSPARENCY
            rotation = 0
            base_scale = Config.LOGO_SIZE_PERCENT  # Use 0.5 for preset positions
            logging.info(f"Using preset position: {position}")

        # Resize logo using appropriate base scale and user-specified scale
        max_logo_width = int(img_width * base_scale * scale)
        if max_logo_width > 0:
            logo_ratio = max_logo_width / logo.size[0]
            logo_height = int(logo.size[1] * logo_ratio)
            logo = logo.resize((max_logo_width, logo_height), Image.Resampling.LANCZOS)
        else:
            logging.warning(f"Invalid logo width {max_logo_width} for scale {scale}. Using original size.")
        
        # Apply rotation
        if rotation != 0:
            logo = logo.rotate(rotation, expand=True)
        
        # Apply transparency
        logo_array = np.array(logo)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * opacity).astype(np.uint8)
        logo = Image.fromarray(logo_array)

        # Determine position
        offset = int(min(img_width, img_height) * Config.LOGO_OFFSET_PERCENT)
        if x_pos is not None and y_pos is not None:
            x, y = x_pos, y_pos
        else:
            position_map = {
                "top": ((img_width - logo.size[0]) // 2, offset),
                "center": ((img_width - logo.size[0]) // 2, (img_height - logo.size[1]) // 2),
                "bottom": ((img_width - logo.size[0]) // 2, img_height - logo.size[1] - offset),
                "left": (offset, (img_height - logo.size[1]) // 2),
                "right": (img_width - logo.size[0] - offset, (img_height - logo.size[1]) // 2),
                "top_left": (offset, offset),
                "top_right": (img_width - logo.size[0] - offset, offset),
                "left_center": (offset, (img_height - logo.size[1]) // 2),
                "right_center": (img_width - logo.size[0] - offset, (img_height - logo.size[1]) // 2),
                "left_bottom": (offset, img_height - logo.size[1] - offset),
                "right_bottom": (img_width - logo.size[0] - offset, img_height - logo.size[1] - offset)
            }
            x, y = position_map.get(position.lower(), position_map["center"])

        # Ensure logo stays within image bounds
        x, y = max(0, x), max(0, y)
        if x + logo.size[0] > img_width:
            x = img_width - logo.size[0]
        if y + logo.size[1] > img_height:
            y = img_height - logo.size[1]

        # Composite logo onto image
        output = Image.new("RGBA", image.size)
        output.paste(image, (0, 0))
        output.paste(logo, (x, y), logo)
        logging.info(f"Logo overlaid on image at ({x}, {y}) with size={logo.size}, scale={scale}, base_scale={base_scale}, opacity={opacity}, rotation={rotation}")
        return output
    except Exception as e:
        logging.error(f"Error overlaying logo on image: {str(e)}")
        st.error(f"Failed to overlay logo: {str(e)}")
        return image

# Overlay logo on video
def overlay_logo_on_video(video_path, logo_path, output_path, position="center", media_key=None):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio  # Preserve original audio
        vid_width, vid_height = video.size
        logo = Image.open(logo_path).convert("RGBA")

        # Get manual positioning parameters if available
        if media_key and st.session_state.manual_positioning and media_key in st.session_state.logo_positions:
            params = st.session_state.logo_positions[media_key]
            x_pos = params.get("x_pos", 500)
            y_pos = params.get("y_pos", 500)
            scale = params.get("scale", 1.0)
            opacity = params.get("opacity", 1.0)
            rotation = params.get("rotation", 0)
            base_scale = 0.1  # Base scale for manual positioning
            logging.info(f"Using manual position for video {media_key}: x={x_pos}, y={y_pos}, scale={scale}, opacity={opacity}, rotation={rotation}")
        else:
            x_pos, y_pos = None, None
            scale = 1.0
            opacity = Config.LOGO_TRANSPARENCY
            rotation = 0
            base_scale = Config.LOGO_SIZE_PERCENT  # Use 0.5 for preset positions
            logging.info(f"Using preset position for video: {position}")

        # Resize logo using appropriate base scale and user-specified scale
        max_logo_width = int(vid_width * base_scale * scale)
        if max_logo_width > 0:
            logo_ratio = max_logo_width / logo.size[0]
            logo_height = int(logo.size[1] * logo_ratio)
            logo = logo.resize((max_logo_width, logo_height), Image.Resampling.LANCZOS)
        else:
            logging.warning(f"Invalid logo width {max_logo_width} for scale {scale}. Using original size.")
        
        # Apply rotation
        if rotation != 0:
            logo = logo.rotate(rotation, expand=True)
        
        # Apply transparency
        logo_array = np.array(logo)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * opacity).astype(np.uint8)
        logo = Image.fromarray(logo_array)

        # Save temporary logo file
        temp_logo_path = os.path.join(Config.BASE_DIR, f"temp_logo_{uuid.uuid4()}.png")
        logo.save(temp_logo_path, "PNG")
        logo_clip = ImageClip(temp_logo_path).set_duration(video.duration)

        # Determine position
        offset = int(min(vid_width, vid_height) * Config.LOGO_OFFSET_PERCENT)
        if x_pos is not None and y_pos is not None:
            x, y = x_pos, y_pos
        else:
            position_map = {
                "top": ((vid_width - logo.size[0]) // 2, offset),
                "center": ((vid_width - logo.size[0]) // 2, (vid_height - logo.size[1]) // 2),
                "bottom": ((vid_width - logo.size[0]) // 2, vid_height - logo.size[1] - offset),
                "left": (offset, (vid_height - logo.size[1]) // 2),
                "right": (vid_width - logo.size[0] - offset, (vid_height - logo.size[1]) // 2),
                "top_left": (offset, offset),
                "top_right": (vid_width - logo.size[0] - offset, offset),
                "left_center": (offset, (vid_height - logo.size[1]) // 2),
                "right_center": (vid_width - logo.size[0] - offset, (vid_height - logo.size[1]) // 2),
                "left_bottom": (offset, vid_height - logo.size[1] - offset),
                "right_bottom": (vid_width - logo.size[0] - offset, vid_height - logo.size[1] - offset)
            }
            x, y = position_map.get(position.lower(), position_map["center"])

        # Ensure logo stays within video bounds
        x, y = max(0, x), max(0, y)
        if x + logo.size[0] > vid_width:
            x = vid_width - logo.size[0]
        if y + logo.size[1] > vid_height:
            y = vid_height - logo.size[1]

        logo_clip = logo_clip.set_position((x, y))
        final_clip = CompositeVideoClip([video, logo_clip])
        if audio:
            final_clip = final_clip.set_audio(audio)
        
        # Write output video
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio=audio is not None,
            audio_codec="aac" if audio else None,
            fps=video.fps,
            preset="medium",
            bitrate="5000k",
            temp_audiofile=os.path.join(Config.BASE_DIR, f"temp_audio_{uuid.uuid4()}.m4a")
        )
        video.close()
        final_clip.close()
        if os.path.exists(temp_logo_path):
            os.remove(temp_logo_path)
        logging.info(f"Video saved with logo to {output_path} at ({x}, {y}), size={logo.size}, scale={scale}, base_scale={base_scale}, opacity={opacity}, rotation={rotation}, audio={'preserved' if audio else 'none'}")
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        st.error(f"Failed to process video: {str(e)}")
        shutil.copy(video_path, output_path)



# Check license and execution count
def check_license(user_id, force_refresh=False):
    if user_id == Config.ADMIN_USER_ID:
        logging.info(f"Admin user {user_id} bypasses license and subscription checks.")
        State.execution_count = 0
        State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
        State.infinite_count = True
        State.blur_enabled = True
        State.license_expiry = datetime.now(timezone.utc) + timedelta(days=3650)
        State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=3650)
        return True
    if not user_id:
        logging.error("No user_id provided for license check.")
        st.error("User not authenticated. Please log in.")
        return False
    if db is None:
        logging.error("Firestore client not initialized. Using fallback count.")
        st.warning("Firestore unavailable. Using local execution count.")
        if not hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = 0
        State.execution_count = st.session_state.local_execution_count
        State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
        State.infinite_count = False
        State.blur_enabled = True
        State.execution_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        if State.execution_count >= State.max_executions:
            logging.warning("Local execution limit reached.")
            st.error("Execution limit reached. Contact support for a new license.")
            return False
        if datetime.now(timezone.utc) > State.subscription_expiry:
            logging.warning("Local subscription expired.")
            st.error("Subscription expired. Contact support for a new license.")
            return False
        return True

    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            State.execution_count = data.get("count", 0)
            State.max_executions = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
            State.infinite_count = data.get("infinite_count", False)
            State.blur_enabled = data.get("blur_enabled", True)
            State.execution_expiry = data.get("expiry", datetime.now(timezone.utc))
            State.subscription_expiry = data.get("subscription_expiry", datetime.now(timezone.utc))
            if State.execution_expiry.tzinfo is None:
                State.execution_expiry = State.execution_expiry.replace(tzinfo=timezone.utc)
            if State.subscription_expiry.tzinfo is None:
                State.subscription_expiry = State.subscription_expiry.replace(tzinfo=timezone.utc)
            logging.info(f"License checked for user {user_id}: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, blur={State.blur_enabled}, expiry={State.execution_expiry}, sub_expiry={State.subscription_expiry}")
        else:
            State.execution_count = 0
            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
            State.infinite_count = False
            State.blur_enabled = True
            State.execution_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            doc_ref.set({
                "user_id": user_id,
                "count": 0,
                "max_executions": State.max_executions,
                "infinite_count": False,
                "blur_enabled": True,
                "expiry": State.execution_expiry,
                "subscription_expiry": State.subscription_expiry,
                "created_at": datetime.now(timezone.utc)
            })
            logging.info(f"New license created for user {user_id}: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, blur_enabled={State.blur_enabled}, expiry={State.execution_expiry}, sub_expiry={State.subscription_expiry}")

        if State.infinite_count:
            logging.info(f"User {user_id} has infinite executions.")
            return True
        if datetime.now(timezone.utc) > State.execution_expiry:
            logging.warning(f"License expired for user {user_id}.")
            st.error("License expired. Contact support for a new license.")
            return False
        if datetime.now(timezone.utc) > State.subscription_expiry:
            logging.warning(f"Subscription expired for user {user_id}.")
            st.error("Subscription expired. Contact support for a new license.")
            return False
        if State.execution_count >= State.max_executions:
            logging.warning(f"Execution limit reached for user {user_id}.")
            st.error("Execution limit reached. Contact support for a new license.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking license for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error checking license: {str(e)}. Contact support.")
        return False

# Increment execution count
def increment_execution_count(user_id, file_name):
    if user_id == Config.ADMIN_USER_ID:
        logging.info(f"Admin user {user_id} bypasses execution count increment for {file_name}.")
        return
    if not user_id:
        logging.error(f"No user ID provided for execution count increment for {file_name}.")
        return
    if db is None:
        logging.warning(f"Firestore unavailable, using local count for {file_name}")
        if not hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = 0
        st.session_state.local_execution_count += 1
        State.execution_count = st.session_state.local_execution_count
        logging.info(f"Local execution count incremented to {State.execution_count} for user {user_id}, file {file_name}")
        return

    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("infinite_count", False):
                logging.info(f"User {user_id} has infinite count, skipping increment for {file_name}.")
                return
        doc_ref.update({
            "count": firestore.Increment(1),
            "last_updated": datetime.now(timezone.utc),
            "last_file": file_name
        })
        State.execution_count += 1
        logging.info(f"Execution count incremented to {State.execution_count} for user {user_id}, file {file_name}")
    except Exception as e:
        logging.error(f"Error incrementing execution count for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error updating execution count: {str(e)}. Contact support.")

# Apply patch (admin function)
def apply_patch(user_id, new_count, days_valid, subscription_days, max_executions=None, blur_enabled=True):
    if not db:
        logging.error("Firestore unavailable. Cannot apply patch.")
        st.error("Firestore unavailable. Cannot generate patch.")
        return None
    try:
        patch_id = str(uuid.uuid4())
        doc_ref = db.collection(Config.LICENSE_COLLECTION).document(patch_id)
        expiry = datetime.now(timezone.utc) + timedelta(days=days_valid)
        subscription_expiry = datetime.now(timezone.utc) + timedelta(days=subscription_days)
        infinite_count = max_executions == 0
        if max_executions is None:
            max_executions = Config.DEFAULT_MAX_EXECUTIONS
        if max_executions > 0 and new_count > max_executions:
            logging.error(f"Invalid patch: new_count={new_count} > max={max_executions}")
            st.error(f"Start count {new_count} cannot exceed max executions {max_executions}.")
            return None
        doc_ref.set({
            "user_id": user_id,
            "new_count": new_count,
            "max_executions": max_executions,
            "infinite_count": infinite_count,
            "blur_enabled": blur_enabled,
            "enabled": True,
            "expiry": expiry,
            "subscription_expiry": subscription_expiry,
            "used": False,
            "created_at": datetime.now(timezone.utc)
        })
        execution_limit = f"infinite count" if infinite_count else f"max_executions={max_executions}"
        st.success(f"Patch generated: {patch_id} (Valid for {days_valid} days, Subscription valid for {subscription_days} days, Start count={new_count}, {execution_limit}, Blur enabled={blur_enabled})")
        logging.info(f"Patch generated: {patch_id}, for user {user_id}, count={new_count}, max={max_executions}, infinite={infinite_count}, blur_enabled={blur_enabled}, expiry={expiry}, subscription_expiry={subscription_expiry}")
        return patch_id
    except Exception as e:
        logging.error(f"Error generating patch for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error generating patch: {str(e)}")
        return None

# Validate and apply patch
def validate_patch(patch_id, user_id):
    if user_id == Config.ADMIN_USER_ID:
        logging.info(f"Admin user {user_id} bypasses patch validation.")
        st.info("Admin users do not need to apply patches.")
        return True
    if not db:
        logging.error("Firestore unavailable. Cannot validate patch.")
        st.error("Firestore unavailable. Cannot apply patch.")
        return False
    try:
        doc_ref = db.collection(Config.LICENSE_COLLECTION).document(patch_id)
        doc = doc_ref.get()
        if not doc.exists:
            logging.error(f"Patch {patch_id} not found for user {user_id}.")
            st.error("Invalid patch ID.")
            return False
        data = doc.to_dict()
        if data.get("used"):
            logging.error(f"Patch {patch_id} already used.")
            st.error("Patch has already been used.")
            return False
        expiry = data.get("expiry")
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            logging.error(f"Expired patch {patch_id} for user {user_id}.")
            st.error("Patch expired.")
            return False
        if data.get("user_id") != user_id:
            logging.error(f"Patch {patch_id} not authorized for user {user_id}.")
            st.error("Patch not authorized for this user.")
            return False
        execution_ref = db.collection(Config.EXECUTION_COLLECTION).document(user_id)
        execution_ref.set({
            "user_id": user_id,
            "count": data.get("new_count"),
            "max_executions": data.get("max_executions"),
            "infinite_count": data.get("infinite_count"),
            "blur_enabled": data.get("blur_enabled", True),
            "expiry": data.get("expiry"),
            "subscription_expiry": data.get("subscription_expiry"),
            "last_updated": datetime.now(timezone.utc)
        }, merge=True)
        doc_ref.update({
            "used": True,
            "used_at": datetime.now(timezone.utc)
        })
        State.execution_count = data.get("new_count")
        State.max_executions = data.get("max_executions")
        State.infinite_count = data.get("infinite_count")
        State.blur_enabled = data.get("blur_enabled", True)
        State.execution_expiry = data.get("expiry")
        State.subscription_expiry = data.get("subscription_expiry")
        if hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = data.get("new_count")
        logging.info(f"Patch {patch_id} applied for user {user_id}: count={data.get('new_count')}, max={data.get('max_executions')}, infinite={data.get('infinite_count')}, blur={data.get('blur_enabled')}, expiry={data.get('expiry')}, sub_expiry={data.get('subscription_expiry')}")
        st.success(f"Patch {patch_id} applied. Execution count set to {data.get('new_count')}, max={data.get('max_executions')}, blur={'enabled' if data.get('blur_enabled') else 'disabled'}")
        return True
    except Exception as e:
        logging.error(f"Error validating patch {patch_id} for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error applying patch: {str(e)}")
        return False

# JavaScript-based download
def trigger_multiple_downloads(files):
    if not files:
        return
    js_code = """
    <style>
        .download-all-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .download-all-btn:hover {
            background-color: #45a049;
        }
    </style>
    <script>
        function downloadFiles(files) {
            files.forEach(function(file) {
                var link = document.createElement('a');
                link.href = file.url;
                link.download = file.name;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
        }
    </script>
    """
    files_json = []
    for file_path, file_data in files:
        files_json.append({
            "url": f"data:application/octet-stream;base64,{base64.b64encode(file_data).decode('utf-8')}",
            "name": os.path.basename(file_path)
        })
    st.markdown(js_code, unsafe_allow_html=True)
    st.markdown(f"""
    <button class="download-all-btn" onclick="downloadFiles({json.dumps(files_json)})">Download All Files</button>
    """, unsafe_allow_html=True)

# Verify user
def verify_user(email, password):
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {
            "email": email.strip(),
            "password": password,
            "returnSecureToken": True
        }
        logging.info(f"Attempting to verify user: {email}")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        user_data = response.json()
        user_id = user_data.get("localId")
        logging.info(f"User verified successfully: {email}, user_id={user_id}")
        return user_id, None
    except requests.exceptions.HTTPError as e:
        try:
            error_response = response.json()
            error_code = error_response.get("error", {}).get("message", str(e))
            error_map = {
                "INVALID_EMAIL": "Invalid email format. Please check your email address.",
                "INVALID_PASSWORD": "Incorrect password. Try again or reset your password.",
                "USER_NOT_FOUND": "No account found with this email. Please sign up.",
                "USER_DISABLED": "This account is disabled. Contact support to enable it.",
                "TOO_MANY_ATTEMPTS": "Too many login attempts. Try again later.",
                "INVALID_LOGIN_CREDENTIALS": "Invalid email or password. Please sign up if you don't have an account or reset your password."
            }
            user_message = error_map.get(error_code, f"Authentication failed: {error_code}")
            logging.error(f"Error verifying user {email}: {error_code}")
            return None, user_message
        except:
            logging.error(f"Error verifying user {email}: {str(e)}")
            return None, f"Failed to verify user: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error verifying user {email}: {str(e)}")
        return None, f"An unexpected error occurred: {str(e)}"

# Debug tool to manage license
def debug_license_management(user_id):
    if not user_id:
        logging.warning("No user provided for debug license management.")
        st.error("No user ID provided for debug.")
        return

    st.subheader("Debug License Management")
    st.write(f"Firestore Status: {'Connected' if db else 'Disconnected'}")

    if not db:
        st.error("Firestore unavailable for debug mode.")
        return

    try:
        target_user_id = st.text_input("Target User ID", value=user_id, key="debug_user_id")
        if not target_user_id:
            return

        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(target_user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            current_count = data.get("count", 0)
            current_max = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
            current_infinite = data.get("infinite_count", False)
            current_blur_enabled = data.get("blur_enabled", True)
            current_expiry = data.get("expiry", datetime.now(timezone.utc))
            current_sub_expiry = data.get("subscription_expiry", datetime.now(timezone.utc))
            if current_expiry.tzinfo is None:
                current_expiry = current_expiry.replace(tzinfo=timezone.utc)
            if current_sub_expiry.tzinfo is None:
                current_sub_expiry = current_sub_expiry.replace(tzinfo=timezone.utc)
            st.write(f"Current Count: {current_count}")
            st.write(f"Max Executions: {current_max}")
            st.write(f"Infinite Count: {current_infinite}")
            st.write(f"Blur Enabled: {current_blur_enabled}")
            st.write(f"Expiry: {current_expiry}")
            st.write(f"Subscription Expiry: {current_sub_expiry}")
        else:
            st.warning(f"No license data found for user {target_user_id}")
            current_count = 0
            current_max = Config.DEFAULT_MAX_EXECUTIONS
            current_infinite = False
            current_blur_enabled = True
            current_expiry = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
            current_sub_expiry = datetime.now(timezone.utc).replace(tzinfo=timezone.utc) + timedelta(days=30)

        col1, col2, col3 = st.columns(3)
        with col1:
            new_count = st.number_input("Set Count", min_value=0, value=current_count, key="debug_count")
            if st.button("Apply Count", key="apply_count"):
                doc_ref.update({"count": new_count})
                if target_user_id == user_id:
                    State.execution_count = new_count
                st.success(f"Count set to {new_count}")
                logging.info(f"Debug: Set count to {new_count} for {target_user_id}")

            new_max = st.number_input("Set Max Executions", min_value=0, value=current_max, key="debug_max")
            if st.button("Apply Max", key="apply_max"):
                if new_max > 0 and new_count > new_max:
                    st.error(f"Max executions ({new_max}) cannot be less than count ({new_count}).")
                else:
                    doc_ref.update({
                        "max_executions": new_max,
                        "infinite_count": new_max == 0
                    })
                    if target_user_id == user_id:
                        State.max_executions = new_max
                        State.infinite_count = new_max == 0
                    st.success(f"Max executions set to {new_max} ({'infinite' if new_max == 0 else 'limited'})")
                    logging.info(f"Debug: Set max to {new_max}, infinite={new_max == 0} for {target_user_id}")
        with col2:
            expiry_days = st.number_input("Expiry Days", min_value=1, value=30, key="debug_expiry")
            if st.button("Apply Expiry", key="apply_expiry"):
                new_expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
                doc_ref.update({"expiry": new_expiry})
                if target_user_id == user_id:
                    State.execution_expiry = new_expiry
                st.success(f"Set expiry to {new_expiry}")
                logging.info(f"Debug: Set expiry to {new_expiry} for {target_user_id}")

            sub_days = st.number_input("Subscription Days", min_value=1, value=30, key="debug_sub")
            if st.button("Apply Sub Expiry", key="apply_sub"):
                new_sub_expiry = datetime.now(timezone.utc) + timedelta(days=sub_days)
                doc_ref.update({"subscription_expiry": new_sub_expiry})
                if target_user_id == user_id:
                    State.subscription_expiry = new_sub_expiry
                st.success(f"Set subscription expiry to {new_sub_expiry}")
                logging.info(f"Debug: Set subscription expiry to {new_sub_expiry} for {target_user_id}")
        with col3:
            blur_enabled = st.checkbox("Enable Blur", value=current_blur_enabled, key="debug_blur")
            if st.button("Apply Blur", key="apply_blur"):
                doc_ref.update({"blur_enabled": blur_enabled})
                if target_user_id == user_id:
                    State.blur_enabled = blur_enabled
                st.success(f"Blur {'enabled' if blur_enabled else 'disabled'}")
                logging.info(f"Debug: Set blur to {blur_enabled} for {target_user_id}")

            infinite = st.checkbox("Infinite Count", value=current_infinite, key="debug_infinite")
            if st.button("Apply Infinite", key="apply_infinite"):
                doc_ref.update({
                    "infinite_count": infinite,
                    "count": 0 if infinite else current_count,
                    "max_executions": 0 if infinite else Config.DEFAULT_MAX_EXECUTIONS
                })
                if target_user_id == user_id:
                    State.infinite_count = infinite
                    State.execution_count = 0 if infinite else current_count
                    State.max_executions = 0 if infinite else Config.DEFAULT_MAX_EXECUTIONS
                st.success(f"Infinite count set to {infinite}")
                logging.info(f"Debug: Set infinite={infinite}, count={0 if infinite else current_count}, max={0 if infinite else Config.DEFAULT_MAX_EXECUTIONS} for {target_user_id}")
    except Exception as e:
        logging.error(f"Error in debug license management for {target_user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error in debug: {str(e)}")





# Stub for increment_execution
def increment_execution(user_id, filename):
    logging.info(f"Execution incremented for user {user_id} on file {filename}")
    # Replace with actual Firestore update if available
    pass

# Debug tool to manage license limits (admin only)
def debug_license_limits(admin_user_id):
    if not admin_user_id:
        st.error("No user_id for debug license limits.")
        return
    st.subheader("Debug License Limits")
    st.write(f"Firestore Status: {'Connected' if db is not None else 'Disconnected'}")
    target_user_id = st.text_input("Enter Target User ID for Debug", key="debug_user_id")
    if target_user_id and db is not None:
        try:
            doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(target_user_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                current_count = data.get("count", 0)
                current_max = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
                current_infinite = data.get("infinite_count", False)
                current_blur_enabled = data.get("blur_enabled", True)
                current_expiry = data.get("expiry", datetime.now(timezone.utc))
                current_sub_expiry = data.get("subscription_expiry", datetime.now(timezone.utc))
                if current_expiry.tzinfo is None:
                    current_expiry = current_expiry.replace(tzinfo=timezone.utc)
                if current_sub_expiry.tzinfo is None:
                    current_sub_expiry = current_sub_expiry.replace(tzinfo=timezone.utc)
                st.write(f"Current Count: {current_count}")
                st.write(f"Current Max Executions: {current_max}")
                st.write(f"Infinite Count Enabled: {current_infinite}")
                st.write(f"Blur Enabled: {current_blur_enabled}")
                st.write(f"Current License Expiry: {current_expiry}")
                st.write(f"Current Subscription Expiry: {current_sub_expiry}")
            else:
                st.warning(f"No license found for user {target_user_id}.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                custom_count = st.number_input("Set Custom Execution Count", min_value=0, value=current_count, key="custom_count")
                if st.button("Apply Custom Count", key="apply_custom_count"):
                    doc_ref.update({"count": custom_count})
                    State.execution_count = custom_count if target_user_id == admin_user_id else State.execution_count
                    st.success(f"Execution count set to {custom_count}. Reload to continue.")
                    logging.info(f"Debug: Set execution count to {custom_count} for user {target_user_id}")
                custom_max = st.number_input("Set Custom Max Executions", min_value=0, value=current_max, key="custom_max_executions")
                if st.button("Apply Custom Max Executions", key="apply_custom_max"):
                    if custom_max > 0 and custom_max < current_count:
                        st.error(f"Max Executions ({custom_max}) cannot be less than Current Count ({current_count}).")
                        logging.error(f"Invalid max_executions: {custom_max} < count={current_count} for user {target_user_id}")
                    else:
                        doc_ref.update({
                            "max_executions": custom_max,
                            "infinite_count": custom_max == 0
                        })
                        if target_user_id == admin_user_id:
                            State.max_executions = custom_max
                            State.infinite_count = custom_max == 0
                        st.success(f"Max executions set to {custom_max}{' (infinite)' if custom_max == 0 else ''}. Reload to continue.")
                        logging.info(f"Debug: Set max_executions to {custom_max}, infinite_count={custom_max == 0} for user {target_user_id}")
            with col2:
                expiry_days = st.number_input("Set License Expiry Days", min_value=1, value=30, key="expiry_days")
                if st.button("Apply Expiry Days", key="apply_expiry_days"):
                    new_expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
                    doc_ref.update({"expiry": new_expiry})
                    if target_user_id == admin_user_id:
                        State.license_expiry = new_expiry
                    st.success(f"License expiry set to {new_expiry}. Reload to continue.")
                    logging.info(f"Debug: Set license expiry to {new_expiry} for user {target_user_id}")
            with col3:
                sub_expiry_days = st.number_input("Set Subscription Expiry Days", min_value=1, value=30, key="sub_expiry_days")
                if st.button("Apply Subscription Days", key="apply_sub_expiry_days"):
                    new_sub_expiry = datetime.now(timezone.utc) + timedelta(days=sub_expiry_days)
                    doc_ref.update({"subscription_expiry": new_sub_expiry})
                    if target_user_id == admin_user_id:
                        State.subscription_expiry = new_sub_expiry
                    st.success(f"Subscription expiry set to {new_sub_expiry}. Reload to continue.")
                    logging.info(f"Debug: Set subscription expiry to {new_sub_expiry} for user {target_user_id}")
            with col4:
                blur_enabled_toggle = st.checkbox("Enable Face Blurring", value=current_blur_enabled, key="blur_enabled_toggle")
                if st.button("Apply Blur Setting", key="apply_blur_enabled"):
                    doc_ref.update({"blur_enabled": blur_enabled_toggle})
                    if target_user_id == admin_user_id:
                        State.blur_enabled = blur_enabled_toggle
                    st.success(f"Face blurring {'enabled' if blur_enabled_toggle else 'disabled'}. Reload to continue.")
                    logging.info(f"Debug: Set blur_enabled to {blur_enabled_toggle} for user {target_user_id}")
                infinite_count_toggle = st.checkbox("Enable Infinite Count", value=current_infinite, key="infinite_count_toggle")
                if st.button("Apply Infinite Count", key="apply_infinite_count"):
                    if infinite_count_toggle:
                        doc_ref.update({"infinite_count": True, "count": 0, "max_executions": 0})
                        if target_user_id == admin_user_id:
                            State.infinite_count = True
                            State.execution_count = 0
                            State.max_executions = 0
                        st.success("Infinite count enabled, count and max_executions set to 0.")
                        logging.info(f"Debug: Enabled infinite count for user {target_user_id}")
                    else:
                        doc_ref.update({"infinite_count": False, "max_executions": Config.DEFAULT_MAX_EXECUTIONS})
                        if target_user_id == admin_user_id:
                            State.infinite_count = False
                            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
                        st.success(f"Infinite count disabled, max_executions set to {Config.DEFAULT_MAX_EXECUTIONS}.")
                        logging.info(f"Debug: Disabled infinite count for user {target_user_id}")
                if st.button("Reset Count to 0", key="reset_count"):
                    doc_ref.update({"count": 0})
                    State.execution_count = 0 if target_user_id == admin_user_id else State.execution_count
                    st.success("Execution count reset to 0. Reload to continue.")
                    logging.info(f"Debug: Reset execution count to 0 for user {target_user_id}")
                if st.button("Set Expiry to Past", key="set_expiry_past"):
                    past_expiry = datetime.now(timezone.utc) - timedelta(days=1)
                    doc_ref.update({"expiry": past_expiry, "subscription_expiry": past_expiry})
                    if target_user_id == admin_user_id:
                        State.license_expiry = past_expiry
                        State.subscription_expiry = past_expiry
                    st.success("Expiry set to yesterday. Reload to test expiry.")
                    logging.info(f"Debug: Set expiry to {past_expiry} for user {target_user_id}")
                if st.button("Delete License", key="delete_license"):
                    doc_ref.delete()
                    if target_user_id == admin_user_id:
                        State.execution_count = 0
                        State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
                        State.infinite_count = False
                        State.blur_enabled = True
                        State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
                        State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
                    st.success("License deleted. Reload to recreate.")
                    logging.info(f"Debug: Deleted license for user {target_user_id}")
        except Exception as e:
            logging.error(f"Error in debug license limits for user {target_user_id}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error accessing Firestore for user {target_user_id}: {str(e)}")
    elif target_user_id:
        st.warning("Firestore unavailable. Debug tools limited.")

# Generate preview image for display

def generate_preview_image(media_path, logo_path, x_pos=500, y_pos=500, scale=1.0, opacity=1.0, rotation=0):
    try:
        media_type = "image" if os.path.basename(media_path).lower().endswith((".jpg", ".jpeg", ".png")) else "video"
        logo_image = Image.open(logo_path).convert("RGBA")
        has_transparency = any(pixel[3] < 255 for pixel in logo_image.getdata())
        if not has_transparency:
            logging.warning(f"Logo {os.path.basename(logo_path)} has no transparent areas.")
        
        # Use Config.LOGO_TRANSPARENCY for preset positions (default x_pos, y_pos)
        if x_pos == 500 and y_pos == 500:
            opacity = Config.LOGO_TRANSPARENCY
            logging.info(f"Using preset opacity={opacity} for preview")
        
        # Apply opacity to logo
        if opacity < 1.0:
            logo_data = logo_image.split()
            alpha = logo_data[3].point(lambda x: x * opacity)
            logo_image.putalpha(alpha)
        
        # Apply rotation
        if rotation != 0:
            logo_image = logo_image.rotate(rotation, expand=True)
        
        if media_type == "image":
            media_image = Image.open(media_path).convert("RGBA")
            media_width, media_height = media_image.size
            # Use Config.LOGO_SIZE_PERCENT for preset positions, 0.1 for manual
            base_scale = 0.1 if x_pos != 500 or y_pos != 500 else Config.LOGO_SIZE_PERCENT
            logo_width = int(media_width * base_scale * scale)
            if logo_width > 0:
                logo_ratio = logo_width / logo_image.size[0]
                logo_height = int(logo_image.size[1] * logo_ratio)
                logo_image = logo_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            else:
                logging.warning(f"Invalid logo width {logo_width} for scale {scale}. Using original size.")
            x = min(max(0, x_pos), media_width - logo_width)
            y = min(max(0, y_pos), media_height - logo_height)
            preview_image = Image.new("RGBA", media_image.size)
            preview_image.paste(media_image, (0, 0))
            preview_image.paste(logo_image, (x, y), logo_image)
            output = io.BytesIO()
            preview_image.save(output, format="PNG")
            logging.info(f"Generated preview for {os.path.basename(media_path)} with logo size=({logo_width}, {logo_height}), scale={scale}, base_scale={base_scale}, opacity={opacity}, rotation={rotation}")
            return output.getvalue()
        else:
            video = VideoFileClip(media_path)
            frame = video.get_frame(0)
            media_image = Image.fromarray(frame).convert("RGBA")
            media_width, media_height = media_image.size
            # Use Config.LOGO_SIZE_PERCENT for preset positions, 0.1 for manual
            base_scale = 0.1 if x_pos != 500 or y_pos != 500 else Config.LOGO_SIZE_PERCENT
            logo_width = int(media_width * base_scale * scale)
            if logo_width > 0:
                logo_ratio = logo_width / logo_image.size[0]
                logo_height = int(logo_image.size[1] * logo_ratio)
                logo_image = logo_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            else:
                logging.warning(f"Invalid logo width {logo_width} for scale {scale}. Using original size.")
            x = min(max(0, x_pos), media_width - logo_width)
            y = min(max(0, y_pos), media_height - logo_height)
            preview_image = Image.new("RGBA", media_image.size)
            preview_image.paste(media_image, (0, 0))
            preview_image.paste(logo_image, (x, y), logo_image)
            output = io.BytesIO()
            preview_image.save(output, format="PNG")
            video.close()
            logging.info(f"Generated preview for {os.path.basename(media_path)} with logo size=({logo_width}, {logo_height}), scale={scale}, base_scale={base_scale}, opacity={opacity}, rotation={rotation}")
            return output.getvalue()
    except Exception as e:
        logging.error(f"Error generating preview for {os.path.basename(media_path)}: {str(e)}\n{traceback.format_exc()}")
        return None

# Streamlit app
def main():
    st.set_page_config(page_title="Logo Adder App", layout="wide")
    st.title("Logo Adder App")

    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "patch_applied" not in st.session_state:
        st.session_state.patch_applied = False
    if "logo_positions" not in st.session_state:
        st.session_state.logo_positions = {}
    if "manual_positioning" not in st.session_state:
        st.session_state.manual_positioning = False
    if "blur_enabled" not in st.session_state:
        st.session_state.blur_enabled = False
    if "selected_position" not in st.session_state:
        st.session_state.selected_position = "Center"
    if "auth_error" not in st.session_state:
        st.session_state.auth_error = None
    if "reset_message" not in st.session_state:
        st.session_state.reset_message = None
    if "output_files" not in st.session_state:
        st.session_state.output_files = []
    if "device_id" not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
        logging.info(f"Initialized session state with device_id: {st.session_state.device_id}")
        logging.info(f"Session state at start: user={st.session_state.user}, user_id={st.session_state.user_id}, device_id={st.session_state.device_id}, patch_applied={st.session_state.patch_applied}, blur_enabled={st.session_state.blur_enabled}, manual_positioning={st.session_state.manual_positioning}")

    # ... (Other main() code before sidebar remains unchanged)

    with st.sidebar:
        st.header("User Authentication")
        if not st.session_state.user:
            auth_choice = st.radio("Choose Action", ["Login", "Sign Up"], key="auth_choice")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if auth_choice == "Login":
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Login"):
                        if not email or not password:
                            st.session_state.auth_error = "Please enter both email and password."
                            logging.error("Login attempted without email or password")
                        else:
                            user_id, error = verify_user(email, password)
                            if user_id:
                                st.session_state.user = email
                                st.session_state.user_id = user_id
                                st.session_state.auth_error = None
                                st.session_state.reset_message = None
                                st.success(f"Logged in as {email}")
                                logging.info(f"User logged in: {email}, user_id={user_id}")
                                if check_license(user_id):
                                    st.session_state.patch_applied = True
                                else:
                                    st.session_state.patch_applied = False
                                st.rerun()
                            else:
                                st.session_state.auth_error = error
                                logging.error(f"Login failed for {email}: {error}")
                with col2:
                    if st.button("Forgot Password?"):
                        if not email:
                            st.session_state.reset_message = "Please enter your email to reset the password."
                            logging.error("Password reset attempted without email")
                        else:
                            try:
                                user = auth.get_user_by_email(email)
                                reset_link = auth.generate_password_reset_link(email)
                                st.session_state.reset_message = f"Password reset link sent to {email}. Check your inbox."
                                st.session_state.auth_error = None
                                logging.info(f"Password reset link generated for {email}")
                            except auth.UserNotFoundError:
                                st.session_state.reset_message = "Email not registered. Please sign up."
                                logging.error(f"Password reset failed: Email {email} not registered")
                            except Exception as e:
                                st.session_state.reset_message = f"Error generating reset link: {str(e)}"
                                logging.error(f"Error generating reset link for {email}: {str(e)}")
            else:  # Sign Up
                if st.button("Sign Up"):
                    if not email or not password:
                        st.session_state.auth_error = "Please enter both email and password."
                        logging.error("Sign up attempted without email or password")
                    else:
                        try:
                            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
                            payload = {
                                "email": email.strip(),
                                "password": password,
                                "returnSecureToken": True
                            }
                            response = requests.post(url, json=payload)
                            response.raise_for_status()
                            user_data = response.json()
                            user_id = user_data.get("localId")
                            st.session_state.user = email
                            st.session_state.user_id = user_id
                            st.session_state.auth_error = None
                            st.session_state.reset_message = None
                            st.success(f"Account created and logged in as {email}")
                            logging.info(f"User signed up: {email}, user_id={user_id}")
                            if check_license(user_id):
                                st.session_state.patch_applied = True
                            else:
                                st.session_state.patch_applied = False
                            st.rerun()
                        except requests.exceptions.HTTPError as e:
                            error_response = response.json()
                            error_code = error_response.get("error", {}).get("message", str(e))
                            error_map = {
                                "EMAIL_EXISTS": "This email is already registered. Try logging in.",
                                "INVALID_EMAIL": "Invalid email format. Please check your email address.",
                                "WEAK_PASSWORD": "Password must be at least 6 characters long."
                            }
                            user_message = error_map.get(error_code, f"Sign-up failed: {error_code}")
                            st.session_state.auth_error = user_message
                            logging.error(f"Sign-up failed for {email}: {error_code}")
                        except Exception as e:
                            st.session_state.auth_error = f"Unexpected error during sign-up: {str(e)}"
                            logging.error(f"Unexpected error during sign-up for {email}: {str(e)}")

            if st.session_state.auth_error:
                st.error(f"Error: {st.session_state.auth_error}")
            if st.session_state.reset_message:
                if "reset link" in st.session_state.reset_message.lower():
                    st.success(st.session_state.reset_message)
                else:
                    st.error(st.session_state.reset_message)
        else:
            st.write(f"Logged in as: {st.session_state.user}")
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.user_id = None
                st.session_state.patch_applied = False
                st.session_state.logo_positions = {}
                st.session_state.manual_positioning = False
                st.session_state.blur_enabled = False
                st.session_state.selected_position = "Center"
                st.session_state.auth_error = None
                st.session_state.reset_message = None
                st.session_state.output_files = []
                st.success("Logged out successfully.")
                logging.info(f"User logged out: {st.session_state.user}")
                st.rerun()

            st.markdown("---")
            st.header("Apply Patch")
            patch_id = st.text_input("Enter Patch ID")
            if st.button("Apply Patch"):
                if not patch_id:
                    st.error("Please enter a patch ID.")
                else:
                    if validate_patch(patch_id, st.session_state.user_id):
                        st.session_state.patch_applied = True
                        logging.info(f"Patch {patch_id} applied, forcing license refresh")
                        check_license(st.session_state.user_id, force_refresh=True)
                        st.rerun()

        if st.session_state.user_id == Config.ADMIN_USER_ID:
            st.markdown("---")
            st.header("Generate Patch (Admin)")
            patch_user_id = st.text_input("Target User ID for Patch")
            new_count = st.number_input("Start Execution Count", min_value=0, value=0)
            days_valid = st.number_input("Patch Validity (Days)", min_value=1, value=30)
            subscription_days_valid = st.number_input("Subscription Validity (Days)", min_value=1, value=30)
            max_executions = st.number_input("Max Executions (0 for infinite)", min_value=0, value=Config.DEFAULT_MAX_EXECUTIONS)
            blur_enabled = st.checkbox("Enable Face Blurring for Patch", value=True)
            if st.button("Generate Patch"):
                if not patch_user_id:
                    st.error("Please provide a target user ID.")
                else:
                    patch_id = apply_patch(patch_user_id, new_count, days_valid, subscription_days_valid, max_executions, blur_enabled)
                    if patch_id:
                        st.session_state.patch_applied = True
                        st.rerun()

        if st.session_state.user_id == Config.ADMIN_USER_ID:
            st.markdown("---")
            debug_license_management(st.session_state.user_id)

    if st.session_state.user is None or not st.session_state.patch_applied:
        if st.session_state.user is None:
            st.warning("Please log in to use the app.")
        elif not st.session_state.patch_applied:
            st.warning("Please apply a valid patch or ensure your license is active.")
        return

    ensure_directories(Config.BASE_DIR)
    if not all([State.face_detector, State.face_mesh, State.yolo_model, State.tracker, State.dnn_net]):
        initialize_ai_models()

    st.header("Upload Files")
    logo_file = st.file_uploader("Upload Logo (PNG with transparency recommended)", type=["png", "jpg", "jpeg"])
    media_files = st.file_uploader("Upload Media (Images or Videos)", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True)

    if media_files and media_files != st.session_state.get('last_media_files', []):
        st.session_state.output_files = []
        st.session_state.last_media_files = media_files

    st.header("Logo Position")
    position_options = ["center", "top", "bottom", "left", "right", "top_left", "top_right", "left_center", "right_center", "left_bottom", "right_bottom"]
    manual_positioning = st.checkbox("Enable Manual Positioning", value=st.session_state.manual_positioning, key="manual_positioning")
    if manual_positioning != st.session_state.manual_positioning:
        st.session_state.manual_positioning = manual_positioning

    position_option = st.selectbox(
        "Select Logo Position",
        position_options,
        index=position_options.index(st.session_state.selected_position.lower()),
        key="logo_position_select"
    )
    st.session_state.selected_position = position_option

    st.write("Quick Position Selectors:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Top", key="quick_top"):
            st.session_state.selected_position = "top"
            st.rerun()
    with col2:
        if st.button("Bottom", key="quick_bottom"):
            st.session_state.selected_position = "bottom"
            st.rerun()
    with col3:
        if st.button("Left", key="quick_left"):
            st.session_state.selected_position = "left"
            st.rerun()
    with col4:
        if st.button("Right", key="quick_right"):
            st.session_state.selected_position = "right"
            st.rerun()
    with col5:
        if st.button("Center", key="quick_center"):
            st.session_state.selected_position = "center"
            st.rerun()

    st.write("**Debug Info**")
    st.write(f"- Selected Position: {position_option}")
    st.write(f"- Manual Positioning Enabled: {st.session_state.manual_positioning}")
    st.write(f"- Logo File Uploaded: {'Yes' if logo_file else 'No'}")
    st.write(f"- Media Files Uploaded: {len(media_files) if media_files else 0}")

    custom_positions = []
    if st.session_state.manual_positioning:
        st.subheader("Manual Logo Positioning")
        if not logo_file or not media_files:
            st.warning("Please upload both a logo and at least one media file to configure manual positioning.")
            logging.info("Manual positioning UI skipped: logo_file or media_files missing")
        else:
            st.warning("Note: Manual positioning settings (X, Y, Scale, Rotation, Opacity) are applied to previews and output.")
            logo_path = os.path.join(Config.BASE_DIR, "Logos", logo_file.name)
            try:
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getvalue())
                logo_image = Image.open(logo_path).convert("RGBA")
                if not any(pixel[3] < 255 for pixel in logo_image.getdata()):
                    st.warning("Logo does not have transparent areas. Use a PNG with transparency for best results.")
            except Exception as e:
                st.error(f"Failed to save logo file: {str(e)}")
                logging.error(f"Error saving logo file to {logo_path}: {str(e)}\n{traceback.format_exc()}")
                return

            for media_file in media_files:
                media_key = media_file.name
                safe_media_key = ''.join(c if c.isalnum() else '_' for c in media_key).strip('_')
                if media_key not in st.session_state.logo_positions:
                    st.session_state.logo_positions[media_key] = {
                        "x_pos": 250,
                        "y_pos": 250,
                        "scale": 2.0,
                        "rotation": 0,
                        "opacity": 1.0
                    }

                st.markdown(f"### Positioning for {media_key}")
                col_preview, col_controls = st.columns([3, 2])

                with col_controls:
                    st.markdown("**Adjust Logo Settings**")
                    def update_position():
                        logging.info(
                            f"Slider updated for {media_key}: "
                            f"x_pos={st.session_state.logo_positions[media_key]['x_pos']}, "
                            f"y_pos={st.session_state.logo_positions[media_key]['y_pos']}, "
                            f"scale={st.session_state.logo_positions[media_key]['scale']}, "
                            f"rotation={st.session_state.logo_positions[media_key]['rotation']}, "
                            f"opacity={st.session_state.logo_positions[media_key]['opacity']}"
                        )
                        st.rerun()

                    x_pos = st.slider(
                        "X Position",
                        0,
                        1000,
                        st.session_state.logo_positions[media_key]["x_pos"],
                        key=f"x_pos_{safe_media_key}",
                        on_change=update_position
                    )
                    y_pos = st.slider(
                        "Y Position",
                        0,
                        1000,
                        st.session_state.logo_positions[media_key]["y_pos"],
                        key=f"y_pos_{safe_media_key}",
                        on_change=update_position
                    )
                    scale = st.slider(
                        "Scale",
                        0.5,
                        10.0,
                        st.session_state.logo_positions[media_key]["scale"],
                        step=0.1,
                        key=f"scale_{safe_media_key}",
                        on_change=update_position
                    )
                    rotation = st.slider(
                        "Rotation (degrees)",
                        -180,
                        180,
                        st.session_state.logo_positions[media_key]["rotation"],
                        step=1,
                        key=f"rotation_{safe_media_key}",
                        on_change=update_position
                    )
                    opacity = st.slider(
                        "Opacity (0.0 = transparent, 1.0 = opaque)",
                        0.0,
                        1.0,
                        st.session_state.logo_positions[media_key]["opacity"],
                        step=0.01,
                        key=f"opacity_{safe_media_key}",
                        on_change=update_position
                    )

                    st.session_state.logo_positions[media_key].update({
                        "x_pos": x_pos,
                        "y_pos": y_pos,
                        "scale": scale,
                        "rotation": rotation,
                        "opacity": opacity
                    })

                    st.markdown("**Click or Drag Logo to Position**")
                    click_position = st.text_input("Click/Drag Position (X, Y)", "", key=f"click_pos_{safe_media_key}", disabled=True)

                with col_preview:
                    media_path = os.path.join(Config.BASE_DIR, "Media", media_file.name)
                    try:
                        with open(media_path, "wb") as f:
                            f.write(media_file.getvalue())
                        logging.info(f"Saved media file to {media_path} for preview")
                    except Exception as e:
                        st.error(f"Failed to save media file {media_file.name} for preview: {str(e)}")
                        logging.error(f"Error saving media file to {media_path}: {str(e)}\n{traceback.format_exc()}")
                        continue

                    logging.info(f"Generating preview for {media_key} with x_pos={x_pos}, y_pos={y_pos}, scale={scale}, rotation={rotation}, opacity={opacity}")
                    preview_bytes = generate_preview_image(
                        media_path,
                        logo_path,
                        x_pos=x_pos,
                        y_pos=y_pos,
                        scale=scale,
                        opacity=opacity,
                        rotation=rotation
                    )
                    if preview_bytes:
                        st.image(preview_bytes, caption=f"Preview for {media_key}", use_container_width=True)
                        js_code = f"""
                        <script>
                        let isDragging_{safe_media_key} = false;
                        let lastUpdateTime_{safe_media_key} = 0;
                        const debounceDelay_{safe_media_key} = 100;

                        function startDrag_{safe_media_key}(event) {{
                            event.preventDefault();
                            isDragging_{safe_media_key} = true;
                            updatePosition_{safe_media_key}(event);
                        }}

                        function drag_{safe_media_key}(event) {{
                            if (isDragging_{safe_media_key}) {{
                                const now = Date.now();
                                if (now - lastUpdateTime_{safe_media_key} > debounceDelay_{safe_media_key}) {{
                                    updatePosition_{safe_media_key}(event);
                                    lastUpdateTime_{safe_media_key} = now;
                                }}
                            }}
                        }}

                        function stopDrag_{safe_media_key}() {{
                            isDragging_{safe_media_key} = false;
                        }}

                        function updatePosition_{safe_media_key}(event) {{
                            const img = document.getElementById('preview_{safe_media_key}');
                            const rect = img.getBoundingClientRect();
                            const x = event.clientX - rect.left;
                            const y = event.clientY - rect.top;
                            const scaleX = 1000 / rect.width;
                            const scaleY = 1000 / rect.height;
                            const scaledX = Math.round(Math.max(0, Math.min(1000, x * scaleX)));
                            const scaledY = Math.round(Math.max(0, Math.min(1000, y * scaleY)));
                            const input = document.querySelector('[data-click-pos="{safe_media_key}"]');
                            if (input) {{
                                input.value = '(' + scaledX + ', ' + scaledY + ')';
                                window.Streamlit.setComponentValue('x_pos_{safe_media_key}', scaledX);
                                window.Streamlit.setComponentValue('y_pos_{safe_media_key}', scaledY);
                            }}
                        }}

                        document.addEventListener('mouseup', stopDrag_{safe_media_key});
                        </script>
                        <img id="preview_{safe_media_key}" 
                            src="data:image/png;base64,{base64.b64encode(preview_bytes).decode('utf-8')}" 
                            onmousedown="startDrag_{safe_media_key}(event)"
                            onmousemove="drag_{safe_media_key}(event)"
                            onmouseup="stopDrag_{safe_media_key}()"
                            data-click-pos="{safe_media_key}"
                            style="cursor: move; max-width: 100%;">
                        """
                        st.markdown(js_code, unsafe_allow_html=True)
                    else:
                        st.warning(f"Failed to generate preview for {media_key}. Please check file formats or try again.")

                custom_positions.append({
                    "media_key": media_key,
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "scale": scale,
                    "rotation": rotation,
                    "opacity": opacity
                })

    st.header("Face Blurring")
    blur_enabled = st.checkbox(
        "Enable Face Blurring",
        value=st.session_state.get('blur_enabled', False),
        disabled=not State.blur_enabled or State.face_detector is None or State.face_mesh is None or State.yolo_model is None or State.tracker is None or State.dnn_net is None,
        key="blur_enabled"
    )
    if blur_enabled != st.session_state.get('blur_enabled'):
        st.session_state.blur_enabled = blur_enabled
    if not State.blur_enabled:
        st.warning("Face blurring is disabled for your license.")
    elif State.face_detector is None or State.face_mesh is None or State.yolo_model is None or State.tracker is None or State.dnn_net is None:
        st.warning("AI models not loaded. Face blurring disabled.")
        st.session_state.blur_enabled = False

    if st.button("Process Files") and logo_file and media_files:
        st.session_state.output_files.clear()

        if check_license(st.session_state.user_id):
            logo_path = os.path.join(Config.BASE_DIR, "Logos", logo_file.name)
            try:
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getvalue())
                logging.info(f"Saved logo file to {logo_path}")
            except Exception as e:
                st.error(f"Failed to save logo file: {str(e)}")
                logging.error(f"Error saving logo file to {logo_path}: {str(e)}\n{traceback.format_exc()}")
                return

            for media_file in media_files:
                media_path = os.path.join(Config.BASE_DIR, "Media", media_file.name)
                try:
                    with open(media_path, "wb") as f:
                        f.write(media_file.getvalue())
                    logging.info(f"Saved media file to {media_path}")
                except Exception as e:
                    st.error(f"Failed to save media file {media_file.name}: {str(e)}")
                    logging.error(f"Error saving media file to {media_path}: {str(e)}\n{traceback.format_exc()}")
                    continue
                output_filename = f"logoed_{media_file.name}"
                if media_file.name.lower().endswith(".mov"):
                    output_filename = f"logoed_{os.path.splitext(media_file.name)[0]}.mp4"
                output_path = os.path.join(Config.BASE_DIR, "Logoed_Media", output_filename)
                media_type = "image" if media_file.name.lower().endswith((".jpg", ".jpeg", ".png")) else "video"
                media_key = media_file.name

                try:
                    # Apply face blurring if enabled
                    blurred_regions = []
                    temp_media_path = media_path
                    if st.session_state.get('blur_enabled', False):
                        if media_type == "image":
                            image = Image.open(media_path).convert("RGBA")
                            processed_image, blurred_regions = process_image(image, State.dnn_net, True)
                            if blurred_regions:
                                approved = review_blurred_regions(blurred_regions, media_type, Config.BASE_DIR, media_file.name)
                                if not approved:
                                    st.error(f"Blurring not approved for {media_file.name}. Skipping processing.")
                                    logging.info(f"Blurring not approved for {media_file.name}")
                                    continue
                            temp_media_path = os.path.join(Config.BASE_DIR, "Media", f"blurred_{media_file.name}")
                            processed_image.save(temp_media_path, "PNG")
                            logging.info(f"Applied face blurring to image {media_file.name}, saved to {temp_media_path}")
                        else:
                            temp_media_path = os.path.join(Config.BASE_DIR, "Media", f"blurred_{media_file.name}")
                            blurred_regions = process_video(
                                media_path,
                                temp_media_path,
                                State.face_detector,
                                State.face_mesh,
                                State.yolo_model,
                                State.tracker,
                                True
                            )
                            if blurred_regions:
                                approved = review_blurred_regions(blurred_regions, media_type, Config.BASE_DIR, media_file.name)
                                if not approved:
                                    st.error(f"Blurring not approved for {media_file.name}. Skipping processing.")
                                    logging.info(f"Blurring not approved for {media_file.name}")
                                    continue
                            logging.info(f"Applied face blurring to video {media_file.name}, saved to {temp_media_path}")

                    # Apply logo
                    position = st.session_state.selected_position
                    if st.session_state.manual_positioning and media_key in st.session_state.logo_positions:
                        position = "manual"
                    else:
                        position = st.session_state.selected_position

                    if media_type == "image":
                        image = Image.open(temp_media_path).convert("RGBA")
                        processed_image = overlay_logo_on_image(
                            image,
                            logo_path,
                            position=position,
                            media_key=media_key
                        )
                        processed_image.save(output_path, "PNG")
                        logging.info(f"Processed image saved to {output_path}")
                    else:
                        overlay_logo_on_video(
                            temp_media_path,
                            logo_path,
                            output_path,
                            position=position,
                            media_key=media_key
                        )
                        logging.info(f"Processed video saved to {output_path}")

                    # Read output file for download
                    with open(output_path, "rb") as f:
                        output_data = f.read()
                    st.session_state.output_files.append((output_path, output_filename, output_data))
                    increment_execution_count(st.session_state.user_id, media_file.name)
                    st.success(f"Processed {media_file.name} successfully!")
                except Exception as e:
                    st.error(f"Error processing {media_file.name}: {str(e)}")
                    logging.error(f"Error processing {media_file.name}: {str(e)}\n{traceback.format_exc()}")

    if st.session_state.output_files:
        st.header("Download Processed Files")
        for file_path, file_name, file_data in st.session_state.output_files:
            mime_type = "image/png" if file_name.lower().endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
            st.download_button(
                label=f"Download {file_name}",
                data=file_data,
                file_name=file_name,
                mime=mime_type,
                key=f"download_{file_name}"
            )

if __name__ == "__main__":
    main()