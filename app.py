import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth
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

# Configuration
class Config:
    LOGO_SIZE_PERCENT = 0.5
    LOGO_TRANSPARENCY = 0.5  # Set to 50% transparency
    LOGO_OFFSET_PERCENT = 0.05
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
        audio = original_clip.audio
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
            bitrate='5000k'
        )
        video_clip.close()
        original_clip.close()
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logging.info(f"Removed temporary file: {temp_output_path}")
        logging.info(f"Video saved with blur: {output_path}, {len(blurred_regions)} regions blurred")
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
        img_base64 = base64.b64encode(img_buffer.getbuffer()).decode('utf-8')
        img_src = f"data:image/png;base64,{img_base64}"

        transparency = st.slider(
            "Logo Transparency (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key=f"{key_prefix}_transparency"
        )
        st.session_state[f"{key_prefix}_transparency"] = transparency / 100.0

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
            <img src="%s" id="%s" onclick="updatePosition(event, '%s')" style="cursor: crosshair; max-width: 100%%;" />
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
def overlay_logo_on_image(image, logo_path, position="center"):
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            logging.info("Image converted to RGBA")
        logo = Image.open(logo_path).convert("RGBA")
        img_width, img_height = image.size
        max_logo_size = int(min(img_width, img_height) * Config.LOGO_SIZE_PERCENT)
        logo.thumbnail((max_logo_size, max_logo_size), Image.Resampling.LANCZOS)
        logo_array = np.array(logo)

        transparency = st.session_state.get('logo_position_transparency', Config.LOGO_TRANSPARENCY)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * transparency).astype(np.uint8)
        logo = Image.fromarray(logo_array)
        offset = int(min(img_width, img_height) * Config.LOGO_OFFSET_PERCENT)

        if 'custom_position' in st.session_state and st.session_state.custom_position:
            x, y = st.session_state.custom_position['x'], st.session_state.custom_position['y']
            logging.info(f"Using custom logo position: ({x}, {y}) with transparency {transparency*100}%")
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
            x, y = position_map.get(position, position_map["center"])

        x, y = max(0, x), max(0, y)
        if x + logo.size[0] > img_width:
            x = img_width - logo.size[0]
        if y + logo.size[1] > img_height:
            y = img_height - logo.size[1]
        output = Image.new("RGBA", image.size)
        output.paste(image, (0, 0))
        output.paste(logo, (x, y), logo)
        logging.info(f"Logo overlaid on image at ({x}, {y}) with transparency {transparency*100}%")
        return output
    except Exception as e:
        logging.error(f"Error overlaying logo on image: {str(e)}")
        return image

# Overlay logo on video
def overlay_logo_on_video(video_path, logo_path, output_path, position="center"):
    try:
        video = VideoFileClip(video_path)
        logo = Image.open(logo_path).convert("RGBA")
        vid_width, vid_height = video.size
        max_logo_size = int(min(vid_width, vid_height) * Config.LOGO_SIZE_PERCENT)
        logo.thumbnail((max_logo_size, max_logo_size), Image.Resampling.LANCZOS)
        logo_array = np.array(logo)

        transparency = st.session_state.get('logo_position_transparency', Config.LOGO_TRANSPARENCY)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * transparency).astype(np.uint8)
        logo = Image.fromarray(logo_array)

        temp_logo_path = f"temp_logo_{uuid.uuid4()}.png"
        logo.save(temp_logo_path, "PNG")
        logo_clip = ImageClip(temp_logo_path).set_duration(video.duration)
        offset = int(min(vid_width, vid_height) * Config.LOGO_OFFSET_PERCENT)

        if 'custom_position' in st.session_state and st.session_state.custom_position:
            x, y = st.session_state.custom_position['x'], st.session_state.custom_position['y']
            logging.info(f"Using custom logo position for video: ({x}, {y}) with transparency {transparency*100}%")
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
            x, y = position_map.get(position, position_map["center"])

        x, y = max(0, x), max(0, y)
        if x + logo.size[0] > vid_width:
            x = vid_width - logo.size[0]
        if y + logo.size[1] > vid_height:
            y = vid_height - logo.size[1]
        logo_clip = logo_clip.set_position((x, y))
        final_clip = CompositeVideoClip([video, logo_clip])
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio=video.audio is not None,
            audio_codec="aac" if video.audio is not None else None,
            fps=video.fps,
            preset="medium",
            bitrate="5000k"
        )
        video.close()
        final_clip.close()
        if os.path.exists(temp_logo_path):
            os.remove(temp_logo_path)
        logging.info(f"Video saved with logo to {output_path} with transparency {transparency*100}%")
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        shutil.copy(video_path, output_path)

# Generate preview image
def generate_preview_image(media_file, logo_path):
    try:
        media_type = "image" if media_file.name.lower().endswith((".jpg", ".jpeg", ".png")) else "video"
        if media_type == "image":
            image = Image.open(media_file).convert("RGBA")
            preview_image = overlay_logo_on_image(image, logo_path)
        else:
            video = VideoFileClip(media_file.name)
            frame = video.get_frame(0)
            video.close()
            image = Image.fromarray(frame).convert("RGBA")
            preview_image = overlay_logo_on_image(image, logo_path)
        buffer = io.BytesIO()
        preview_image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"Error generating preview for {media_file.name}: {str(e)}")
        return None

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
        st.warning("Firestore unavailable. Using local execution count (temporary).")
        if not hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = 0
        State.execution_count = st.session_state.local_execution_count
        State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
        State.infinite_count = False
        State.blur_enabled = True
        State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
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
            State.license_expiry = data.get("expiry", datetime.now(timezone.utc))
            State.subscription_expiry = data.get("subscription_expiry", datetime.now(timezone.utc))
            if State.license_expiry.tzinfo is None:
                State.license_expiry = State.license_expiry.replace(tzinfo=timezone.utc)
            if State.subscription_expiry.tzinfo is None:
                State.subscription_expiry = State.subscription_expiry.replace(tzinfo=timezone.utc)
            logging.info(f"License checked for user {user_id}: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, blur={State.blur_enabled}, expiry={State.license_expiry}, sub_expiry={State.subscription_expiry}")
        else:
            State.execution_count = 0
            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
            State.infinite_count = False
            State.blur_enabled = True
            State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            doc_ref.set({
                "user_id": user_id,
                "count": 0,
                "max_executions": Config.DEFAULT_MAX_EXECUTIONS,
                "infinite_count": False,
                "blur_enabled": True,
                "expiry": State.license_expiry,
                "subscription_expiry": State.subscription_expiry,
                "created_at": datetime.now(timezone.utc)
            })
            logging.info(f"New license created for user {user_id}: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, blur={State.blur_enabled}, expiry={State.license_expiry}, sub_expiry={State.subscription_expiry}")

        if State.infinite_count:
            logging.info(f"User {user_id} has infinite executions.")
            return True
        if datetime.now(timezone.utc) > State.license_expiry:
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
        logging.info(f"Admin user {user_id} bypasses execution count increment for file {file_name}.")
        return
    if not user_id:
        logging.warning(f"No user ID provided for execution count increment for {file_name}. Skipping.")
        return
    if db is None:
        logging.warning(f"Firestore unavailable, using local count for {file_name}.")
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
    if db is None:
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
            st.error(f"Start count ({new_count}) cannot exceed max executions ({max_executions}).")
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
            "created": datetime.now(timezone.utc)
        })
        execution_limit = f"Infinite count" if infinite_count else f"max_executions={max_executions}"
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
    if db is None:
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
        State.license_expiry = data.get("expiry")
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
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if "idToken" in data:
            user_id = data.get("localId")
            logging.info(f"User {user_id} authenticated successfully")
            return user_id, None
        else:
            error_message = data.get("error", {}).get("message", "Invalid credentials")
            logging.error(f"Authentication failed: {error_message}")
            st.error(f"Authentication failed: {error_message}")
            return None, error_message
    except Exception as e:
        logging.error(f"Error verifying user: {str(e)}")
        st.error(f"Error verifying credentials: {str(e)}")
        return None, str(e)

# Debug tool to manage license limits
def debug_license_management(user_id):
    if not user_id:
        logging.warning("No user ID provided for debug license management.")
        st.error("No user ID provided for debug.")
        return

    st.subheader("Debug License Management")
    st.write(f"Firestore status: {'Connected' if db else 'Disconnected'}")

    if db is None:
        st.error("Firestore unavailable for debug mode.")
        return

    target_user_id = st.text_input("Target User ID", value=user_id, key="debug_user_id")
    if not target_user_id:
        return

    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(target_user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            current_count = data.get("count", 0)
            current_max = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
            current_infinite = data.get("infinite_count", False)
            current_blur = data.get("blur_enabled", True)
            current_expiry = data.get("expiry", datetime.now(timezone.utc))
            current_sub_expiry = data.get("subscription_expiry", datetime.now(timezone.utc))
            if current_expiry.tzinfo is None:
                current_expiry = current_expiry.replace(tzinfo=timezone.utc)
            if current_sub_expiry.tzinfo is None:
                current_sub_expiry = current_sub_expiry.replace(tzinfo=timezone.utc)
            st.write(f"Count: {current_count}")
            st.write(f"Max Executions: {current_max}")
            st.write(f"Infinite Count: {current_infinite}")
            st.write(f"Blur Enabled: {current_blur}")
            st.write(f"Expiry: {current_expiry}")
            st.write(f"Subscription Expiry: {current_sub_expiry}")
        else:
            st.warning(f"No license data for user {target_user_id}")
            current_count = 0
            current_max = Config.DEFAULT_MAX_EXECUTIONS
            current_infinite = False
            current_blur = True
            current_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            current_sub_expiry = datetime.now(timezone.utc) + timedelta(days=30)

        col1, col2, col3 = st.columns(3)
        with col1:
            new_count = st.number_input("Set Count", min_value=0, value=current_count, key="debug_count")
            if st.button("Apply Count", key="apply_count"):
                doc_ref.update({"count": new_count})
                if target_user_id == user_id:
                    State.execution_count = new_count
                st.success(f"Set count to {new_count}")
                logging.info(f"Debug: Set count to {new_count} for {target_user_id}")

            new_max = st.number_input("Set Max", min_value=0, value=current_max, key="debug_max")
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
                    st.success(f"Set max executions to {new_max} ({'infinite' if new_max == 0 else 'limited'})")
                    logging.info(f"Debug: Set max to {new_max}, infinite={new_max == 0} for {target_user_id}")
        with col2:
            expiry_days = st.number_input("Expiry Days", min_value=1, value=30, key="debug_expiry_days")
            if st.button("Apply Expiry", key="apply_expiry"):
                new_expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
                doc_ref.update({"expiry": new_expiry})
                if target_user_id == user_id:
                    State.license_expiry = new_expiry
                st.success(f"Set expiry to {new_expiry}")
                logging.info(f"Debug: Set expiry to {new_expiry} for {target_user_id}")

            sub_days = st.number_input("Sub Days", min_value=1, value=30, key="debug_sub_days")
            if st.button("Apply Sub Expiry", key="apply_sub_expiry"):
                new_sub_expiry = datetime.now(timezone.utc) + timedelta(days=sub_days)
                doc_ref.update({"subscription_expiry": new_sub_expiry})
                if target_user_id == user_id:
                    State.subscription_expiry = new_sub_expiry
                st.success(f"Set subscription expiry to {new_sub_expiry}")
                logging.info(f"Debug: Set sub_expiry to {new_sub_expiry} for {target_user_id}")
        with col3:
            blur_enabled = st.checkbox("Enable Face Blur", value=current_blur, key="debug_blur")
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
                st.success(f"Set infinite count to {infinite}")
                logging.info(f"Debug: Set infinite={infinite}, count={0 if infinite else current_count}, max={0 if infinite else Config.DEFAULT_MAX_EXECUTIONS} for {target_user_id}")
    except Exception as e:
        logging.error(f"Error in debug mode for {target_user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error in debug: {str(e)}")
        
# Main application
def main():
    st.title("Logo Adder Application")
    ensure_directories(Config.BASE_DIR)
    initialize_ai_models()

    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
        st.session_state['custom_position'] = None
        st.session_state['local_execution_count'] = None

    user_id = st.session_state.user_id
    if not user_id:
        st.subheader("Login")
        email = st.text_input("Email Address", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user_id, error = verify_user(email, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
                logging.info(f"User {user_id} logged in")
                st.rerun()
            else:
                st.error(f"Login failed: {error}")
        st.subheader("Apply License Patch")
        patch_id = st.text_input("Patch ID", key="patch_input")
        if st.button("Apply Patch"):
            if validate_patch(patch_id, user_id):
                st.rerun()
        return

    if not check_license(user_id):
        st.error("License check failed. Please apply a new patch or contact support.")
        return

    st.subheader("Upload Files")
    logo_file = st.file_uploader("Upload Logo (PNG)", ["png", "jpg", "jpeg"], key="logo_upload")
    media_files = st.file_uploader("Upload Media (JPEG, PNG, MP4)", ["jpg", "jpeg", "png", "mp4"], key="media_upload", accept_multiple_files=True)
    blur_enabled = st.checkbox("Enable Face Blurring", value=True, key="blur_toggle")
    position = st.selectbox("Logo Position", [
        "top", "center", "bottom", "left", "right", "top_left",
        "top_right", "left_center", "right_center", "left_bottom", "right_bottom"
    ], key="position_select")
    use_manual_position = st.checkbox("Manually Position Logo", False, key="manual_position")

    if use_manual_position and media_files:
        st.subheader("Set Logo Position")
        media_file = media_files[0]
        coords = logo_position_component(media_file, key_prefix="custom_position")
        if coords and coords.get("value"):
            st.session_state['custom_position'] = coords["value"]
            st.write(f"Custom position set to: {st.session_state['custom_position']}")
            logging.info(f"Custom position set: {st.session_state['custom_position']}")

    if st.button("Process"):
        if not logo_file or not media_files:
            st.error("Please upload both a logo and at least one media file.")
            return

        logo_path = os.path.join(Config.BASE_DIR, "Logos", logo_file.name)
        with open(logo_path, "wb") as f:
            f.write(logo_file.getbuffer())

        output_files = []
        for media_file in media_files:
            try:
                media_path = os.path.join(Config.BASE_DIR, "Media", media_file.name)
                with open(media_path, "wb") as f:
                    f.write(media_file.getbuffer())

                media_type = "image" if media_file.name.lower().endswith((".jpg", ".jpeg", ".png")) else "video"
                output_name = f"logoed_{media_file.name}"
                output_path = os.path.join(Config.BASE_DIR, "Logoed_Media", output_name)

                blurred_regions = []
                if blur_enabled and State.blur_enabled:
                    if media_type == "image":
                        image = Image.open(media_path).convert("RGBA")
                        processed_image, blurred_regions = process_image(image, State.dnn_net, blur_enabled)
                        processed_image.save(media_path, "PNG")
                    else:
                        blurred_regions = process_video(media_path, output_path, State.face_detector, State.face_mesh, State.yolo_model, State.tracker, blur_enabled)

                if blurred_regions:
                    approved = review_blurred_regions(blurred_regions, media_type, Config.BASE_DIR, media_file.name)
                    if not approved:
                        st.warning(f"Processing aborted for {media_file.name} due to unapproved blur regions.")
                        continue

                if media_type == "image":
                    image = Image.open(media_path).convert("RGBA")
                    final_image = overlay_logo_on_image(image, logo_path, position=position)
                    final_image.save(output_path, "PNG")
                else:
                    overlay_logo_on_video(media_path, logo_path, output_path, position=position)

                with open(output_path, "rb") as f:
                    output_files.append((output_path, f.read()))

                increment_execution_count(user_id, media_file.name)
                st.success(f"Processed {media_file.name} successfully!")
                logging.info(f"Processed {media_file.name} for user {user_id}, output={output_path}")

            except Exception as e:
                logging.error(f"Error processing {media_file.name}: {str(e)}\n{traceback.format_exc()}")
                st.error(f"Error processing {media_file.name}: {str(e)}")

        if output_files:
            if Config.USE_JAVASCRIPT_DOWNLOAD:
                trigger_multiple_downloads(output_files)
            else:
                for file_path, file_data in output_files:
                    st.download_button(
                        f"Download {os.path.basename(file_path)}",
                        file_data,
                        file_name=os.path.basename(file_path)
                    )

    if user_id == Config.ADMIN_USER_ID:
        st.subheader("Admin - Generate Patch")
        target_user_id = st.text_input("Target User ID", key="admin_user_id")
        new_count = st.number_input("New Execution Count", min_value=0, value=0, key="admin_count")
        days_valid = st.number_input("Days Valid", min_value=1, value=30, key="admin_days")
        subscription_days = st.number_input("Subscription Days Valid", min_value=1, value=30, key="admin_sub_days")
        max_executions = st.number_input("Max Executions", min_value=0, value=Config.DEFAULT_MAX_EXECUTIONS, key="admin_max")
        blur_enabled_admin = st.checkbox("Enable Blur", value=True, key="admin_blur")
        if st.button("Generate Patch"):
            patch_id = apply_patch(
                target_user_id,
                new_count,
                days_valid,
                subscription_days,
                max_executions,
                blur_enabled_admin
            )
            if patch_id:
                st.success(f"Patch ID: {patch_id}")
                logging.info(f"Admin {user_id} generated patch {patch_id} for {target_user_id}")

        st.subheader("Admin - Debug License")
        debug_license_management(user_id)

if __name__ == "__main__":
    main()