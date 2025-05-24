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
    if not firebase_admin._apps:
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
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            logging.info("Firebase Admin SDK and Firestore initialized successfully")
        else:
            logging.error("No valid Firebase credentials provided.")
            st.error("Firebase credentials missing. Contact support.")
    else:
        db = firestore.client()
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
    LOGO_TRANSPARENCY = 0.7
    LOGO_OFFSET_PERCENT = 0.05
    DEFAULT_MAX_EXECUTIONS = 27
    EXECUTION_COLLECTION = "executions"
    LICENSE_COLLECTION = "licenses"
    FOLDERS = ["Logos", "Media", "Logoed_Media", "Blur_Preview"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIDENCE_THRESHOLD = 0.2
    BLUR_KERNEL_FACTOR = 0.15
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
    blur_enabled = True  # Default to True
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

# Process frame for blurring faces (video, MediaPipe-based)
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

# Process image for blurring (DNN-based)
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
            st.image(frame_rgb, caption=f"Frame {idx} Preview", use_container_width=True)
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

# Overlay logo on image
def overlay_logo_on_image(image, logo_path, position="center"):
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            logging.info("Converted image to RGBA")
        logo = Image.open(logo_path).convert("RGBA")
        img_width, img_height = image.size
        max_logo_size = int(min(img_width, img_height) * Config.LOGO_SIZE_PERCENT)
        logo.thumbnail((max_logo_size, max_logo_size), Image.Resampling.LANCZOS)
        logo_array = np.array(logo)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * Config.LOGO_TRANSPARENCY).astype(np.uint8)
        logo = Image.fromarray(logo_array)
        offset = int(min(img_width, img_height) * Config.LOGO_OFFSET_PERCENT)
        position_map = {
            "top": ((img_width - logo.size[0]) // 2, offset),
            "bottom": ((img_width - logo.size[0]) // 2, img_height - logo.size[1] - offset),
            "left": (offset, (img_height - logo.size[1]) // 2),
            "right": (img_width - logo.size[0] - offset, (img_height - logo.size[1]) // 2),
            "top_left": (offset, offset),
            "top_right": (img_width - logo.size[0] - offset, offset),
            "left_center": (offset, (img_height - logo.size[1]) // 2),
            "right_center": (img_width - logo.size[0] - offset, (img_height - logo.size[1]) // 2),
            "left_bottom": (offset, img_height - logo.size[1] - offset),
            "right_bottom": (img_width - logo.size[0] - offset, img_height - logo.size[1] - offset),
            "center": ((img_width - logo.size[0]) // 2, (img_height - logo.size[1]) // 2)
        }
        x, y = position_map.get(position, position_map["center"])
        x, y = max(0, x), max(0, y)
        if x + logo.size[0] > img_width or y + logo.size[1] > img_height:
            logging.warning(f"Logo position ({x}, {y}) adjusted to fit image")
            x = min(x, img_width - logo.size[0])
            y = min(y, img_height - logo.size[1])
        output = Image.new("RGBA", image.size)
        output.paste(image, (0, 0))
        output.paste(logo, (x, y), logo)
        logging.info(f"Logo overlaid at position ({x}, {y})")
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
        logo_array[:, :, 3] = (logo_array[:, :, 3] * Config.LOGO_TRANSPARENCY).astype(np.uint8)
        logo = Image.fromarray(logo_array)
        temp_logo_path = f"temp_logo_{uuid.uuid4()}.png"
        logo.save(temp_logo_path, "PNG")
        logo_clip = ImageClip(temp_logo_path).set_duration(video.duration)
        offset = int(min(vid_width, vid_height) * Config.LOGO_OFFSET_PERCENT)
        position_map = {
            "top": ((vid_width - logo.size[0]) // 2, offset),
            "bottom": ((vid_width - logo.size[0]) // 2, vid_height - logo.size[1] - offset),
            "left": (offset, (vid_height - logo.size[1]) // 2),
            "right": (vid_width - logo.size[0] - offset, (vid_height - logo.size[1]) // 2),
            "top_left": (offset, offset),
            "top_right": (vid_width - logo.size[0] - offset, offset),
            "left_center": (offset, (vid_height - logo.size[1]) // 2),
            "right_center": (vid_width - logo.size[0] - offset, (vid_height - logo.size[1]) // 2),
            "left_bottom": (offset, vid_height - logo.size[1] - offset),
            "right_bottom": (vid_width - logo.size[0] - offset, vid_height - logo.size[1] - offset),
            "center": ((vid_width - logo.size[0]) // 2, (vid_height - logo.size[1]) // 2)
        }
        x, y = position_map.get(position, position_map["center"])
        logo_clip = logo_clip.set_position((x, y))
        final_clip = CompositeVideoClip([video, logo_clip])
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio=video.audio is not None,
            audio_codec="aac" if video.audio else None,
            fps=video.fps,
            preset="fast",
            bitrate="5000k"
        )
        video.close()
        final_clip.close()
        os.remove(temp_logo_path)
        logging.info(f"Video saved with logo to {output_path}")
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise

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
            st.error("Execution limit reached. Contact the service team for a new patch.")
            return False
        if datetime.now(timezone.utc) > State.subscription_expiry:
            st.error("Subscription expired. Contact the service team for a new patch.")
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
            logging.info(f"License checked for user {user_id}: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, blur_enabled={State.blur_enabled}, expiry={State.license_expiry}, subscription_expiry={State.subscription_expiry}")
        else:
            State.execution_count = 0
            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
            State.infinite_count = False
            State.blur_enabled = True
            State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            doc_ref.set({
                "user_id": user_id,
                "count": State.execution_count,
                "max_executions": State.max_executions,
                "infinite_count": State.infinite_count,
                "blur_enabled": State.blur_enabled,
                "expiry": State.license_expiry,
                "subscription_expiry": State.subscription_expiry,
                "created_at": datetime.now(timezone.utc)
            })
            logging.info(f"New license created for user {user_id}: count=0, max={State.max_executions}, infinite={State.infinite_count}, blur_enabled={State.blur_enabled}, expiry={State.license_expiry}, subscription_expiry={State.subscription_expiry}")
        
        try:
            expiry = State.license_expiry
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                st.error("License expired. Contact the service team for a new patch.")
                return False
            
            sub_expiry = State.subscription_expiry
            if sub_expiry.tzinfo is None:
                sub_expiry = sub_expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > sub_expiry:
                st.error("Subscription expired. Contact the service team for a new patch.")
                return False
                
            if State.infinite_count and (State.max_executions == 0 and State.execution_count == 0):
                logging.info(f"User {user_id} has valid infinite count license.")
                return True
            if State.execution_count >= State.max_executions:
                st.error("Execution limit reached. Contact the service team for a new patch.")
                return False
            return True
        except TypeError as e:
            logging.error(f"Datetime comparison error for user {user_id}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error checking license/subscription expiry: {str(e)}. Using local count temporarily.")
            State.execution_count = getattr(st.session_state, 'local_execution_count', 0)
            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
            State.infinite_count = False
            State.blur_enabled = True
            State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            if State.execution_count >= State.max_executions:
                st.error("Execution limit reached. Contact the service team for a new patch.")
                return False
            return True
    except Exception as e:
        if "PERMISSION_DENIED" in str(e).upper():
            logging.error(f"Firestore permission denied for user {user_id}: {str(e)}\n{traceback.format_exc()}")
            st.error("Firestore access denied. Check Firebase security rules or contact support.")
        else:
            logging.error(f"License check failed for user {user_id}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error checking license: {str(e)}. Contact the service team.")
        return False

# Increment execution count
def increment_execution(user_id, file_name):
    if user_id == Config.ADMIN_USER_ID:
        logging.info(f"Admin user {user_id} bypasses execution count increment for file {file_name}.")
        return
    if not user_id:
        logging.warning(f"No user_id for execution count increment for file {file_name}. Skipping.")
        State.execution_count += 1
        return
    if db is None:
        logging.warning(f"Firestore unavailable for increment, using local count for file {file_name}.")
        if not hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = 0
        st.session_state.local_execution_count += 1
        State.execution_count = st.session_state.local_execution_count
        logging.info(f"Local execution count updated to {State.execution_count} for user {user_id}, file {file_name}")
        return
    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            infinite_count = data.get("infinite_count", False)
            max_executions = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
            current_count = data.get("count", 0)
            if infinite_count and max_executions == 0 and current_count == 0:
                logging.info(f"User {user_id} has infinite count, skipping increment for file {file_name}.")
                return
        doc_ref.update({
            "count": firestore.Increment(1),
            "last_updated": datetime.now(timezone.utc),
            "last_file": file_name
        })
        State.execution_count += 1
        logging.info(f"Execution count updated to {State.execution_count} for user {user_id}, file {file_name}")
    except Exception as e:
        if "PERMISSION_DENIED" in str(e).upper():
            logging.error(f"Firestore permission denied incrementing count for user {user_id}, file {file_name}: {str(e)}\n{traceback.format_exc()}")
            st.error("Firestore access denied. Contact support.")
        else:
            logging.error(f"Error incrementing execution count for user {user_id}, file {file_name}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error updating execution count for {file_name}. Contact support.")

# Apply patch (admin function)
def apply_patch(user_id, new_count, days_valid, subscription_days_valid, max_executions=None, blur_enabled=True):
    if db is None:
        logging.error("Firestore client not initialized. Cannot generate patch.")
        st.error("Firestore unavailable. Cannot generate patch.")
        return None
    try:
        patch_id = str(uuid.uuid4())
        doc_ref = db.collection(Config.LICENSE_COLLECTION).document(patch_id)
        expiry = datetime.now(timezone.utc) + timedelta(days=days_valid)
        subscription_expiry = datetime.now(timezone.utc) + timedelta(days=subscription_days_valid)
        infinite_count = max_executions == 0
        if max_executions is None:
            max_executions = Config.DEFAULT_MAX_EXECUTIONS
        if max_executions > 0 and new_count > max_executions:
            st.error(f"Start Execution Count ({new_count}) cannot exceed Max Executions ({max_executions}).")
            logging.error(f"Invalid patch parameters: new_count={new_count} > max_executions={max_executions}")
            return None
        doc_ref.set({
            "user_id": user_id,
            "new_count": new_count,
            "infinite_count": infinite_count,
            "max_executions": max_executions,
            "blur_enabled": blur_enabled,
            "expiry": expiry,
            "subscription_expiry": subscription_expiry,
            "used": False,
            "created_at": datetime.now(timezone.utc)
        })
        execution_limit_msg = "Infinite executions" if infinite_count else f"Max executions={max_executions}"
        st.success(f"Patch ID: {patch_id} (Valid for {days_valid} days, Subscription valid for {subscription_days_valid} days, Start count={new_count}, {execution_limit_msg}, Blur enabled={blur_enabled})")
        logging.info(f"Patch generated: {patch_id} for user {user_id}, count={new_count}, max_executions={max_executions}, infinite={infinite_count}, blur_enabled={blur_enabled}, expiry={expiry}, subscription_expiry={subscription_expiry}")
        return patch_id
    except Exception as e:
        logging.error(f"Error generating patch for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error("Error generating patch.")
        return None

# Validate and apply patch
def validate_patch(patch_id, user_id):
    if user_id == Config.ADMIN_USER_ID:
        logging.info(f"Admin user {user_id} does not require patch application.")
        st.info("Admin users do not need to apply patches.")
        return True
    if db is None:
        logging.error("Firestore client not initialized. Cannot apply patch.")
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
        if data["used"]:
            logging.error(f"Patch {patch_id} already used for user {user_id}.")
            st.error("Patch already used.")
            return False
        expiry = data["expiry"]
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            logging.error(f"Patch {patch_id} expired for user {user_id}.")
            st.error("Patch expired.")
            return False
        if data["user_id"] != user_id:
            logging.error(f"Patch {patch_id} not valid for user {user_id}.")
            st.error("Patch not valid for this user.")
            return False
        execution_ref = db.collection(Config.EXECUTION_COLLECTION).document(user_id)
        new_data = {
            "count": data["new_count"],
            "max_executions": data["max_executions"],
            "infinite_count": data["infinite_count"],
            "blur_enabled": data.get("blur_enabled", True),
            "expiry": data["expiry"],
            "subscription_expiry": data["subscription_expiry"],
            "last_updated": datetime.now(timezone.utc)
        }
        execution_ref.set(new_data, merge=True)
        logging.info(f"Execution data updated for user {user_id}: {new_data}")
        doc_ref.update({"used": True, "used_at": datetime.now(timezone.utc)})
        logging.info(f"Patch {patch_id} marked as used for user {user_id}.")
        State.execution_count = data["new_count"]
        State.max_executions = data["max_executions"]
        State.infinite_count = data["infinite_count"]
        State.blur_enabled = data.get("blur_enabled", True)
        State.license_expiry = data["expiry"]
        State.subscription_expiry = data["subscription_expiry"]
        if hasattr(st.session_state, 'local_execution_count'):
            st.session_state.local_execution_count = data["new_count"]
        st.session_state.patch_applied = True
        st.success(f"Patch {patch_id} applied successfully. Execution count set to {data['new_count']}, max executions set to {data['max_executions']}, blur enabled={data.get('blur_enabled', True)}.")
        logging.info(f"Patch applied: {patch_id} for user {user_id}, count={data['new_count']}, max_executions={data['max_executions']}, infinite={data['infinite_count']}, blur_enabled={data.get('blur_enabled', True)}, expiry={data['expiry']}, subscription_expiry={data['subscription_expiry']}")
        return True
    except Exception as e:
        logging.error(f"Error validating patch {patch_id} for user {user_id}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error applying patch: {str(e)}")
        return False

# JavaScript-based download
def trigger_multiple_downloads(files):
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
            files.forEach(file => {
                const link = document.createElement('a');
                link.href = file.url;
                link.download = file.name;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
        }
    </script>
    """
    files_json = json.dumps([{
        "url": f"data:application/octet-stream;base64,{base64.b64encode(file_data).decode('utf-8')}",
        "name": os.path.basename(file_path)
    } for file_path, _, file_data in files])
    st.markdown(js_code, unsafe_allow_html=True)
    st.markdown(f"""
    <button class="download-all-btn" onclick='downloadFiles({files_json})'>Download All Files</button>
    """, unsafe_allow_html=True)

# Verify user
def verify_user(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Firebase Auth response: {data}")
        if "idToken" in data:
            user_id = data.get("localId")
            logging.info(f"User authenticated: {user_id}")
            return user_id, None
        else:
            error_message = data.get("error", {}).get("message", "Invalid credentials")
            logging.error(f"Authentication failed: {error_message}")
            if "EMAIL_NOT_FOUND" in error_message:
                return None, "Email not registered."
            elif "INVALID_PASSWORD" in error_message:
                return None, "Incorrect password."
            elif "INVALID_LOGIN_CREDENTIALS" in error_message:
                return None, "Invalid email or password."
            else:
                return None, f"Authentication error: {error_message}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error verifying credentials: {str(e)}")
        return None, f"Error verifying credentials: {str(e)}"

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

# Streamlit app
def main():
    st.title("Logo Adder App")
    st.markdown("""
        <style>
        div.stButton > button[kind="primary"][id="start_logoing"] {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        div.stButton > button[kind="primary"][id="start_logoing"]:hover {
            background-color: #0056b3;
        }
        div.stButton > button[id^="download_all_btn"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        div.stButton > button[id^="download_all_btn"]:hover {
            background-color: #45a049;
        }
        div.stButton > button {
            margin: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.user_id = None
        st.session_state.auth_error = None
        st.session_state.reset_message = None
        st.session_state.logo_position = "center"
        st.session_state.logoed_files = []
        st.session_state.logo_file = None
        st.session_state.media_files = []
        st.session_state.processed_files_data = []
        st.session_state.download_all_index = None
        st.session_state.download_all_trigger = False
        st.session_state.blur_enabled = False
        st.session_state.device_id = str(uuid.uuid4())
        st.session_state.patch_applied = False
        st.session_state.blur_reviewed = {}
        logging.info(f"Initialized session state with device_id: {st.session_state.device_id}")

    logging.info(f"Session state at start: user={st.session_state.user}, user_id={st.session_state.user_id}, device_id={st.session_state.device_id}, patch_applied={st.session_state.patch_applied}")

    if not st.session_state.user:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Login"):
                if not email or not password:
                    st.session_state.auth_error = "Please enter both email and password."
                    logging.error("Login attempted without email or password")
                else:
                    uid, error = verify_user(email, password)
                    if uid:
                        st.session_state.user = uid
                        st.session_state.user_id = uid
                        st.session_state.auth_error = None
                        st.session_state.reset_message = None
                        logging.info(f"Login successful: user={uid}, setting session state")
                        st.success("Logged in successfully.")
                        st.rerun()
                    else:
                        st.session_state.auth_error = error
                        logging.error(f"Login failed: {error}")
        with col2:
            if st.button("Forgot Password?"):
                if not email:
                    st.session_state.reset_message = "Please enter your email to reset the password."
                    logging.error("Password reset attempted without email")
                else:
                    try:
                        auth.get_user_by_email(email)
                        reset_link = auth.generate_password_reset_link(email)
                        st.session_state.reset_message = f"Password reset link: {reset_link}"
                        st.session_state.auth_error = None
                        logging.info(f"Password reset link generated for {email}")
                    except auth.UserNotFoundError:
                        st.session_state.reset_message = "Email not registered."
                        logging.error(f"Password reset failed: Email {email} not registered")
                    except Exception as e:
                        st.session_state.reset_message = f"Error generating reset link: {str(e)}"
                        logging.error(f"Password reset error: {str(e)}")
        if st.session_state.auth_error:
            st.error(f"Login failed: {st.session_state.auth_error}")
        if st.session_state.reset_message:
            if "link" in st.session_state.reset_message:
                st.success(st.session_state.reset_message)
            else:
                st.error(st.session_state.reset_message)
        return

    if st.session_state.user and not st.session_state.user_id:
        st.session_state.user_id = st.session_state.user
        logging.info(f"Synced user_id from user: {st.session_state.user_id}")

    logging.info(f"Session state after login: user={st.session_state.user}, user_id={st.session_state.user_id}, device_id={st.session_state.device_id}, patch_applied={st.session_state.patch_applied}")

    license_valid = check_license(st.session_state.user_id, force_refresh=True)
    if st.session_state.user_id != Config.ADMIN_USER_ID and not st.session_state.patch_applied and not license_valid:
        st.subheader("Apply Patch")
        patch_id = st.text_input("Enter Patch ID")
        if st.button("Apply Patch"):
            if validate_patch(patch_id, st.session_state.user_id):
                st.session_state.patch_applied = True
                logging.info(f"Patch {patch_id} applied, forcing license refresh for user {st.session_state.user_id}")
                check_license(st.session_state.user_id, force_refresh=True)
                st.rerun()
        return

    if st.session_state.patch_applied and check_license(st.session_state.user_id, force_refresh=True):
        st.session_state.patch_applied = False
        logging.info(f"Patch applied and license check passed for user {st.session_state.user_id}, cleared patch_applied flag")

    if any(model is None for model in [State.face_detector, State.face_mesh, State.yolo_model, State.tracker, State.dnn_net]):
        initialize_ai_models()
        if any(model is None for model in [State.face_detector, State.face_mesh, State.yolo_model, State.tracker, State.dnn_net]):
            st.warning("AI models failed to load. Blurring functionality disabled.")

    if st.session_state.user_id == Config.ADMIN_USER_ID:
        debug_license_limits(st.session_state.user_id)

    st.subheader("Upload Files")
    base_path = "temp_files"
    ensure_directories(base_path)

    st.subheader("Upload Logo")
    logo_file = st.file_uploader("Upload Logo (PNG)", type=["png"], key="logo")
    if logo_file:
        temp_logo_path = os.path.join(base_path, "Logos", logo_file.name)
        with open(temp_logo_path, "wb") as f:
            f.write(logo_file.getbuffer())
        st.session_state.logo_file = logo_file
        st.success(f"Logo {logo_file.name} uploaded successfully.")

    media_files = st.file_uploader("Upload Media (Images/Videos)", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True, key="media")
    if State.blur_enabled:
        st.session_state.blur_enabled = st.checkbox("Enable Face Blurring", value=st.session_state.blur_enabled)
    

    if media_files:
        new_files = [f for f in media_files if f.name not in [mf.name for mf in st.session_state.media_files]]
        for media_file in new_files:
            increment_execution(st.session_state.user_id, media_file.name)
        st.session_state.media_files = media_files
        st.session_state.blur_reviewed = {}
        if new_files:
            st.rerun()

    st.subheader("Logo Position")
    col1, col2, col3, col4, col5 = st.columns(5)
    positions = ["top", "center", "bottom", "left", "right"]
    for col, pos in zip([col1, col2, col3, col4, col5], positions):
        with col:
            button_style = "background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; border: none; cursor: pointer;"
            if st.session_state.logo_position == pos:
                button_style = "background-color: #45a049; color: white; padding: 10px; border-radius: 5px; border: 2px solid #000;"
            if st.button(pos.capitalize(), key=f"pos_{pos}", help=f"Place logo at {pos}"):
                st.session_state.logo_position = pos

    advanced_position = st.selectbox(
        "Advanced Logo Position",
        ["Select Advanced Position", "top_left", "top_right", "left_center", "right_center", "left_bottom", "right_bottom"],
        index=0,
        key="advanced_position"
    )
    if advanced_position != "Select Advanced Position":
        st.session_state.logo_position = advanced_position

    st.write(f"Selected Logo Position: **{st.session_state.logo_position.capitalize()}**")

    if st.session_state.processed_files_data:
        st.subheader("Download Logoed Files")
        for file_path, original_name, file_data in st.session_state.processed_files_data:
            if os.path.exists(file_path):
                st.download_button(
                    label=f"Download {os.path.basename(file_path)}",
                    data=file_data,
                    file_name=os.path.basename(file_path),
                    key=f"download_{uuid.uuid4()}",
                    help=f"Download logoed file: {original_name}"
                )
        if len(st.session_state.processed_files_data) > 1:
            st.warning("Please allow multiple downloads in your browser if prompted.")
            if Config.USE_JAVASCRIPT_DOWNLOAD:
                trigger_multiple_downloads(st.session_state.processed_files_data)
            else:
                unique_key = f"download_all_btn_{uuid.uuid4()}"
                if st.button("Download All Files", key=unique_key, help="Download all logoed files"):
                    st.session_state.download_all_trigger = True
                    st.session_state.download_all_index = 0
                    logging.info("Download All Files initiated")
                    st.rerun()

    if not Config.USE_JAVASCRIPT_DOWNLOAD and st.session_state.download_all_trigger and st.session_state.processed_files_data:
        index = st.session_state.download_all_index
        if index is None or index >= len(st.session_state.processed_files_data):
            st.session_state.download_all_trigger = False
            st.session_state.download_all_index = None
            logging.info("Download All Files completed")
            st.success("All files downloaded successfully.")
        else:
            file_path, original_name, file_data = st.session_state.processed_files_data[index]
            logging.info(f"Triggering download for file {index + 1}/{len(st.session_state.processed_files_data)}: {os.path.basename(file_path)}")
            st.download_button(
                label=f"Downloading {os.path.basename(file_path)}",
                data=file_data,
                file_name=os.path.basename(file_path),
                key=f"download_all_{index}_{uuid.uuid4()}",
                help=f"Downloading logoed file: {original_name}"
            )
            st.session_state.download_all_index += 1
            logging.info(f"Updated download_all_index to {st.session_state.download_all_index}")
            st.rerun()

    if st.session_state.logo_file and st.session_state.media_files:
        if st.button("Start Logoing", key="start_logoing", type="primary"):
            st.session_state.logoed_files = []
            st.session_state.processed_files_data = []
            st.session_state.download_all_index = None
            st.session_state.download_all_trigger = False
            logo_path = os.path.join(base_path, "Logos", st.session_state.logo_file.name)
            if not os.path.exists(logo_path):
                with open(logo_path, "wb") as f:
                    f.write(st.session_state.logo_file.getbuffer())
            processed_files = []
            for media_file in st.session_state.media_files:
                media_path = os.path.join(base_path, "Media", media_file.name)
                output_filename = f"logoed_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{media_file.name}"
                output_path = os.path.join(base_path, "Logoed_Media", output_filename)
                with open(media_path, "wb") as f:
                    f.write(media_file.getbuffer())
                try:
                    media_type = "image" if media_file.name.lower().endswith((".jpg", "jpeg", "png")) else "video"
                    blurred_regions = []
                    if media_type == "image":
                        image = Image.open(media_path).convert("RGBA")
                        if st.session_state.blur_enabled and State.blur_enabled:
                            blurred_image, blurred_regions = process_image(
                                image,
                                State.dnn_net,
                                st.session_state.blur_enabled
                            )
                            st.image(blurred_image, caption="Blur Preview (Before Logo)", use_container_width=True)
                            if not blurred_regions:
                                st.warning("No faces detected for blurring. The image will be processed with the logo only.")
                        else:
                            blurred_image = image
                        if st.session_state.blur_enabled and State.blur_enabled:
                            if media_file.name not in st.session_state.blur_reviewed:
                                approved = review_blurred_regions(blurred_regions, media_type, base_path, media_file.name)
                                st.session_state.blur_reviewed[media_file.name] = approved
                                if not approved:
                                    st.error(f"Blur review not approved for {media_file.name}. Please adjust and reprocess.")
                                    continue
                        output_image = overlay_logo_on_image(blurred_image, logo_path, st.session_state.logo_position)
                        output_image.save(output_path, "PNG")
                        with open(output_path, "rb") as f:
                            file_data = f.read()
                        processed_files.append((output_path, media_file.name, file_data))
                    else:
                        intermediate_path = os.path.join(base_path, "Media", f"blurred_{media_file.name}")
                        blurred_regions = process_video(
                            media_path,
                            intermediate_path,
                            State.face_detector,
                            State.face_mesh,
                            State.yolo_model,
                            State.tracker,
                            st.session_state.blur_enabled and State.blur_enabled
                        )
                        if st.session_state.blur_enabled and State.blur_enabled:
                            if media_file.name not in st.session_state.blur_reviewed:
                                approved = review_blurred_regions(blurred_regions, media_type, base_path, media_file.name)
                                st.session_state.blur_reviewed[media_file.name] = approved
                                if not approved:
                                    st.error(f"Blur review not approved for {media_file.name}. Please adjust and reprocess.")
                                    continue
                        overlay_logo_on_video(intermediate_path if os.path.exists(intermediate_path) else media_path, logo_path, output_path, st.session_state.logo_position)
                        with open(output_path, "rb") as f:
                            file_data = f.read()
                        processed_files.append((output_path, media_file.name, file_data))
                        if os.path.exists(intermediate_path):
                            os.remove(intermediate_path)
                except Exception as e:
                    st.error(f"Error processing {media_file.name}: {e}")
                    logging.error(f"Error processing {media_file.name}: {str(e)}")
            st.session_state.logoed_files = [(file_path, original_name) for file_path, original_name, _ in processed_files]
            st.session_state.processed_files_data = processed_files
            st.subheader("Download Logoed Files")
            for file_path, original_name, file_data in st.session_state.processed_files_data:
                st.download_button(
                    label=f"Download {os.path.basename(file_path)}",
                    data=file_data,
                    file_name=os.path.basename(file_path),
                    key=f"download_{uuid.uuid4()}",
                    help=f"Download logoed file: {original_name}"
                )
            if len(st.session_state.processed_files_data) > 1:
                st.warning("Please allow multiple downloads in your browser if prompted.")
                if Config.USE_JAVASCRIPT_DOWNLOAD:
                    trigger_multiple_downloads(st.session_state.processed_files_data)
                else:
                    st.button("Download All Files", key=f"download_all_btn_{uuid.uuid4()}", help="Download all logoed files")

    if st.session_state.user_id == Config.ADMIN_USER_ID:
            st.subheader("Debug Session State")
            st.write(f"user: {st.session_state.user}")
            st.write(f"user_id: {st.session_state.user_id}")
            st.write(f"device_id: {st.session_state.device_id}")
            st.write(f"auth_error: {st.session_state.auth_error}")
            st.write(f"logo_file: {st.session_state.logo_file}")
            st.write(f"media_files: {len(st.session_state.media_files)} files")
            st.write(f"processed_files_data: {len(st.session_state.processed_files_data)} files")
            st.write(f"download_all_index: {st.session_state.download_all_index}")
            st.write(f"download_all_trigger: {st.session_state.download_all_trigger}")
            st.write(f"patch_applied: {st.session_state.patch_applied}")
            st.write(f"State.execution_count: {State.execution_count}")
            st.write(f"State.max_executions: {State.max_executions}")
            st.write(f"State.infinite_count: {State.infinite_count}")
            st.write(f"State.blur_enabled: {State.blur_enabled}")
            st.write(f"State.license_expiry: {State.license_expiry}")
            st.write(f"State.subscription_expiry: {State.subscription_expiry}")
            
# Admin panel
    if st.session_state.user_id == Config.ADMIN_USER_ID:
        st.subheader("Admin: Generate Patch")
        target_user = st.text_input("Target User ID")
        new_count = st.number_input("Start Execution Count", min_value=0, value=0)
        max_executions = st.number_input("Max Executions (0 for infinite)", min_value=0, value=Config.DEFAULT_MAX_EXECUTIONS)
        days_valid = st.number_input("License Days Valid", min_value=1, value=30)
        subscription_days_valid = st.number_input("Subscription Days Valid", min_value=1, value=30)
        blur_enabled = st.checkbox("Enable Face Blurring for User", value=True)
        if st.button("Generate Patch"):
            patch_id = apply_patch(target_user, new_count, days_valid, subscription_days_valid, max_executions, blur_enabled)

if __name__ == "__main__":
    main()