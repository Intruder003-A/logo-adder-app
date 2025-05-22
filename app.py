import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from datetime import datetime, timedelta
import uuid
import logging
import shutil
import requests
import json
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase Admin SDK
# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    try:
        # Use st.secrets for Firebase credentials in the cloud
        firebase_credentials = st.secrets["firebase"]["credential"]
        # Parse the JSON string into a dictionary
        cred_dict = json.loads(firebase_credentials)
        cred = credentials.Certificate(cred_dict)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        # Fallback for local development
        logging.warning(f"Failed to load credentials from st.secrets: {str(e)}. Falling back to local file.")
        cred = credentials.Certificate("logoadder-d22b5-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase Web API Key (stored in st.secrets for cloud deployment)
try:
    FIREBASE_API_KEY = st.secrets["firebase"]["api_key"]
except KeyError:
    # Fallback for local development
    FIREBASE_API_KEY = "AIzaSyD5DufwXe2cOPZniy-3K-LTRA-csWcbWEg"

# Configuration
class Config:
    LOGO_SIZE_PERCENT = 0.3
    LOGO_TRANSPARENCY = 0.45
    MAX_EXECUTIONS = 27
    EXECUTION_COLLECTION = "executions"
    LICENSE_COLLECTION = "licenses"
    FOLDERS = ["Logos", "Media", "Logoed_Media"]
    # Compute paths relative to app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DNN_PROTO_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt")
    DNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    BLUR_KERNEL = (101, 101)
    CONFIDENCE_THRESHOLD = 0.2

# State management
class State:
    user_id = None
    execution_count = 0
    license_expiry = None
    device_id = str(uuid.uuid4())
    net = None

# Ensure directories exist
def ensure_directories(base_path):
    for folder in Config.FOLDERS:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# Load DNN model for face detection
def load_dnn_model():
    # Log the paths being checked
    logging.info("Checking for DNN model files at: %s and %s", Config.DNN_PROTO_PATH, Config.DNN_MODEL_PATH)
    
    # Check if files exist
    if not os.path.exists(Config.DNN_PROTO_PATH):
        logging.error("DNN proto file not found at: %s", Config.DNN_PROTO_PATH)
    if not os.path.exists(Config.DNN_MODEL_PATH):
        logging.error("DNN model file not found at: %s", Config.DNN_MODEL_PATH)
    
    if not (os.path.exists(Config.DNN_PROTO_PATH) and os.path.exists(Config.DNN_MODEL_PATH)):
        logging.error("DNN model files not found.")
        return None
    
    # Attempt to load the model
    try:
        net = cv2.dnn.readNetFromCaffe(Config.DNN_PROTO_PATH, Config.DNN_MODEL_PATH)
        logging.info("DNN model loaded successfully.")
        return net
    except Exception as e:
        logging.error("Error loading DNN model: %s", str(e))
        return None

# Process frame for blurring faces
def process_frame(frame, net, width, height, blur_enabled):
    if not blur_enabled or net is None:
        logging.info("Blur skipped: blur_enabled=%s, net=%s", blur_enabled, net is not None)
        return frame
    logging.info("Applying blur to frame")
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    kernel_size = Config.BLUR_KERNEL
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > Config.CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            blurred_face = cv2.GaussianBlur(face_roi, kernel_size, 0)
            frame[y1:y2, x1:x2] = blurred_face
    return frame

# Process image for blurring
def process_image(image, net, blur_enabled):
    if not blur_enabled or net is None:
        logging.info("Blur skipped for image: blur_enabled=%s, net=%s", blur_enabled, net is not None)
        return image
    img_array = np.array(image.convert('RGB'))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_array.shape[:2]
    processed = process_frame(img_array, net, width, height, blur_enabled)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    logging.info("Blur applied to image")
    return Image.fromarray(processed).convert('RGBA')

# Process video for blurring, preserving audio
def process_video(video_path, output_path, net, blur_enabled):
    if not blur_enabled or net is None:
        logging.info("Blur skipped for video: blur_enabled=%s, net=%s", blur_enabled, net is not None)
        shutil.copy(video_path, output_path)
        return
    logging.info("Applying blur to video: input=%s, output=%s", video_path, output_path)
    try:
        original_clip = VideoFileClip(video_path)
        audio = original_clip.audio
        has_audio = audio is not None
        logging.info("Original video loaded, has_audio=%s", has_audio)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Failed to open video: %s", video_path)
            original_clip.close()
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        temp_output_path = output_path + "_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            original_clip.close()
            logging.error("Failed to initialize video writer: %s", temp_output_path)
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = process_frame(frame, net, width, height, blur_enabled)
            out.write(frame)

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
            logging.info("Removed temporary file: %s", temp_output_path)
        logging.info("Video saved with blur: %s", output_path)
    except Exception as e:
        logging.error("Error processing video blur: %s", e)
        shutil.copy(video_path, output_path)

# Check license and execution count
def check_license(user_id):
    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(f"{user_id}_{State.device_id}")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            State.execution_count = data.get("count", 0)
            State.license_expiry = data.get("expiry", datetime.now())
        else:
            State.execution_count = 0
            State.license_expiry = datetime.now() + timedelta(days=30)
            doc_ref.set({
                "user_id": user_id,
                "device_id": State.device_id,
                "count": State.execution_count,
                "expiry": State.license_expiry
            })
        if datetime.now() > State.license_expiry:
            st.error("License expired. Contact the service team for a new patch.")
            return False
        if State.execution_count >= Config.MAX_EXECUTIONS:
            st.error("Execution limit reached. Contact the service team for a new patch.")
            return False
        return True
    except Exception as e:
        logging.error("License check failed: %s", e)
        st.error("Error checking license. Contact the service team.")
        return False

# Increment execution count
def increment_execution(user_id):
    try:
        doc_ref = db.collection(Config.EXECUTION_COLLECTION).document(f"{user_id}_{State.device_id}")
        State.execution_count += 1
        doc_ref.update({"count": State.execution_count})
        logging.info("Execution count updated to %d for user %s", State.execution_count, user_id)
    except Exception as e:
        logging.error("Error incrementing execution count: %s", e)

# Apply patch (admin function)
def apply_patch(user_id, new_count, days_valid):
    try:
        patch_id = str(uuid.uuid4())
        doc_ref = db.collection(Config.LICENSE_COLLECTION).document(patch_id)
        expiry = datetime.now() + timedelta(days=days_valid)
        doc_ref.set({
            "user_id": user_id,
            "device_id": State.device_id,
            "new_count": new_count,
            "expiry": expiry,
            "used": False
        })
        st.success(f"Patch ID: {patch_id} (Valid for {days_valid} days, {new_count} executions)")
        return patch_id
    except Exception as e:
        logging.error("Error generating patch: %s", e)
        st.error("Error generating patch.")
        return None

# Validate and apply patch
def validate_patch(patch_id, user_id):
    try:
        doc_ref = db.collection(Config.LICENSE_COLLECTION).document(patch_id)
        doc = doc_ref.get()
        if not doc.exists:
            st.error("Invalid patch ID.")
            return False
        data = doc.to_dict()
        if data["used"]:
            st.error("Patch already used.")
            return False
        if datetime.now() > data["expiry"]:
            st.error("Patch expired.")
            return False
        if data["user_id"] != user_id or data["device_id"] != State.device_id:
            st.error("Patch not valid for this user or device.")
            return False
        execution_ref = db.collection(Config.EXECUTION_COLLECTION).document(f"{user_id}_{State.device_id}")
        execution_ref.update({
            "count": data["new_count"],
            "expiry": data["expiry"]
        })
        doc_ref.update({"used": True})
        State.execution_count = data["new_count"]
        State.license_expiry = data["expiry"]
        st.success("Patch applied successfully.")
        return True
    except Exception as e:
        logging.error("Error validating patch: %s", e)
        st.error("Error applying patch.")
        return False

# Overlay logo on image
def overlay_logo_on_image(image, logo_path):
    try:
        logo = Image.open(logo_path).convert("RGBA")
        img_width, img_height = image.size
        max_logo_size = int(min(img_width, img_height) * Config.LOGO_SIZE_PERCENT)
        logo.thumbnail((max_logo_size, max_logo_size), Image.Resampling.LANCZOS)
        logo_array = np.array(logo)
        logo_array[:, :, 3] = (logo_array[:, :, 3] * Config.LOGO_TRANSPARENCY).astype(np.uint8)
        logo = Image.fromarray(logo_array)
        x = (img_width - logo.size[0]) // 2
        y = (img_height - logo.size[1]) // 2
        output = Image.new("RGBA", image.size)
        output.paste(image, (0, 0))
        output.paste(logo, (x, y), logo)
        return output
    except Exception as e:
        logging.error("Error overlaying logo on image: %s", e)
        return image

# Overlay logo on video
def overlay_logo_on_video(video_path, logo_path, output_path):
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
        x = (vid_width - logo.size[0]) // 2
        y = (vid_height - logo.size[1]) // 2
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
        logging.info("Video saved with logo to %s", output_path)
    except Exception as e:
        logging.error("Error processing video: %s", e)
        raise

# Verify email and password using Firebase Auth REST API
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
        if "idToken" in data:
            return data.get("localId"), None
        else:
            error_message = data.get("error", {}).get("message", "Invalid credentials")
            if "EMAIL_NOT_FOUND" in error_message:
                return None, "Email not registered."
            elif "INVALID_PASSWORD" in error_message:
                return None, "Incorrect password."
            else:
                return None, error_message
    except requests.exceptions.RequestException as e:
        return None, f"Error verifying credentials: {str(e)}"

# Streamlit app
def main():
    st.title("Logo Adder App")
    
    # Authentication
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.auth_error = None
        st.session_state.reset_message = None

    if not st.session_state.user:
        st.subheader("Login")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Login"):
                if not email or not password:
                    st.session_state.auth_error = "Please enter both email and password."
                else:
                    uid, error = verify_user(email, password)
                    if uid:
                        st.session_state.user = uid
                        State.user_id = uid
                        st.session_state.auth_error = None
                        st.session_state.reset_message = None
                        st.success("Logged in successfully.")
                        st.rerun()
                    else:
                        st.session_state.auth_error = error

        with col2:
            if st.button("Forgot Password?"):
                if not email:
                    st.session_state.reset_message = "Please enter your email to reset the password."
                else:
                    try:
                        auth.get_user_by_email(email)
                        reset_link = auth.generate_password_reset_link(email)
                        st.session_state.reset_message = f"Password reset link: {reset_link}"
                        st.session_state.auth_error = None
                    except auth.UserNotFoundError:
                        st.session_state.reset_message = "Email not registered."
                    except Exception as e:
                        st.session_state.reset_message = f"Error generating reset link: {str(e)}"

        if st.session_state.auth_error:
            st.error(f"Login failed: {st.session_state.auth_error}")
        if st.session_state.reset_message:
            if "link" in st.session_state.reset_message:
                st.success(st.session_state.reset_message)
            else:
                st.error(st.session_state.reset_message)

        return

    # Check license (only after login)
    if not check_license(State.user_id):
        st.subheader("Apply Patch")
        patch_id = st.text_input("Enter Patch ID")
        if st.button("Apply Patch"):
            if validate_patch(patch_id, State.user_id):
                st.rerun()
        return

    # Load DNN model (only once, after login)
    if State.net is None:
        State.net = load_dnn_model()
        if State.net is None:
            st.warning("Face detection model failed to load. Blurring functionality will be disabled.")

    # File upload and processing (only after login and license check)
    st.subheader("Upload Files")
    base_path = "temp_files"
    ensure_directories(base_path)

    # Initialize session state for files and options
    if "logo_file" not in st.session_state:
        st.session_state.logo_file = None
    if "media_files" not in st.session_state:
        st.session_state.media_files = []
    if "blur_enabled" not in st.session_state:
        st.session_state.blur_enabled = False

    # File upload
    logo_file = st.file_uploader("Upload Logo (PNG)", type=["png"], key="logo")
    media_files = st.file_uploader("Upload Media (Images/Videos)", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True, key="media")
    st.session_state.blur_enabled = st.checkbox("Enable Face Blurring", value=st.session_state.blur_enabled)

    # Update session state with uploaded files
    if logo_file:
        st.session_state.logo_file = logo_file
    if media_files:
        st.session_state.media_files = media_files

    # Start Logoing button
    if st.session_state.logo_file and st.session_state.media_files:
        if st.button("Start Logoing"):
            logo_path = os.path.join(base_path, "Logos", st.session_state.logo_file.name)
            with open(logo_path, "wb") as f:
                f.write(st.session_state.logo_file.getbuffer())

            for media_file in st.session_state.media_files:
                media_path = os.path.join(base_path, "Media", media_file.name)
                output_filename = f"logoed_{datetime.now().strftime('%Y%m%d%H%M%S')}_{media_file.name}"
                output_path = os.path.join(base_path, "Logoed_Media", output_filename)
                intermediate_path = os.path.join(base_path, "Media", f"blurred_{media_file.name}")

                with open(media_path, "wb") as f:
                    f.write(media_file.getbuffer())

                try:
                    # Step 1: Apply blurring if enabled
                    if media_file.name.lower().endswith((".jpg", "jpeg", "png")):
                        image = Image.open(media_path).convert("RGBA")
                        blurred_image = process_image(image, State.net, st.session_state.blur_enabled)
                        blurred_image.save(intermediate_path, "PNG")
                    else:
                        process_video(media_path, intermediate_path, State.net, st.session_state.blur_enabled)

                    # Step 2: Apply logo
                    if media_file.name.lower().endswith((".jpg", "jpeg", "png")):
                        image = Image.open(intermediate_path).convert("RGBA")
                        output_image = overlay_logo_on_image(image, logo_path)
                        output_image.save(output_path, "PNG")
                        with open(output_path, "rb") as f:
                            st.download_button(f"Download {output_filename}", f, file_name=output_filename)
                    else:
                        overlay_logo_on_video(intermediate_path, logo_path, output_path)
                        with open(output_path, "rb") as f:
                            st.download_button(f"Download {output_filename}", f, file_name=output_filename)
                    increment_execution(State.user_id)
                except Exception as e:
                    st.error(f"Error processing {media_file.name}: {e}")
                finally:
                    # Clean up intermediate file
                    if os.path.exists(intermediate_path):
                        os.remove(intermediate_path)

            # Clean up base directory
            shutil.rmtree(base_path)
            ensure_directories(base_path)

            # Clear session state
            st.session_state.logo_file = None
            st.session_state.media_files = []

    # Admin panel
    if st.session_state.user == "CO9n9TnhWoclEtyuH8jfzsXs7tt2":
        st.subheader("Admin: Generate Patch")
        target_user = st.text_input("Target User ID")
        new_count = st.number_input("New Execution Count", min_value=0, value=27)
        days_valid = st.number_input("Days Valid", min_value=1, value=30)
        if st.button("Generate Patch"):
            patch_id = apply_patch(target_user, new_count, days_valid)

if __name__ == "__main__":
    main()