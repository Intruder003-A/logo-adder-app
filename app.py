import os
import uuid
import logging
import traceback
from datetime import datetime, timedelta, timezone
import streamlit as st
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud.firestore_v1 import FieldFilter
import json
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Firebase initialization
try:
    if not firebase_admin._apps:
        firebase_creds = json.loads(st.secrets["firebase"]["credentials"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase Admin SDK and Firestore initialized successfully")
except Exception as e:
    logging.error(f"Firebase initialization failed: {str(e)}\n{traceback.format_exc()}")
    st.error("Failed to initialize Firebase. Please check configuration.")
    db = None

# Configuration class
class Config:
    BASE_DIR = os.path.join(os.getcwd(), "Data")
    DEFAULT_MAX_EXECUTIONS = 27
    ADMIN_USER_ID = "CO9n9TnhWoclEtyuH8jfzsXs7tt2"
    USE_JAVASCRIPT_DOWNLOAD = False

# State class for global state
class State:
    execution_count = 0
    max_executions = Config.DEFAULT_MAX_EXECUTIONS
    infinite_count = False
    license_expiry = datetime.now(timezone.utc) + timedelta(days=30)  # Default to 30 days from now
    subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)  # Default to 30 days from now
    blur_enabled = True
    face_detector = None
    face_mesh = None
    yolo_model = None
    tracker = None
    dnn_net = None

# Ensure directories exist
def ensure_directories(base_dir):
    for subdir in ["Logos", "Media", "Logoed_Media"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    logging.info(f"Ensured directories exist under {base_dir}")

# Initialize AI models
def initialize_ai_models():
    try:
        logging.info("Initializing AI models for face detection, landmarks, and body detection")
        State.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        State.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=10, min_detection_confidence=0.5)
        State.yolo_model = YOLO("yolov8n.pt")
        State.tracker = DeepSort(max_age=30, embedder="mobilenet", max_cosine_distance=0.2)
        prototxt_path = os.path.join("models", "deploy.prototxt")
        caffemodel_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
        logging.info(f"Checking for DNN model files at: {prototxt_path} and {caffemodel_path}")
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            State.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            logging.info("DNN model loaded successfully.")
        else:
            logging.error("DNN model files not found.")
            st.error("DNN model files not found. Face blurring may not work.")
        logging.info("AI models initialized successfully")
    except Exception as e:
        logging.error(f"AI model initialization failed: {str(e)}\n{traceback.format_exc()}")
        st.error("Failed to initialize AI models. Some features may be unavailable.")

# Verify user credentials
def verify_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        # Note: Firebase Admin SDK cannot verify passwords; assume frontend handles this
        logging.info(f"User verified: {email}, UID: {user.uid}")
        return user.uid, None
    except auth.UserNotFoundError:
        logging.error(f"User not found: {email}")
        return None, "Email not registered."
    except Exception as e:
        logging.error(f"User verification failed: {str(e)}\n{traceback.format_exc()}")
        return None, f"Authentication error: {str(e)}"

# Check license validity
def check_license(user_id, force_refresh=False):
    if not db:
        logging.error("Firestore unavailable. License check failed.")
        return False
    try:
        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()
        if not doc.exists:
            logging.error(f"No license data found for user {user_id}")
            State.execution_count = 0
            State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
            State.infinite_count = False
            State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
            State.blur_enabled = False
            return False
        data = doc.to_dict()
        State.execution_count = data.get("execution_count", 0)
        State.max_executions = data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS)
        State.infinite_count = data.get("infinite_count", False)
        State.blur_enabled = data.get("blur_enabled", True)
        # Handle expiry dates
        license_expiry = data.get("license_expiry")
        subscription_expiry = data.get("subscription_expiry")
        now = datetime.now(timezone.utc)
        State.license_expiry = license_expiry.to_pydatetime() if license_expiry else now + timedelta(days=30)
        State.subscription_expiry = subscription_expiry.to_pydatetime() if subscription_expiry else now + timedelta(days=30)
        logging.info(f"License checked for {user_id}: count={State.execution_count}, max={State.max_executions}, expiry={State.license_expiry}, sub_expiry={State.subscription_expiry}")
        if State.infinite_count or State.execution_count < State.max_executions:
            if State.license_expiry > now and State.subscription_expiry > now:
                return True
        logging.warning(f"License invalid for {user_id}: count={State.execution_count}/{State.max_executions}, expiry={State.license_expiry}, sub_expiry={State.subscription_expiry}")
        return False
    except Exception as e:
        logging.error(f"License check failed for {user_id}: {str(e)}\n{traceback.format_exc()}")
        State.execution_count = 0
        State.max_executions = Config.DEFAULT_MAX_EXECUTIONS
        State.infinite_count = False
        State.license_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        State.subscription_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        State.blur_enabled = False
        return False

# Validate patch
def validate_patch(patch_id, user_id):
    if not db:
        logging.error("Firestore unavailable. Patch validation failed.")
        return False
    try:
        patch_ref = db.collection("patches").document(patch_id)
        patch = patch_ref.get()
        if not patch.exists:
            logging.error(f"Patch {patch_id} not found")
            return False
        patch_data = patch.to_dict()
        if patch_data.get("user_id") != user_id:
            logging.error(f"Patch {patch_id} not valid for user {user_id}")
            return False
        expiry = patch_data.get("patch_expiry").to_pydatetime() if patch_data.get("patch_expiry") else datetime.now(timezone.utc)
        if expiry < datetime.now(timezone.utc):
            logging.error(f"Patch {patch_id} expired")
            return False
        # Apply patch data
        user_ref = db.collection("users").document(user_id)
        user_ref.update({
            "execution_count": min(patch_data.get("new_count", 0), Config.DEFAULT_MAX_EXECUTIONS),
            "max_executions": patch_data.get("max_executions", Config.DEFAULT_MAX_EXECUTIONS),
            "infinite_count": patch_data.get("infinite_count", False),
            "license_expiry": patch_data.get("license_expiry", datetime.now(timezone.utc) + timedelta(days=30)),
            "subscription_expiry": patch_data.get("subscription_expiry", datetime.now(timezone.utc) + timedelta(days=30)),
            "blur_enabled": patch_data.get("blur_enabled", True)
        })
        logging.info(f"Patch {patch_id} validated and applied for user {user_id}")
        patch_ref.delete()  # Delete patch after use
        return True
    except Exception as e:
        logging.error(f"Patch validation failed for {patch_id}: {str(e)}\n{traceback.format_exc()}")
        return False

# Apply patch (Admin)
def apply_patch(user_id, new_count, days_valid, subscription_days_valid, max_executions, blur_enabled):
    if not db:
        logging.error("Firestore unavailable. Patch generation failed.")
        return None
    try:
        patch_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        patch_data = {
            "user_id": user_id,
            "new_count": min(new_count, Config.DEFAULT_MAX_EXECUTIONS),
            "max_executions": max_executions,
            "infinite_count": max_executions == 0,
            "patch_expiry": now + timedelta(days=days_valid),
            "license_expiry": now + timedelta(days=days_valid),
            "subscription_expiry": now + timedelta(days=subscription_days_valid),
            "blur_enabled": blur_enabled,
            "created_at": now
        }
        db.collection("patches").document(patch_id).set(patch_data)
        logging.info(f"Patch {patch_id} generated for user {user_id}")
        return patch_id
    except Exception as e:
        logging.error(f"Patch generation failed: {str(e)}\n{traceback.format_exc()}")
        return None

# Increment execution count
def increment_execution(user_id, media_key):
    if not db:
        logging.error("Firestore unavailable. Execution count not incremented.")
        return
    try:
        user_ref = db.collection("users").document(user_id)
        if not State.infinite_count and State.execution_count < State.max_executions:
            State.execution_count += 1
            user_ref.update({"execution_count": State.execution_count})
            logging.info(f"Execution count incremented for user {user_id} to {State.execution_count} for {media_key}")
    except Exception as e:
        logging.error(f"Failed to increment execution count for {user_id}: {str(e)}\n{traceback.format_exc()}")

# Debug license limits (Admin)
def debug_license_limits(user_id):
    if not db:
        st.warning("Firestore unavailable. Debug tools limited.")
        return
    try:
        st.write(f"User ID: {user_id}")
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            st.write(f"Execution Count: {data.get('execution_count', 0)}")
            st.write(f"Max Executions: {data.get('max_executions', Config.DEFAULT_MAX_EXECUTIONS)}")
            st.write(f"Infinite Count: {data.get('infinite_count', False)}")
            expiry = data.get("license_expiry")
            sub_expiry = data.get("subscription_expiry")
            st.write(f"License Expiry: {expiry.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S %Z') if expiry else 'Not set'}")
            st.write(f"Subscription Expiry: {sub_expiry.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S %Z') if sub_expiry else 'Not set'}")
            st.write(f"Blur Enabled: {data.get('blur_enabled', False)}")
        else:
            st.write("No license data found.")
    except Exception as e:
        logging.error(f"Debug license limits failed: {str(e)}\n{traceback.format_exc()}")
        st.error("Failed to retrieve license data.")


# Streamlit app
# Streamlit app
def main():
    st.set_page_config(page_title="Logo Adder App", layout="wide")
    st.title("Logo Adder App")

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'device_id' not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
    if 'patch_applied' not in st.session_state:
        st.session_state.patch_applied = False
    if 'blur_enabled' not in st.session_state:
        st.session_state.blur_enabled = False
    if 'logo_positions' not in st.session_state:
        st.session_state.logo_positions = {}
    if 'manual_positioning' not in st.session_state:
        st.session_state.manual_positioning = False
    if 'selected_position' not in st.session_state:
        st.session_state.selected_position = "Center"
    if 'auth_error' not in st.session_state:
        st.session_state.auth_error = None
    if 'reset_message' not in st.session_state:
        st.session_state.reset_message = None
    if 'output_files' not in st.session_state:
        st.session_state.output_files = []
    logging.info(f"Initialized session state with device_id: {st.session_state.device_id}")
    logging.info(f"Session state at start: user={st.session_state.user}, user_id={st.session_state.user_id}, device_id={st.session_state.device_id}, patch_applied={st.session_state.patch_applied}, blur_enabled={st.session_state.blur_enabled}, manual_positioning={st.session_state.manual_positioning}")

    # Sidebar for login and patch application
    with st.sidebar:
        st.header("User Authentication")
        if st.session_state.user is None:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns([1, 2])
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

        # Admin patch generation
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

        # Admin debug tools
        if st.session_state.user_id == Config.ADMIN_USER_ID:
            st.markdown("---")
            debug_license_limits(st.session_state.user_id)

    # Main app logic
    if st.session_state.user is None or not st.session_state.patch_applied:
        if st.session_state.user is None:
            st.warning("Please log in to use the app.")
        elif not st.session_state.patch_applied:
            st.warning("Please apply a valid patch or ensure your license is active.")
        return

    # Ensure directories exist
    ensure_directories(Config.BASE_DIR)

    # Initialize AI models if not already done
    if not all([State.face_detector, State.face_mesh, State.yolo_model, State.tracker, State.dnn_net]):
        initialize_ai_models()

    # File upload section
    st.header("Upload Files")
    logo_file = st.file_uploader("Upload Logo (PNG with transparency recommended)", type=["png", "jpg", "jpeg"])
    media_files = st.file_uploader("Upload Media (Images or Videos)", type=["jpg", "jpeg", "png", "mp4"], accept_multiple_files=True)

    # Clear output_files when new media files are uploaded or on app start
    if media_files and media_files != st.session_state.get('last_media_files', []):
        st.session_state.output_files = []
        st.session_state.last_media_files = media_files
    elif not media_files:
        st.session_state.output_files = []

    # Logo position selection
    st.header("Logo Position")
    position_options = ["Center", "Top", "Bottom", "Left", "Right", "Top Left", "Top Right", "Left Center", "Right Center", "Left Bottom", "Right Bottom"]
    manual_positioning = st.checkbox("Enable Manual Logo Positioning", value=st.session_state.manual_positioning, key="manual_positioning")
    if manual_positioning != st.session_state.manual_positioning:
        st.session_state.manual_positioning = manual_positioning
    
    # Preset position dropdown
    position_option = st.selectbox(
        "Select Logo Position",
        position_options,
        index=position_options.index(st.session_state.selected_position),
        key="logo_position_select"
    )
    st.session_state.selected_position = position_option
    
    # Quick position buttons
    st.write("Quick Position Selectors:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Top", key="quick_top"):
            st.session_state.selected_position = "Top"
            st.rerun()
    with col2:
        if st.button("Bottom", key="quick_bottom"):
            st.session_state.selected_position = "Bottom"
            st.rerun()
    with col3:
        if st.button("Left", key="quick_left"):
            st.session_state.selected_position = "Left"
            st.rerun()
    with col4:
        if st.button("Right", key="quick_right"):
            st.session_state.selected_position = "Right"
            st.rerun()
    with col5:
        if st.button("Center", key="quick_center"):
            st.session_state.selected_position = "Center"
            st.rerun()

    # Debug information
    st.write("**Debug Info**")
    st.write(f"- Selected Position: {position_option}")
    st.write(f"- Manual Positioning Enabled: {st.session_state.manual_positioning}")
    st.write(f"- Logo File Uploaded: {'Yes' if logo_file else 'No'}")
    st.write(f"- Media Files Uploaded: {len(media_files) if media_files else 0}")
    logging.info(f"Logo position selected: {position_option}, manual_positioning={st.session_state.manual_positioning}, logo_file={bool(logo_file)}, media_files_count={len(media_files) if media_files else 0}")

    # Manual positioning controls
    custom_positions = {}
    if st.session_state.manual_positioning:
        st.subheader("Manual Logo Positioning")
        if not logo_file or not media_files:
            st.warning("Please upload both a logo and at least one media file to configure manual positioning.")
            logging.info("Manual positioning UI skipped: logo_file or media_files missing")
        else:
            logo_path = os.path.join(Config.BASE_DIR, "Logos", logo_file.name)
            try:
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getbuffer())
                logging.info(f"Saved logo file to {logo_path}")
            except Exception as e:
                st.error(f"Failed to save logo file: {str(e)}")
                logging.error(f"Error saving logo file to {logo_path}: {str(e)}\n{traceback.format_exc()}")
                return
            
            for media_file in media_files:
                media_key = media_file.name
                # Sanitize media_key for DOM ID
                safe_media_key = ''.join(c if c.isalnum() else '_' for c in media_key).strip('_')
                
                # Calculate initial position based on preset
                image = Image.open(media_file)
                img_width, img_height = image.size
                preset_positions = {
                    "Center": (500, 500),
                    "Top": (500, 100),
                    "Bottom": (500, 900),
                    "Left": (100, 500),
                    "Right": (900, 500),
                    "Top Left": (100, 100),
                    "Top Right": (900, 100),
                    "Left Center": (100, 500),
                    "Right Center": (900, 500),
                    "Left Bottom": (100, 900),
                    "Right Bottom": (900, 900)
                }
                initial_x, initial_y = preset_positions.get(st.session_state.selected_position, (500, 500))
                
                # Initialize logo position with preset values
                if media_key not in st.session_state.logo_positions:
                    st.session_state.logo_positions[media_key] = {
                        "x_pos": initial_x,
                        "y_pos": initial_y,
                        "scale": 1.0,
                        "rotation": 0,
                        "opacity": 1.0
                    }
                logging.info(f"Initialized logo position for {media_key}: x_pos={initial_x}, y_pos={initial_y}")
                
                st.markdown(f"### Positioning for {media_key}")
                col_preview, col_controls = st.columns([3, 2])
                
                with col_controls:
                    st.markdown("**Adjust Logo Settings**")
                    def update_position():
                        logging.info(f"Slider updated for {media_key}: x_pos={st.session_state.logo_positions[media_key]['x_pos']}, y_pos={st.session_state.logo_positions[media_key]['y_pos']}, opacity={st.session_state.logo_positions[media_key]['opacity']}")
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
                        2.0,
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
                        "Opacity",
                        0.0,
                        1.0,
                        st.session_state.logo_positions[media_key]["opacity"],
                        step=0.05,
                        key=f"opacity_{safe_media_key}",
                        on_change=update_position
                    )
                    
                    # Update session state
                    st.session_state.logo_positions[media_key].update({
                        "x_pos": x_pos,
                        "y_pos": y_pos,
                        "scale": scale,
                        "rotation": rotation,
                        "opacity": opacity
                    })
                    
                    # Click-to-position and drag functionality
                    st.markdown("**Click or Drag Logo to Position**")
                    click_position = st.text_input("Click/Drag Position (X, Y)", "", key=f"click_pos_{safe_media_key}", disabled=True)
                
                with col_preview:
                    # Generate preview with debug logging
                    logging.info(f"Generating preview for {media_key} with x_pos={x_pos}, y_pos={y_pos}, opacity={opacity}")
                    preview_bytes = generate_preview_image(
                        media_file,
                        logo_path,
                        custom_position=(x_pos, y_pos),
                        scale=scale,
                        rotation=rotation,
                        opacity=opacity
                    )
                    if preview_bytes:
                        st.image(preview_bytes, caption=f"Preview for {media_key}", use_container_width=True)
                        
                        # JavaScript for click-to-position and drag with reattachment
                        js_code = f"""
                        <script>
                        function attachDragListeners_{safe_media_key}() {{
                            let isDragging = false;
                            let currentX;
                            let currentY;
                            const previewImg = document.querySelector('img[alt="Preview for {media_key}"]');
                            const positionInput = document.getElementById('click_pos_{safe_media_key}');
                            if (!previewImg || !positionInput) {{
                                console.warn('Preview image or position input not found for {safe_media_key}');
                                return;
                            }}
                            const imgRect = previewImg.getBoundingClientRect();
                            const updatePosition = (x, y) => {{
                                const xScaled = Math.min(Math.max((x / imgRect.width) * 1000, 0), 1000);
                                const yScaled = Math.min(Math.max((y / imgRect.height) * 1000, 0), 1000);
                                positionInput.value = `(${Math.round(xScaled)}, ${Math.round(yScaled)})`;
                                window.StreamlitAPI.setComponentValue('x_pos_{safe_media_key}', xScaled);
                                window.StreamlitAPI.setComponentValue('y_pos_{safe_media_key}', yScaled);
                            }};
                            previewImg.addEventListener('click', (e) => {{
                                const x = e.clientX - imgRect.left;
                                const y = e.clientY - imgRect.top;
                                updatePosition(x, y);
                            }});
                            previewImg.addEventListener('mousedown', (e) => {{
                                isDragging = true;
                                currentX = e.clientX - imgRect.left;
                                currentY = e.clientY - imgRect.top;
                                updatePosition(currentX, currentY);
                            }});
                            previewImg.addEventListener('mousemove', (e) => {{
                                if (isDragging) {{
                                    currentX = e.clientX - imgRect.left;
                                    currentY = e.clientY - imgRect.top;
                                    updatePosition(currentX, currentY);
                                }}
                            }});
                            previewImg.addEventListener('mouseup', () => {{
                                isDragging = false;
                            }});
                            previewImg.addEventListener('mouseleave', () => {{
                                isDragging = false;
                            }});
                            console.log('Drag listeners attached for {safe_media_key}');
                        }}
                        document.addEventListener('DOMContentLoaded', function() {{
                            attachDragListeners_{safe_media_key}();
                            setTimeout(attachDragListeners_{safe_media_key}, 1000); // Retry after 1s
                        }});
                        </script>
                        """
                        st.markdown(js_code, unsafe_allow_html=True)
                    else:
                        st.error(f"Failed to generate preview for {media_key}")
                        logging.error(f"Preview generation failed for {media_key}")
                
                custom_positions[media_key] = st.session_state.logo_positions[media_key]

    # Face blurring option
    st.header("Face Blurring")
    st.session_state.blur_enabled = st.checkbox(
        "Enable Face Blurring",
        value=st.session_state.blur_enabled,
        key="blur_enabled_toggle",
        disabled=not State.blur_enabled
    )
    if not State.blur_enabled:
        st.warning("Face blurring is disabled for your license.")
    logging.info(f"Blur enabled: {st.session_state.blur_enabled}, license blur_enabled: {State.blur_enabled}")

    # Process files
    if st.button("Process Files"):
        if not logo_file or not media_files:
            st.error("Please upload both a logo and at least one media file.")
            logging.error("Process attempted without logo or media files")
        elif st.session_state.user_id is None:
            st.error("User not authenticated. Please log in.")
            logging.error("Process attempted without authenticated user")
        elif not check_license(st.session_state.user_id):
            st.error("License check failed. Please apply a valid patch or contact support.")
            logging.error(f"License check failed for user {st.session_state.user_id}")
        else:
            st.session_state.output_files = []
            logo_path = os.path.join(Config.BASE_DIR, "Logos", logo_file.name)
            try:
                with open(logo_path, "wb") as f:
                    f.write(logo_file.getbuffer())
                logging.info(f"Saved logo file to {logo_path}")
            except Exception as e:
                st.error(f"Failed to save logo file: {str(e)}")
                logging.error(f"Error saving logo file to {logo_path}: {str(e)}\n{traceback.format_exc()}")
                return
            
            for media_file in media_files:
                media_key = media_file.name
                media_path = os.path.join(Config.BASE_DIR, "Media", media_key)
                output_filename = f"logoed_{media_key}"
                output_path = os.path.join(Config.BASE_DIR, "Logoed_Media", output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    with open(media_path, "wb") as f:
                        f.write(media_file.getbuffer())
                    logging.info(f"Saved media file to {media_path}")
                    
                    # Determine media type
                    if media_key.lower().endswith(('.jpg', '.jpeg', '.png')):
                        media_type = "image"
                        image = Image.open(media_path).convert("RGBA")
                        if st.session_state.blur_enabled:
                            processed_image, blurred_regions = process_image(image, State.dnn_net, st.session_state.blur_enabled)
                            if not review_blurred_regions(blurred_regions, media_type, Config.BASE_DIR, media_key):
                                st.warning(f"Skipping {media_key} due to unapproved blur regions.")
                                logging.info(f"Skipped processing {media_key} due to unapproved blur regions")
                                continue
                        else:
                            processed_image = image
                            blurred_regions = []
                        
                        position = None if st.session_state.manual_positioning else st.session_state.selected_position
                        custom_position = custom_positions.get(media_key, {}).get("x_pos"), custom_positions.get(media_key, {}).get("y_pos")
                        scale = custom_positions.get(media_key, {}).get("scale", 1.0)
                        rotation = custom_positions.get(media_key, {}).get("rotation", 0)
                        opacity = custom_positions.get(media_key, {}).get("opacity", 1.0)
                        if not st.session_state.manual_positioning:
                            custom_position = None
                        
                        logging.info(f"Processing image {media_key}: position={position}, custom_position={custom_position}, scale={scale}, rotation={rotation}, opacity={opacity}")
                        processed_image = overlay_logo_on_image(
                            processed_image,
                            logo_path,
                            position=position,
                            custom_position=custom_position,
                            scale=scale,
                            rotation=rotation,
                            opacity=opacity
                        )
                        processed_image.save(output_path, "PNG")
                        logging.info(f"Saved logoed image to {output_path}")
                    else:
                        media_type = "video"
                        if st.session_state.blur_enabled:
                            temp_output_path = os.path.join(Config.BASE_DIR, "Logoed_Media", f"temp_{media_key}")
                            blurred_regions = process_video(
                                media_path,
                                temp_output_path,
                                State.face_detector,
                                State.face_mesh,
                                State.yolo_model,
                                State.tracker,
                                st.session_state.blur_enabled
                            )
                            if not review_blurred_regions(blurred_regions, media_type, Config.BASE_DIR, media_key):
                                st.warning(f"Skipping {media_key} due to unapproved blur regions.")
                                logging.info(f"Skipped processing {media_key} due to unapproved blur regions")
                                if os.path.exists(temp_output_path):
                                    os.remove(temp_output_path)
                                continue
                            media_path = temp_output_path
                        
                        position = None if st.session_state.manual_positioning else st.session_state.selected_position
                        custom_position = custom_positions.get(media_key, {}).get("x_pos"), custom_positions.get(media_key, {}).get("y_pos")
                        scale = custom_positions.get(media_key, {}).get("scale", 1.0)
                        rotation = custom_positions.get(media_key, {}).get("rotation", 0)
                        opacity = custom_positions.get(media_key, {}).get("opacity", 1.0)
                        if not st.session_state.manual_positioning:
                            custom_position = None
                        
                        logging.info(f"Processing video {media_key}: position={position}, custom_position={custom_position}, scale={scale}, rotation={rotation}, opacity={opacity}")
                        overlay_logo_on_video(
                            media_path,
                            logo_path,
                            output_path,
                            position=position,
                            custom_position=custom_position,
                            scale=scale,
                            rotation=rotation,
                            opacity=opacity
                        )
                        if st.session_state.blur_enabled and os.path.exists(media_path):
                            os.remove(media_path)
                        logging.info(f"Saved logoed video to {output_path}")
                    
                    try:
                        with open(output_path, "rb") as f:
                            file_data = f.read()
                        st.session_state.output_files.append((output_path, output_filename, file_data))
                        increment_execution(st.session_state.user_id, media_key)
                        logging.info(f"Added {output_filename} to output files, execution count incremented")
                    except Exception as e:
                        st.error(f"Failed to read output file {output_filename}: {str(e)}")
                        logging.error(f"Error reading output file {output_filename}: {str(e)}\n{traceback.format_exc()}")
                
                except Exception as e:
                    st.error(f"Error processing {media_key}: {str(e)}")
                    logging.error(f"Error processing {media_key}: {str(e)}\n{traceback.format_exc()}")
                    continue

    # Download section
    if st.session_state.output_files:
        st.header("Download Processed Files")
        if Config.USE_JAVASCRIPT_DOWNLOAD:
            trigger_multiple_downloads(st.session_state.output_files)
        else:
            for file_path, file_name, file_data in st.session_state.output_files:
                if not isinstance(file_path, str) or not isinstance(file_name, str) or not isinstance(file_data, bytes):
                    logging.error(f"Invalid output file entry: path={file_path}, name={file_name}, data_type={type(file_data)}")
                    continue
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="image/png" if file_name.lower().endswith('.png') else "video/mp4"
                )
                logging.info(f"Provided download button for {file_name}")

    # Display execution count
    st.markdown("---")
    st.write(f"Execution Count: {State.execution_count} / {State.max_executions if not State.infinite_count else 'Unlimited'}")
    if State.infinite_count:
        st.write("Infinite executions enabled.")
    expiry_display = State.license_expiry
    sub_expiry_display = State.subscription_expiry
    if expiry_display is not None and expiry_display.tzinfo is None:
        expiry_display = expiry_display.replace(tzinfo=timezone.utc)
    if sub_expiry_display is not None and sub_expiry_display.tzinfo is None:
        sub_expiry_display = sub_expiry_display.replace(tzinfo=timezone.utc)
    st.write(f"License Expiry: {expiry_display.strftime('%Y-%m-%d %H:%M:%S %Z') if expiry_display is not None else 'Not set'}")
    st.write(f"Subscription Expiry: {sub_expiry_display.strftime('%Y-%m-%d %H:%M:%S %Z') if sub_expiry_display is not None else 'Not set'}")
    st.write(f"Face Blurring Enabled: {State.blur_enabled}")
    logging.info(f"Displayed execution info: count={State.execution_count}, max={State.max_executions}, infinite={State.infinite_count}, license_expiry={State.license_expiry}, subscription_expiry={State.subscription_expiry}, blur_enabled={State.blur_enabled}")

if __name__ == "__main__":
    main()