import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import datetime
import pandas as pd
from PIL import Image
import time
from collections import OrderedDict
from ultralytics import YOLO
import face_recognition

# Set page config must be first Streamlit command
st.set_page_config(
    page_title="Video Surveillance And Tracking",
    page_icon="👤",
    layout="wide"
)

# Constants
KNOWN_FACES_DIR = "face_recog_project/known_faces"
UNKNOWN_FACES_DIR = "face_recog_project/unknown_faces"
LOG_FILE = "face_log.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Configuration
FACE_MATCHING_TOLERANCE = 0.5
MIN_HUMAN_CONFIDENCE = 0.7
FRAME_SKIP = 2
MAX_PROCESSING_FPS = 10
MAX_TRACKED_IDS = 50
TRACKING_DISTANCE_THRESHOLD = 60
MAX_TRACKING_TIME = 5

# Colors
HUMAN_BOX_COLOR = (0, 255, 0)  # Green for human detection
FACE_BOX_COLOR_KNOWN = (255, 0, 0)  # Blue for known faces
FACE_BOX_COLOR_UNKNOWN = (0, 0, 255)  # Red for unknown faces
TEXT_COLOR = (255, 255, 255)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = 0
if 'detected_names' not in st.session_state:
    st.session_state.detected_names = set()
if 'tracked_persons' not in st.session_state:
    st.session_state.tracked_persons = OrderedDict()
if 'next_person_id' not in st.session_state:
    st.session_state.next_person_id = 1
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = {}
if 'recognition_threshold' not in st.session_state:
    st.session_state.recognition_threshold = 0.6

# Initialize YOLOv8 model
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        return None

yolo_model = load_yolo_model()

# Load known face encodings
@st.cache_data
def load_known_faces():
    known_encodings = []
    known_names = []
    
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_encodings.extend(encodings)
                            known_names.extend([person_name] * len(encodings))
                    except Exception as e:
                        st.warning(f"Error processing {image_path}: {str(e)}")
    return known_encodings, known_names

def log_face(name, mode, person_id=None, image=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Safe CSV formatting
    safe_name = f'"{name}"' if "," in name else name
    person_id = person_id if person_id is not None else ""
    
    entry = f"{safe_name},{timestamp},{mode},{person_id}\n"
    
    if name == "Unknown" and image is not None:
        unknown_face_path = os.path.join(UNKNOWN_FACES_DIR, f"{timestamp.replace(':', '-')}_ID{person_id}.jpg")
        cv2.imwrite(unknown_face_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def track_persons(human_boxes, current_time):
    current_centers = [(x + w//2, y + h//2) for (x, y, w, h) in human_boxes]
    updated_persons = OrderedDict()
    used_ids = set()
    
    for i, (x, y, w, h) in enumerate(human_boxes):
        center = current_centers[i]
        min_distance = float('inf')
        matched_id = None
        
        for person_id, person_data in st.session_state.tracked_persons.items():
            if person_id in used_ids:
                continue
                
            distance = np.sqrt((center[0] - person_data['center'][0])**2 + 
                              (center[1] - person_data['center'][1])**2)
            
            if distance < TRACKING_DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                matched_id = person_id
        
        if matched_id is not None:
            person_id = matched_id
            used_ids.add(person_id)
        else:
            person_id = st.session_state.next_person_id
            st.session_state.next_person_id = 1 if st.session_state.next_person_id >= MAX_TRACKED_IDS else st.session_state.next_person_id + 1
        
        updated_persons[person_id] = {
            'center': center,
            'position': (x, y, w, h),
            'last_seen': current_time,
            'face_detected': False,
            'face_name': None,
            'face_location': None,
            'confidence': 0
        }
    
    # Clean up old entries
    for person_id in list(st.session_state.tracked_persons.keys()):
        if person_id not in updated_persons:
            if current_time - st.session_state.tracked_persons[person_id]['last_seen'] > MAX_TRACKING_TIME:
                del st.session_state.tracked_persons[person_id]
            else:
                updated_persons[person_id] = st.session_state.tracked_persons[person_id]
    
    st.session_state.tracked_persons = updated_persons
    return updated_persons

def detect_humans_yolo(frame):
    if yolo_model is None:
        return []
    
    results = yolo_model(frame, verbose=False)
    human_boxes = []
    
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > MIN_HUMAN_CONFIDENCE:  # Class 0 is person in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                human_boxes.append((x1, y1, x2-x1, y2-y1))
    
    return human_boxes

def process_frame(frame, known_encodings, known_names, mode="Live Camera"):
    current_time = time.time()
    
    if current_time - st.session_state.last_processed_time < 1/MAX_PROCESSING_FPS:
        return None
    
    st.session_state.last_processed_time = current_time
    
    # Stage 1: Human detection with YOLOv8
    human_boxes = detect_humans_yolo(frame)
    
    # Convert to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Track humans
    tracked_persons = track_persons(human_boxes, current_time)
    
    for person_id, person_data in tracked_persons.items():
        x, y, w, h = person_data['position']
        
        # Reset face detection status for this person
        st.session_state.tracked_persons[person_id]['face_detected'] = False
        st.session_state.tracked_persons[person_id]['face_name'] = None
        st.session_state.tracked_persons[person_id]['face_location'] = None
        st.session_state.tracked_persons[person_id]['confidence'] = 0
        
        # Stage 2: Face detection within human region
        human_roi = rgb_frame[y:y+h, x:x+w]
        
        # Find face locations
        face_locations = face_recognition.face_locations(human_roi)
        
        # Convert ROI to RGB format expected by face_recognition
        human_roi_rgb = cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB) if len(human_roi.shape) == 3 else cv2.cvtColor(cv2.cvtColor(human_roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(human_roi_rgb, face_locations)
        
        # Process each face found
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            abs_top = y + top
            abs_right = x + right
            abs_bottom = y + bottom
            abs_left = x + left
            
            # Face recognition
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=FACE_MATCHING_TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            confidence = 1.0  # Default for unknown

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    confidence = 1 - face_distances[best_match_index]
                    if confidence > st.session_state.recognition_threshold:
                        name = known_names[best_match_index]
            
            # Update person's face information
            st.session_state.tracked_persons[person_id]['face_detected'] = True
            st.session_state.tracked_persons[person_id]['face_name'] = name
            st.session_state.tracked_persons[person_id]['face_location'] = (abs_top, abs_right, abs_bottom, abs_left)
            st.session_state.tracked_persons[person_id]['confidence'] = confidence
            
            # Show detection message with confidence
            if name not in st.session_state.detected_names or (current_time - st.session_state.last_detection_time.get(name, 0) > 10):
                face_image = frame[abs_top:abs_bottom, abs_left:abs_right]
                log_face(name, mode, person_id, face_image)
                st.session_state.detected_names.add(name)
                st.session_state.last_detection_time[name] = current_time
                
                if name != "Unknown":
                    st.toast(f"👤 {name} (ID: {person_id}, Confidence: {confidence:.2f})", icon="✅")
                else:
                    st.toast(f"⚠️ Unknown person detected (ID: {person_id})", icon="❓")
    
    # Draw all boxes and labels after processing all faces
    for person_id, person_data in tracked_persons.items():
        x, y, w, h = person_data['position']
        face_detected = person_data['face_detected']
        name = person_data['face_name']
        face_location = person_data['face_location']
        confidence = person_data['confidence']
        
        if face_detected and face_location:
            # Face was detected - draw face box and label
            abs_top, abs_right, abs_bottom, abs_left = face_location
            face_color = FACE_BOX_COLOR_KNOWN if name != "Unknown" else FACE_BOX_COLOR_UNKNOWN
            cv2.rectangle(frame, (abs_left, abs_top), (abs_right, abs_bottom), face_color, 2)
            
            if name != "Unknown":
                label = f"ID:{person_id} {name} ({confidence:.2f})"
            else:
                label = f"ID:{person_id} Unknown"
                
            cv2.putText(frame, label, (abs_left, abs_bottom + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
            
            # Also draw human box but thinner
            cv2.rectangle(frame, (x, y), (x+w, y+h), HUMAN_BOX_COLOR, 1)
        else:
            # Only human detected - draw human box only
            cv2.rectangle(frame, (x, y), (x+w, y+h), HUMAN_BOX_COLOR, 2)
            cv2.putText(frame, f"Person {person_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUMAN_BOX_COLOR, 2)
    
    return frame

def process_video(video_path, known_encodings, known_names):
    video = cv2.VideoCapture(video_path)
    stframe = st.empty()
    progress_bar = st.progress(0)
    stop_button = st.button("Stop Processing")
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps if fps > 0 else 0.03
    
    st.session_state.detected_names = set()
    st.session_state.tracked_persons = OrderedDict()
    st.session_state.next_person_id = 1
    st.session_state.frame_count = 0
    st.session_state.last_detection_time = {}
    
    while video.isOpened() and not st.session_state.stop_requested and not stop_button:
        ret, frame = video.read()
        if not ret:
            break

        st.session_state.frame_count += 1
        
        if st.session_state.frame_count % FRAME_SKIP != 0:
            time.sleep(frame_delay)
            continue
            
        processed_frame = process_frame(frame, known_encodings, known_names, "Surveillance")
        
        if processed_frame is not None:
            stframe.image(processed_frame, channels="BGR", use_container_width=True)
            progress_bar.progress(st.session_state.frame_count / frame_count)
        
        time.sleep(frame_delay)
        
        if stop_button:
            st.session_state.stop_requested = True
            break

    video.release()
    progress_bar.empty()
    if st.session_state.stop_requested:
        st.warning("Video processing stopped")
    else:
        st.success("Video processing completed!")
    st.session_state.stop_requested = False

def run_live_camera(known_encodings, known_names):
    st.session_state.camera_active = True
    st.session_state.stop_requested = False
    st.session_state.detected_names = set()
    st.session_state.tracked_persons = OrderedDict()
    st.session_state.next_person_id = 1
    st.session_state.frame_count = 0
    st.session_state.last_detection_time = {}
    
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    stframe = st.empty()
    stop_button = st.button("Stop Camera")
    
    try:
        while video.isOpened() and not st.session_state.stop_requested and not stop_button:
            st.session_state.frame_count += 1
            
            ret, frame = video.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            if st.session_state.frame_count % FRAME_SKIP != 0:
                time.sleep(1/MAX_PROCESSING_FPS)
                continue
                
            processed_frame = process_frame(frame, known_encodings, known_names, "Live Camera")
            
            if processed_frame is not None:
                stframe.image(processed_frame, channels="BGR", use_container_width=True)
            
            time.sleep(1/MAX_PROCESSING_FPS)
            
            if stop_button:
                st.session_state.stop_requested = True
                break
    finally:
        video.release()
        st.session_state.camera_active = False
        if st.session_state.stop_requested:
            st.warning("Live camera stopped")
        else:
            st.warning("Live camera ended")

def add_new_face():
    st.subheader("Add New Face to Database")
    person_name = st.text_input("Enter person's name")
    uploaded_files = st.file_uploader("Upload face images (clear front view recommended)", 
                                    type=["jpg", "png", "jpeg"], 
                                    accept_multiple_files=True)
    
    if person_name and uploaded_files:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        success_count = 0
        
        with st.spinner("Processing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file)
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    image.thumbnail((500, 500))
                    
                    image_path = os.path.join(person_dir, f"{person_name}_{i}.jpg")
                    image.save(image_path, "JPEG", quality=90)
                    success_count += 1
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if success_count > 0:
            st.success(f"Added {success_count} images for {person_name} to the database")
            st.cache_data.clear()
        else:
            st.error("Failed to add any images")

def view_logbook():
    st.subheader("Detection Logbook")
    
    if os.path.exists(LOG_FILE):
        try:
            # Read CSV with proper handling of quoted fields
            log_df = pd.read_csv(LOG_FILE, 
                               header=None,
                               names=["Name", "Timestamp", "Mode", "PersonID"],
                               quotechar='"',
                               escapechar="\\",
                               on_bad_lines='warn')
            
            # Fix for NaTType error - handle invalid dates
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], errors='coerce')
            log_df = log_df.dropna().sort_values('Timestamp', ascending=False)
            
            st.dataframe(log_df, use_container_width=True)
            
            st.subheader("Filter Logs")
            col1, col2 = st.columns(2)
            
            with col1:
                name_filter = st.text_input("Filter by name")
            
            with col2:
                date_filter = st.date_input("Filter by date")
            
            if name_filter or date_filter:
                filtered_df = log_df.copy()
                if name_filter:
                    filtered_df = filtered_df[filtered_df['Name'].str.contains(name_filter, case=False)]
                if date_filter:
                    filtered_df = filtered_df[filtered_df['Timestamp'].dt.date == date_filter]
                st.dataframe(filtered_df, use_container_width=True)
            
            st.subheader("Manage Logs")
            with st.expander("Clear Log Data"):
                clear_log_data()
            
            if "Unknown" in log_df["Name"].values:
                st.subheader("Unknown Faces Detected")
                unknown_images = [f for f in os.listdir(UNKNOWN_FACES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if unknown_images:
                    cols = st.columns(3)
                    for i, img_file in enumerate(unknown_images):
                        img_path = os.path.join(UNKNOWN_FACES_DIR, img_file)
                        try:
                            cols[i%3].image(img_path, caption=img_file, use_container_width=True)
                            # Add delete button for each image
                            if cols[i%3].button(f"Delete {img_file}", key=f"del_{img_file}"):
                                os.unlink(img_path)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading {img_file}: {str(e)}")
                    
                    # Add option to delete all unknown faces
                    if st.button("Delete All Unknown Faces"):
                        for img_file in unknown_images:
                            img_path = os.path.join(UNKNOWN_FACES_DIR, img_file)
                            try:
                                os.unlink(img_path)
                            except Exception as e:
                                st.error(f"Error deleting {img_file}: {str(e)}")
                        st.rerun()
                else:
                    st.write("No unknown face images found")
            
            st.download_button(
                label="Download Logbook",
                data=log_df.to_csv(index=False).encode('utf-8'),
                file_name="detection_logbook.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error loading log file: {str(e)}")
            if st.button("Attempt to repair log file"):
                repair_log_file()
    else:
        st.write("No log entries found")

def repair_log_file():
    """Attempt to fix corrupted CSV"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Simple repair - keep only properly formatted lines
        repaired_lines = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:  # At least 4 columns
                # Reconstruct properly quoted name if needed
                if line.count('"') % 2 != 0:  # Unbalanced quotes
                    continue
                repaired_lines.append(line)
        
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.writelines(repaired_lines)
            
        st.success("Log file repaired! Please refresh.")
        st.rerun()
    except Exception as e:
        st.error(f"Repair failed: {str(e)}")

def clear_log_data():
    if os.path.exists(LOG_FILE):
        try:
            log_df = pd.read_csv(LOG_FILE, 
                                header=None,
                                names=["Name", "Timestamp", "Mode", "PersonID"],
                                quotechar='"',
                                escapechar="\\")
            
            # Fix for NaTType error - handle invalid dates
            log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], errors='coerce')
            log_df = log_df.dropna()
            
            min_date = log_df['Timestamp'].min().date()
            max_date = log_df['Timestamp'].max().date()
            
            st.write("### Clear Log Data by Date Range")
            date_range = st.date_input(
                "Select date range to clear",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if st.button("Clear Selected Log Data") and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
                
                # Delete all unknown face images within date range
                deleted_images = 0
                for filename in os.listdir(UNKNOWN_FACES_DIR):
                    file_path = os.path.join(UNKNOWN_FACES_DIR, filename)
                    try:
                        # Extract timestamp from filename (format: YYYY-MM-DD HH-MM-SS_IDX.jpg)
                        file_date_str = '_'.join(filename.split('_')[:2]).replace('-', ' ').split('.')[0]
                        file_date = pd.to_datetime(file_date_str, errors='coerce')
                        if file_date and start_date <= file_date <= end_date:
                            os.unlink(file_path)
                            deleted_images += 1
                    except Exception as e:
                        continue
                
                # Delete log entries within date range
                new_log_df = log_df[~((log_df['Timestamp'] >= start_date) & (log_df['Timestamp'] <= end_date))]
                deleted_entries = len(log_df) - len(new_log_df)
                
                # Save the cleaned log file
                new_log_df.to_csv(LOG_FILE, index=False, header=False)
                
                st.success(f"Permanently deleted {deleted_entries} log entries and {deleted_images} images")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing log file: {str(e)}")
            repair_log_file()

def main():
    st.title("👤 Smart Surveillance System")
    st.sidebar.title("Configuration")
    
    # Add recognition threshold slider to sidebar
    st.session_state.recognition_threshold = st.sidebar.slider(
        "Recognition Confidence Threshold",
        min_value=0.5, max_value=0.9, value=0.6, step=0.05
    )
    
    with st.spinner("Loading known faces..."):
        known_encodings, known_names = load_known_faces()
        st.sidebar.success(f"Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people")
    
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["📹 Live Camera", "🎥 Surveillance Mode", "➕ Add New Face", "📋 Logbook"]
    )
    
    if mode == "➕ Add New Face":
        add_new_face()
    elif mode == "📹 Live Camera":
        st.subheader("Live Camera Monitoring")
        if not st.session_state.camera_active:
            if st.button("Start Live Camera", type="primary"):
                run_live_camera(known_encodings, known_names)
        else:
            st.warning("Camera is already active in another tab/window")
    elif mode == "🎥 Surveillance Mode":
        st.subheader("Video File Processing")
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(video_file.read())
                video_path = tfile.name
            
            st.session_state.stop_requested = False
            process_video(video_path, known_encodings, known_names)
    elif mode == "📋 Logbook":
        view_logbook()

if __name__ == "__main__":
    main()