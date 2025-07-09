# Smart Surveillance System

A comprehensive video surveillance system that combines human detection with face recognition capabilities, built with Python, Streamlit, YOLOv8, and face_recognition libraries.

## Features

- **Real-time Human Detection**: Uses YOLOv8 for accurate human detection in video streams
- **Face Recognition**: Recognizes known faces and logs unknown detections
- **Multi-mode Operation**:
  - 📹 Live camera monitoring
  - 🎥 Video file processing
  - ➕ Add new faces to database
  - 📋 Comprehensive logbook system
- **Tracking System**: Assigns unique IDs to detected persons and tracks their movement
- **Configurable Thresholds**: Adjust recognition sensitivity and processing parameters
- **Log Management**: View, filter, and export detection logs with timestamps

## Technologies Used

- **Python 3.10+**
- **Streamlit** - Web application framework
- **OpenCV** - Video processing and image manipulation
- **YOLOv8** - Human detection model
- **face_recognition** - Face detection and recognition library
- **Pandas** - Log data management
- **Pillow** - Image processing

## Installation

1. Clone the repository:
 

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. install the required dependencies:
   pip install -r requirements.txt

4.Download the YOLOv8 model weights (will be automatically downloaded on first run if not present)

## Usage

1.Run the application:
  streamlit run video_surveillance_app.py

2.The application will open in your default browser at http://localhost:8501

3.Use the sidebar to navigate between different modes:

  3.1 Live Camera: Real-time monitoring from your webcam

  3.2 Surveillance Mode: Process recorded video files

  3.3 Add New Face: Register new faces to the recognition database

  3.4 Logbook: View and manage detection logs if a unkonwn person detected along with image and timestamp with id the person will logged in the logbook.

#Configuration
 Key configuration options available in the sidebar:

  1.Recognition confidence threshold (0.5-0.9)

  2.Processing frame rate

  3.Tracking parameters


## File Structure

```text
smart-surveillance-system/
├── video_surveillance_app.py      # Main Streamlit application
├── requirements.txt               # Python dependencies
├── face_recog_project/            # Auto-created on first run
│   ├── known_faces/               # Database of authorized individuals
│   │   ├── john_doe/              # Individual's folder (name format: first_last)
│   │   │   ├── john1.jpg          # 10-20+ images per person
│   │   │   ├── john2.jpg          # (clear frontal views recommended)
│   │   │   └── ...                # Supports .jpg, .png formats
│   │   ├── jane_smith/
│   │   │   ├── jane1.jpg
│   │   │   └── ...
│   │   └── ...                    # Add more individuals as needed
│   └── unknown_faces/             # Auto-stores unrecognized detections
│       └── YYYY-MM-DD_HH-MM-SS_ID#.jpg  # Auto-naming format
├── face_log.csv                   # Timestamped detection records
└── README.md                      # Project documentation
```

## Dataset

once you run the file using streamlit the face_recof_project will be created automatically and inside that there two sub folder will also created
- **known_faces** = - To add the person which is known to you can add the person name along with there picture (10-20 or more) directly or can be added using the streamlit dashboard
- **unknown_faces** = - While detection it Auto-stores unrecognized detections
