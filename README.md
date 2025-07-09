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

# Usage

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

 ##Recognition confidence threshold (0.5-0.9)

##Processing frame rate

##Tracking parameters


#File Structure


smart-surveillance-system/
├──   video_surveillance_app.py                  # Main application file
├── requirements.txt                             # Python dependencies
├── face_recog_project/                          # automatically create once you run the file
│   ├── known_faces/                             # Directory for registered faces (automatically create once you run the file)
│   └── unknown_faces/                           # Directory for unrecognized faces (automatically create once you run the file)
├── face_log.csv                                 # Detection log file (automatically create once you run the file)
└── README.md                                    # This file

#Dataset
once you run the file using streamlit the face_recof_project will be created automatically and inside that there two sub folder will also created
1.known_faces = To add the person which is known to you can add the person name along with there picture (10-20 or more) directly or can be added using the streamlit dashboard
2.unknown_faces

