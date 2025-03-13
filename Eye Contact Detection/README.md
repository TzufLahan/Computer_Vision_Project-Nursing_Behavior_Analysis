# Eye Contact Detection in Videos

## Overview
This Section analyzes eye contact between nurses and patients in recorded videos. Using computer vision techniques, the system detects when a nurse makes eye contact with a patient based on gaze direction and face alignment.

## How It Works
The system processes video frames using MediaPipe to detect facial landmarks. It calculates gaze direction using key points from the eyes and nose, then determines whether the gaze aligns with the selected patient region. The script tracks the duration of eye contact and logs the results in a CSV file for further analysis.

## Features
- **Video Processing**: Processes multiple videos in a given directory.
- **Patient Region Selection**: Allows manual selection of the patient's head region.
- **Eye Contact Detection**: Uses MediaPipe to detect face landmarks and gaze vectors.
- **Tracking & Analysis**: Logs eye contact duration and calculates the percentage of time the nurse maintains eye contact.
- **CSV Output**: Saves results, including total eye contact time and percentage of video duration.

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

Install dependencies using:
```sh
pip install opencv-python mediapipe numpy
```

## Usage
Run the script with:
```sh
python Eye_Contact.py -d /path/to/videos -o output_folder
```
- `-d` (required): Directory containing videos.
- `-o` (optional): Output directory for analysis results.

## Output
Results are stored in a CSV file in the output folder, containing:
- Video name
- Eye contact duration (seconds)
- Total video duration
- Eye contact percentage

## Demonstration

To better understand the system, watch the demonstration video:

Demonstration video of Eye contact

https://github.com/user-attachments/assets/5c2840de-1b42-4473-bcce-2321ac52b2f0

Demonstration Video of No Eye Contact

https://github.com/user-attachments/assets/7841e501-b5d5-4be1-b8e4-9750ee82edd8

## License
This project is for research purposes and follows the applicable data privacy policies.

