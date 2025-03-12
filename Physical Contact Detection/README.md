# Hand Contact Detection in Rooftop Videos

## Overview
This project analyzes hand contact between nurses and patients in recorded videos. The system utilizes computer vision techniques to detect when a nurse's hand enters a predefined patient area while excluding predefined exclusion zones.

## Features
- **Video Processing**: Processes multiple videos from a given directory.
- **Region Selection**: Allows manual selection of patient area and exclusion zones.
- **Hand Detection**: Uses MediaPipe to detect hands and track their positions.
- **Contact Analysis**: Detects and logs the duration of physical contact.
- **CSV Output**: Saves results including contact time, video duration, and contact percentage.

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
python Physical_Contact.py -d /path/to/videos -o output_folder
```
- `-d` (required): Directory containing videos.
- `-o` (optional): Output directory for analysis results.

## Output
Results are stored in a CSV file in the output folder, containing:
- Video name
- Contact duration (seconds)
- Total video duration
- Contact percentage
- Number of exclusion zones defined

Demonstration

To better understand the system, watch the demonstration video:

https://github.com/user-attachments/assets/88233860-748d-474a-90fe-9764844cfb39

https://github.com/user-attachments/assets/2c3b3132-8b33-40fb-9ce3-286aed13faf3



## Notes
- The script supports interactive region selection for better accuracy.
- Press `q` or `ESC` during video processing to exit.

## License
This project is for research purposes and follows the applicable data privacy policies.


