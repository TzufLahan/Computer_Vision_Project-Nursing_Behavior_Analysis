# Facial Emotion Recognition in Videos

## Overview
This section detects facial emotions, with a focus on **happiness detection**, in recorded videos. The system uses deep learning-based facial expression recognition to analyze emotions frame by frame and log happiness duration.

## How It Works
The system processes video frames using the **FER (Facial Expression Recognition) library**, detecting emotions in real-time. It focuses on the happiness score for each detected face, tracking how long a person maintains a happy expression. The results are stored in a CSV file for further analysis.

## Features
- **Video Processing**: Processes multiple videos in a given directory.
- **Facial Emotion Detection**: Uses FER to detect emotions from facial expressions.
- **Happiness Tracking**: Measures happiness score and logs the duration.
- **Threshold Analysis**: Only considers a frame as "happy" if its happiness score exceeds a predefined threshold.
- **CSV Output**: Saves results, including total happy time and percentage of video duration.

## Requirements
- Python 3.7+
- OpenCV
- FER
- NumPy
- Pandas

Install dependencies using:
```sh
pip install opencv-python fer numpy pandas
```

## Usage
Run the script with:
```sh
python Face_Emotions_Recognition.py -d /path/to/videos -o output_folder
```
- `-d` (required): Directory containing videos.
- `-o` (optional): Output directory for analysis results.

## Output
Results are stored in a CSV file in the output folder, containing:
- Video name
- Happiness duration (seconds)
- Total video duration
- Percentage of time happy

Demonstration

To better understand the system, watch the demonstration video:

[https://github.com/user-attachments/assets/88233860-748d-474a-90fe-9764844cfb39](https://github.com/user-attachments/assets/112aad8e-216b-4e52-8725-7c7883da9986)

## License
This project is for research purposes and follows the applicable data privacy policies.

