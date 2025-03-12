# Automated Empathy Analysis in Nursing Simulations

## Overview
This project aims to analyze **empathy in nursing simulations** by automatically detecting key behavioral indicators, including **eye contact, physical touch, facial expressions**, and **motion patterns**. Using **computer vision and machine learning techniques**, the system processes video recordings to assess empathetic behaviors in a structured and quantifiable manner.

## Motivation
Empathy is a crucial aspect of **high-quality patient care**, significantly impacting **patient satisfaction and medical outcomes**. Traditional assessment methods rely on subjective evaluation, which lacks consistency and scalability. This project introduces an **automated framework** to enhance the assessment and training of nursing students, providing objective and data-driven insights into their empathetic behavior.

## Project Structure
The repository consists of multiple components, each focusing on a different aspect of empathy detection:

- **📂 Data Preprocessing**: Handles video alignment, trimming, and rotation corrections to ensure consistency across different simulation recordings.
- **👀 Eye Contact Detection**: Tracks **gaze direction** and determines if the nurse maintains eye contact with the patient using **MediaPipe Holistic**.
- **😊 Face Emotion Recognition (FER)**: Detects and quantifies **happiness levels** using deep learning models to assess **positive engagement** with the patient.
- **🔥 Heatmap Motion Tracker**: Maps **nurse movement patterns** to analyze their **proximity and engagement** with the patient.
- **✋ Physical Contact Detection**: Identifies instances of **hand contact** between the nurse and patient, measuring interaction time and intensity.

## How It Works
1. **Preprocessing**: Videos undergo **alignment, rotation correction, and trimming** to ensure consistency.
2. **Computer Vision Models**:
   - **Gaze tracking** determines eye contact with the patient.
   - **Facial expression analysis** quantifies the presence and duration of smiling.
   - **Motion heatmaps** visualize nurse movement across the simulation room.
   - **Hand detection** logs instances of physical contact.
3. **Data Output**:
   - Results are stored in structured **CSV files**.
   - Reports include **eye contact duration, happiness percentage, movement heatmaps, and physical contact analysis**.

## Technologies Used
- **Python 3.7+**
- **OpenCV** – Video processing and computer vision tasks
- **MediaPipe** – Facial and hand tracking
- **FER (Facial Expression Recognition)** – Emotion classification
- **NumPy & Pandas** – Data processing and analysis

## Installation
To set up the project, install the required dependencies:
```sh
pip install opencv-python mediapipe fer numpy pandas
```

## Running the System
Run each module separately to analyze different empathy indicators:
```sh
python Eye_Contact.py -d /path/to/videos -o output_folder
python Face_Emotions_Recognition.py -d /path/to/videos -o output_folder
python Heatmap_Tracker.py -d /path/to/videos -o output_folder
python Physical_Contact.py -d /path/to/videos -o output_folder
```

## Output
Each module generates a **CSV file** summarizing the results, including:
- **Eye contact duration** and percentage
- **Happiness scores** and smiling duration
- **Motion heatmaps** highlighting nurse movement
- **Physical contact occurrences** and total interaction time


## License
This project is for research purposes and follows the applicable data privacy policies.

