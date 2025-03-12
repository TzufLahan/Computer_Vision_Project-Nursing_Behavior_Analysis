import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan2
import os
from glob import glob


def rotate_frame(frame, angle, center=None):
    """Rotate frame by given angle"""
    height, width = frame.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height),
                                   flags=cv2.INTER_LINEAR)
    return rotated_frame


def calculate_head_angle(results):
    """Calculate head angle using facial landmarks"""
    if not results.pose_landmarks or not results.face_landmarks:
        return None

    nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    left_eye = results.face_landmarks.landmark[33]
    right_eye = results.face_landmarks.landmark[263]
    right_ear = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
    left_ear = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]

    if nose.x <= right_ear.x or nose.x >= right_eye.x or nose.x >= left_ear.x:
        return None

    head_angle = degrees(atan2(
        nose.y - (left_eye.y + right_eye.y) / 2,
        nose.x - (left_eye.x + right_eye.x) / 2
    ))

    return abs(head_angle)


def calibrate_thresholds():
    base_folder = r'C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\סרטים ניסוי'
    thresholds = {}

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Process each Face Angle folder
        for angle_folder in glob(os.path.join(base_folder, "Face Angle *")):
            angle_folder_name = os.path.basename(angle_folder)
            print(f"\nCalibrating {angle_folder_name}")

            # Find first Face video
            face_videos = []
            for root, dirs, files in os.walk(angle_folder):
                for file in files:
                    if file.startswith("Face") and file.endswith(".mp4"):
                        face_videos.append(os.path.join(root, file))
                        break
                if face_videos:
                    break

            if not face_videos:
                continue

            video_path = face_videos[0]
            cap = cv2.VideoCapture(video_path)

            # Set starting frame
            start_frame = 3000  # You can adjust this
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Create trackbars window
            window_name = f'Calibration - {angle_folder_name}'
            cv2.namedWindow(window_name)
            cv2.createTrackbar('Lower Bound', window_name, 60, 180, lambda x: None)
            cv2.createTrackbar('Upper Bound', window_name, 120, 180, lambda x: None)

            # Get rotation angle for this folder
            rotation_angles = {
                'face angle 1': -56.2,
                'face angle 2': 3.4,
                'face angle 3': -8.1,
                'face angle 4': -18.3,
                'face angle 5': -18.3,
                'face angle 6': -2.8,
                'face angle 7': -11.3,
            }
            rotation_angle = rotation_angles.get(angle_folder_name.lower(), 0)

            print("\nInstructions:")
            print("1. Adjust the trackbars to set lower and upper bounds")
            print("2. Green text means eye contact detected")
            print("3. Press 's' to save thresholds and move to next folder")
            print("4. Press 'q' to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    continue

                # Get current threshold values
                lower = cv2.getTrackbarPos('Lower Bound', window_name)
                upper = cv2.getTrackbarPos('Upper Bound', window_name)

                # Rotate and process frame
                rotated_frame = rotate_frame(frame, rotation_angle)
                image_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                # Calculate head angle
                angle = calculate_head_angle(results)

                # Draw landmarks
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Display angle and eye contact status
                if angle is not None:
                    eye_contact = lower < angle < upper
                    color = (0, 255, 0) if eye_contact else (0, 0, 255)
                    cv2.putText(image_bgr, f"Angle: {angle:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(image_bgr, f"Eye Contact: {eye_contact}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(image_bgr, f"Bounds: {lower} - {upper}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow(window_name, image_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):  # Save thresholds
                    thresholds[angle_folder_name.lower()] = (lower, upper)
                    break
                elif key == ord('q'):  # Quit
                    cap.release()
                    cv2.destroyAllWindows()
                    return thresholds

            cap.release()
            cv2.destroyWindow(window_name)

    return thresholds


if __name__ == "__main__":
    print("Eye Contact Threshold Calibration")
    print("--------------------------------")

    thresholds = calibrate_thresholds()

    print("\nCalibrated Thresholds:")
    print("angle_thresholds = {")
    for folder, (lower, upper) in thresholds.items():
        print(f"    '{folder}': ({lower}, {upper}),")
    print("}")