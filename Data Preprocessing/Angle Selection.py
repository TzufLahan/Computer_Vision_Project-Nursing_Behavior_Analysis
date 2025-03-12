import cv2
import os
from glob import glob
import numpy as np
from math import degrees, atan2


def click_event(event, x, y, flags, params):
    """Handle mouse clicks to collect points"""
    img, points, window_name = params
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw circle and number at clicked point
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(img, str(len(points) + 1), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        points.append((x, y))
        cv2.imshow(window_name, img)


def calculate_rotation_angle(p1, p2):
    """Calculate rotation angle from two points"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = degrees(atan2(dy, dx))
    return angle


def rotate_frame(frame, angle, center=None):
    """Rotate frame by given angle"""
    height, width = frame.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height),
                                   flags=cv2.INTER_LINEAR)
    return rotated_frame


def resize_frame_to_fit_screen(frame, max_width=1150, max_height=800):
    """Resize frame if it's larger than specified dimensions"""
    height, width = frame.shape[:2]
    width_ratio = max_width / width if width > max_width else 1
    height_ratio = max_height / height if height > max_height else 1
    scale_ratio = min(width_ratio, height_ratio)

    if scale_ratio < 1:
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return frame


def test_rotation_angles():
    # Get the base folder path
    base_folder = r'C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\סרטים ניסוי'
    angles_dict = {}

    # For each Face Angle folder
    for angle_folder in glob(os.path.join(base_folder, "Face Angle *")):
        angle_folder_name = os.path.basename(angle_folder)
        print(f"\nTesting {angle_folder_name}")

        # Find first video in the folder structure
        face_videos = []
        for root, dirs, files in os.walk(angle_folder):
            for file in files:
                if file.startswith("Face") and file.endswith(".mp4"):
                    face_videos.append(os.path.join(root, file))
                    break
            if face_videos:
                break

        if not face_videos:
            print(f"No Face videos found in {angle_folder_name}")
            continue

        # Open the first video
        video_path = face_videos[0]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        # Get a frame from the middle of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Could not read frame from video: {video_path}")
            continue

        proceed_to_next = False
        while not proceed_to_next:
            # Create window for this folder
            window_name = f"Rotation Test - {angle_folder_name}"
            frame_copy = frame.copy()
            points = []

            # Create window and set mouse callback
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, click_event, (frame_copy, points, window_name))

            print(f"\nTesting {angle_folder_name}")
            print("Click two points to define the horizontal line")
            print("Press 'r' to reset points")
            print("Press 'c' to confirm and see rotation")
            print("Press 'q' to skip to next folder")

            while not proceed_to_next:
                cv2.imshow(window_name, frame_copy)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):  # Reset
                    frame_copy = frame.copy()
                    points = []
                    cv2.imshow(window_name, frame_copy)

                elif key == ord('c') and len(points) == 2:  # Confirm and show rotation
                    angle = calculate_rotation_angle(points[0], points[1])
                    rotated = rotate_frame(frame.copy(), angle)
                    preview_name = f"Preview - {angle_folder_name}"

                    while True:
                        display_frame = resize_frame_to_fit_screen(rotated)
                        cv2.putText(display_frame, f"Angle: {angle:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(preview_name, display_frame)

                        key2 = cv2.waitKey(1) & 0xFF
                        if key2 == ord('s'):  # Save angle and proceed to next folder
                            print(f"Selected angle for {angle_folder_name}: {angle:.1f}")
                            angles_dict[angle_folder_name] = angle
                            cv2.destroyAllWindows()
                            proceed_to_next = True
                            break
                        elif key2 == ord('r'):  # Retry
                            cv2.destroyWindow(preview_name)
                            break
                        elif key2 == ord('q'):  # Skip to next folder
                            cv2.destroyAllWindows()
                            proceed_to_next = True
                            break

                elif key == ord('q'):  # Skip to next folder
                    cv2.destroyAllWindows()
                    proceed_to_next = True
                    break

    return angles_dict


if __name__ == "__main__":
    print("Rotation Angle Tester")
    print("--------------------")
    print("Instructions:")
    print("1. For each Face Angle folder, a window will open showing a frame from a video")
    print("2. Click two points to define the horizontal line")
    print("3. Press 'c' to confirm and see the rotation preview")
    print("4. In the preview window:")
    print("   - Press 's' to save the angle and continue to next folder")
    print("   - Press 'r' to retry point selection")
    print("   - Press 'q' to quit")
    print("--------------------")

    angles = test_rotation_angles()

    print("\nFinal Results:")
    print("-------------")
    for folder, angle in angles.items():
        print(f"{folder}: {angle:.1f}°")

    print("\nYou can use these angles in your get_rotation_angle_from_folder function:")
