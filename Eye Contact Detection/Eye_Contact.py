import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from Angles_Per_Video import get_rotation_angle_from_folder, rotate_frame
import csv
import datetime

class EyeContactDetector:
    def __init__(self,
                 min_detection_confidence=0.3,
                 min_tracking_confidence=0.3,
                 frame_skip=1,
                 resize_width=640):
        """
        Initialize eye contact detector with performance optimizations

        :param min_detection_confidence: Minimum confidence for detection
        :param min_tracking_confidence: Minimum confidence for tracking
        :param frame_skip: Number of frames to skip between processing
        :param resize_width: Width to resize frames for faster processing
        """
        # MediaPipe initialization
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Detector parameters
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Performance optimization parameters
        self.frame_skip = frame_skip
        self.resize_width = resize_width

        # Holistic model with optimized parameters
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            refine_face_landmarks=True,  # More precise face landmarks
            static_image_mode=False  # Tracking mode for better performance
        )

    def select_patient_head_region(self, frame):
        """
        Advanced patient head region selection with circular approach

        Features:
        - Interactive circle selection
        - Precise center and radius definition
        """
        region = {
            'center': None,
            'radius': 0,
            'drawing': False,
            'completed': False
        }
        frame_copy = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing circle
                region['center'] = (x, y)
                region['drawing'] = True
                region['radius'] = 0

            elif event == cv2.EVENT_MOUSEMOVE and region['drawing']:
                # Update frame and draw current circle
                temp_frame = frame_copy.copy()

                # Calculate radius
                region['radius'] = int(np.sqrt(
                    (x - region['center'][0]) ** 2 +
                    (y - region['center'][1]) ** 2
                ))

                # Draw temporary circle
                cv2.circle(temp_frame,
                           region['center'],
                           region['radius'],
                           (0, 255, 0),
                           2)

                cv2.imshow('Select Patient Head Region', temp_frame)

            elif event == cv2.EVENT_LBUTTONUP:
                # Finalize circle selection
                region['drawing'] = False
                region['completed'] = True

                # Ensure minimum radius
                region['radius'] = max(region['radius'], 20)

        # Create window and set callback
        cv2.namedWindow('Select Patient Head Region', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Patient Head Region', mouse_callback)

        # Instructions
        instructions = [
            "Patient Head Region Selection",
            "Left Click & Drag: Select circular region",
            "Press 'c': Confirm selection",
            "Press 'r': Reset selection",
            "Press 'ESC': Exit"
        ]

        # Add instructions to frame
        for i, text in enumerate(instructions):
            cv2.putText(frame_copy, text,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        while True:
            cv2.imshow('Select Patient Head Region', frame_copy)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and region['completed']:
                # Create circular mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(mask,
                           region['center'],
                           region['radius'],
                           255,
                           -1)  # Filled circle

                # Calculate bounding box of circle
                x1 = max(0, region['center'][0] - region['radius'])
                y1 = max(0, region['center'][1] - region['radius'])
                x2 = min(frame.shape[1], region['center'][0] + region['radius'])
                y2 = min(frame.shape[0], region['center'][1] + region['radius'])

                cv2.destroyWindow('Select Patient Head Region')

                # Return bounding box, mask, and circle parameters
                return (x1, y1, x2, y2), mask, {
                    'center': region['center'],
                    'radius': region['radius']
                }

            elif key == ord('r'):
                # Reset selection
                region['center'] = None
                region['radius'] = 0
                region['drawing'] = False
                region['completed'] = False
                frame_copy = frame.copy()

                # Restore instructions
                for i, text in enumerate(instructions):
                    cv2.putText(frame_copy, text,
                                (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)

            elif key == 27:  # ESC key
                cv2.destroyWindow('Select Patient Head Region')
                return None

    def preprocess_frame(self, frame, rotation_angle):
        """
        Optimized frame preprocessing

        :param frame: Input frame
        :param rotation_angle: Rotation angle
        :return: Preprocessed frame
        """
        # Rotate frame
        rotated_frame = rotate_frame(frame, rotation_angle)

        # Resize for faster processing
        h, w = rotated_frame.shape[:2]
        aspect_ratio = h / w
        new_height = int(self.resize_width * aspect_ratio)
        resized_frame = cv2.resize(rotated_frame,
                                   (self.resize_width, new_height),
                                   interpolation=cv2.INTER_AREA)

        return resized_frame

    def calculate_gaze_vector(self, face_landmarks, frame_shape):
        """
        Advanced 3D gaze vector calculation with head pose estimation

        :param face_landmarks: MediaPipe face landmarks
        :param frame_shape: Shape of the input frame
        :return: Eye center, gaze vector, nose tip, and head rotation angles
        """
        try:
            # Define key facial landmarks for 3D pose estimation
            landmarks_indices = [
                33,  # Left eye outer
                263,  # Right eye outer
                1,  # Nose tip
                61,  # Left mouth corner
                291,  # Right mouth corner
                199,  # Forehead
                417,  # Chin
            ]

            # Extract frame dimensions
            img_h, img_w = frame_shape[:2]

            # Prepare 2D and 3D landmark coordinates
            face_2d = []
            face_3d = []

            for idx in landmarks_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            # Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix estimation
            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])

            # Distortion matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP for head pose
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, _ = cv2.Rodrigues(rot_vec)

            # Get rotation angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Convert to degrees
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            # Define key eye and nose landmarks
            LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263
            RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 133, 33
            NOSE_TIP = 1

            # Extract specific landmarks
            left_inner = np.array([
                face_landmarks.landmark[LEFT_EYE_INNER].x,
                face_landmarks.landmark[LEFT_EYE_INNER].y,
                face_landmarks.landmark[LEFT_EYE_INNER].z
            ])
            left_outer = np.array([
                face_landmarks.landmark[LEFT_EYE_OUTER].x,
                face_landmarks.landmark[LEFT_EYE_OUTER].y,
                face_landmarks.landmark[LEFT_EYE_OUTER].z
            ])
            right_inner = np.array([
                face_landmarks.landmark[RIGHT_EYE_INNER].x,
                face_landmarks.landmark[RIGHT_EYE_INNER].y,
                face_landmarks.landmark[RIGHT_EYE_INNER].z
            ])
            right_outer = np.array([
                face_landmarks.landmark[RIGHT_EYE_OUTER].x,
                face_landmarks.landmark[RIGHT_EYE_OUTER].y,
                face_landmarks.landmark[RIGHT_EYE_OUTER].z
            ])
            nose_tip = np.array([
                face_landmarks.landmark[NOSE_TIP].x,
                face_landmarks.landmark[NOSE_TIP].y,
                face_landmarks.landmark[NOSE_TIP].z
            ])

            # Calculate eye centers
            left_eye_center = (left_inner + left_outer) / 2
            right_eye_center = (right_inner + right_outer) / 2
            eye_center_3d = (left_eye_center + right_eye_center) / 2

            # Calculate gaze vector with head rotation consideration
            left_eye_vec = left_outer - left_inner
            right_eye_vec = right_outer - right_inner
            gaze_vector_3d = (left_eye_vec + right_eye_vec) / 2

            # Adjust vector with nose tip and head rotation
            forward_offset = nose_tip - eye_center_3d
            gaze_vector_3d = gaze_vector_3d + forward_offset * 0.5

            # Apply rotation correction
            rotation_matrix = cv2.Rodrigues(rot_vec)[0]
            gaze_vector_3d = np.dot(rotation_matrix, gaze_vector_3d)

            # Normalize vector
            gaze_vector_3d /= np.linalg.norm(gaze_vector_3d)

            return (
                eye_center_3d,
                gaze_vector_3d,
                nose_tip,
                {
                    'x_angle': x_angle,
                    'y_angle': y_angle,
                    'z_angle': z_angle
                }
            )

        except Exception as e:
            print(f"Advanced gaze vector calculation error: {e}")
            return None, None, None, None

    def check_gaze_intersection(self, eye_center, gaze_vector, patient_region, frame_shape, patient_circle_params=None):
        """
        Check if gaze intersects with patient region
        Supports both rectangular and circular regions

        :param eye_center: Eye center coordinates
        :param gaze_vector: Gaze vector
        :param patient_region: Patient region coordinates
        :param frame_shape: Frame dimensions
        :param patient_circle_params: Optional circle parameters
        :return: Boolean indicating intersection
        """
        h, w = frame_shape[:2]
        start_x = int(eye_center[0] * w)
        start_y = int(eye_center[1] * h)

        gaze_length = 2.0
        end_x = int((eye_center[0] + gaze_vector[0] * gaze_length) * w)
        end_y = int((eye_center[1] + gaze_vector[1] * gaze_length) * h)

        # If circle parameters are provided, use circular intersection
        if patient_circle_params:
            circle_center = patient_circle_params['center']
            circle_radius = patient_circle_params['radius']

            # Calculate distance from line to circle center
            def point_line_distance(point, line_start, line_end):
                x0, y0 = point
                x1, y1 = line_start
                x2, y2 = line_end

                numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
                denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                return numerator / denominator

            # Check if gaze line is close enough to circle center
            dist = point_line_distance(circle_center, (start_x, start_y), (end_x, end_y))
            return dist <= circle_radius

        # Fallback to rectangular region if no circle parameters
        x1, y1, x2, y2 = patient_region

        def line_intersection(pt1, pt2, pt3, pt4):
            x1, y1 = pt1
            x2, y2 = pt2
            x3, y3 = pt3
            x4, y4 = pt4

            denom = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
            if denom == 0:
                return False

            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

            return 0 <= ua <= 1 and 0 <= ub <= 1

        # Check intersection with rectangle edges
        edges = [
            ((x1, y1), (x2, y1)),  # Top
            ((x2, y1), (x2, y2)),  # Right
            ((x2, y2), (x1, y2)),  # Bottom
            ((x1, y2), (x1, y1))  # Left
        ]

        for edge in edges:
            if line_intersection((start_x, start_y), (end_x, end_y), *edge):
                return True

        return False

    def display_frame_info(self, frame, eye_contact, total_frames, frame_count, frame_rate,
                       results, patient_region=None, eye_center_3d=None, gaze_vector_3d=None,
                       show_landmarks=False, circle_params=None):
        """
        Display detailed information with enhanced, stylish gaze vector visualization

        :param show_landmarks: Toggle to show/hide body landmarks
        """
        # Conditionally draw pose landmarks
        if show_landmarks and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw patient region
        if circle_params:
            # Draw circle if circle parameters are available
            cv2.circle(frame,
                       circle_params['center'],
                       circle_params['radius'],
                       (255, 0, 0),  # Blue color
                       2)  # 2-pixel thickness
        elif patient_region:
            # Fallback to rectangle if no circle params
            cv2.rectangle(frame,
                          (patient_region[0], patient_region[1]),
                          (patient_region[2], patient_region[3]),
                          (255, 0, 0), 2)

        # Draw 3D gaze vector if available
        if eye_center_3d is not None and gaze_vector_3d is not None:
            h, w = frame.shape[:2]
            start_point = (int(eye_center_3d[0] * w), int(eye_center_3d[1] * h))

            # Increase depth scale for longer vector
            depth_scale = 0.6  # Increased for a more pronounced vector

            # Calculate end point with increased depth perspective
            end_point = (
                int((eye_center_3d[0] + gaze_vector_3d[0] * depth_scale) * w),
                int((eye_center_3d[1] + gaze_vector_3d[1] * depth_scale) * h)
            )

            # Stylish vector colors
            if eye_contact:
                base_color = (50, 205, 50)  # Lime Green
                tip_color = (0, 255, 127)  # Spring Green
            else:
                base_color = (220, 20, 60)  # Crimson
                tip_color = (255, 99, 71)  # Tomato Red

            # Calculate vector angle and length
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            vector_length = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.arctan2(dy, dx)

            # Draw gradient vector
            num_segments = 20
            for i in range(num_segments):
                t = i / num_segments
                start = (
                    int(start_point[0] + dx * t),
                    int(start_point[1] + dy * t)
                )
                end = (
                    int(start_point[0] + dx * (t + 1 / num_segments)),
                    int(start_point[1] + dy * (t + 1 / num_segments))
                )

                # Interpolate color
                segment_color = tuple(
                    int(base_color[j] * (1 - t) + tip_color[j] * t)
                    for j in range(3)
                )

                cv2.line(frame, start, end, segment_color, 3)

            # Arrowhead (more pronounced)
            arrow_length = min(30, vector_length * 0.2)  # Adaptive arrowhead size
            arrow_angle = np.pi / 6  # 30-degree spread

            # Arrowhead points
            pt1 = (
                int(end_point[0] - arrow_length * np.cos(angle + arrow_angle)),
                int(end_point[1] - arrow_length * np.sin(angle + arrow_angle))
            )
            pt2 = (
                int(end_point[0] - arrow_length * np.cos(angle - arrow_angle)),
                int(end_point[1] - arrow_length * np.sin(angle - arrow_angle))
            )

            # Draw arrowhead with gradient
            cv2.line(frame, end_point, pt1, tip_color, 4)
            cv2.line(frame, end_point, pt2, tip_color, 4)

        # Status text with improved styling
        status_text = f"Eye Contact: {'Yes' if eye_contact else 'No'}"
        cv2.putText(frame, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if eye_contact else (0, 0, 255),
                    2)

        # Duration text
        duration = total_frames / frame_rate
        cv2.putText(frame, f"Total Eye Contact: {duration:.2f} seconds",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Time text
        video_time = frame_count / frame_rate
        cv2.putText(frame, f"Video Time: {video_time:.2f} seconds",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def process_video_with_display(self, video_path, rotation_angle, start_frame=0, max_frames=None, min_eye_contact_frames=10):
        """
        Process video with robust, continuous processing even without face detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        # Get frame rate
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        # Reset to starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        eye_contact_start_frame = None
        total_eye_contact_frames = 0
        current_eye_contact_frames = 0
        frame_count = 0
        eye_contact_history = []
        SMOOTHING_WINDOW = 5

        # Patient region selection (first frame)
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            cap.release()
            return None

        # Preprocess first frame
        display_first_frame = self.preprocess_frame(first_frame, rotation_angle)

        # Select patient region for eye contact detection
        selection_result = self.select_patient_head_region(display_first_frame)
        if selection_result is None:
            print("Region selection cancelled")
            cap.release()
            return None

        patient_region, region_mask, circle_params = selection_result

        # Reset video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            while cap.isOpened():
                ret = cap.grab()
                if not ret:
                    break

                # Process only every nth frame
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                # Read frame
                ret, frame = cap.retrieve()
                if not ret:
                    break

                processed_frame = self.preprocess_frame(frame, rotation_angle)

                image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                results = self.holistic.process(image_rgb)

                eye_contact = False
                eye_center_3d = None
                gaze_vector_3d = None

                # Check for face landmarks
                if results.face_landmarks and results.pose_landmarks:
                    try:
                        # Get landmark references for ear-nose condition
                        nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                        left_eye = results.face_landmarks.landmark[33]  # Left eye
                        right_eye = results.face_landmarks.landmark[263]  # Right eye
                        right_ear = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
                        left_ear = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EAR]

                        # Check ear-nose condition
                        ear_nose_condition = not (
                                nose.x <= right_ear.x or
                                nose.x >= right_eye.x or
                                nose.x >= left_ear.x or
                                nose.x <= left_eye.x
                        )
                        # Calculate gaze vector
                        eye_center_3d, gaze_vector_3d, nose_tip, head_angles = self.calculate_gaze_vector(
                            results.face_landmarks,
                            processed_frame.shape
                        )

                        if eye_center_3d is not None and gaze_vector_3d is not None:
                            # Check gaze intersection and ear-nose condition
                            gaze_intersection = self.check_gaze_intersection(
                                eye_center_3d,
                                gaze_vector_3d,
                                patient_region,
                                processed_frame.shape,
                                circle_params
                            )

                            # Combine both conditions
                            eye_contact = gaze_intersection and ear_nose_condition

                    except Exception as detection_error:
                        print(f"Eye contact detection error: {detection_error}")
                        eye_contact = False

                # Smoothing eye contact detection
                eye_contact_history.append(eye_contact)
                if len(eye_contact_history) > SMOOTHING_WINDOW:
                    eye_contact_history.pop(0)
                smoothed_eye_contact = sum(eye_contact_history) > len(eye_contact_history) / 2

                # Track eye contact duration with minimum frame threshold
                if smoothed_eye_contact:
                    current_eye_contact_frames += 1

                    # Start tracking if minimum frames threshold is met
                    if current_eye_contact_frames >= min_eye_contact_frames:
                        if eye_contact_start_frame is None:
                            eye_contact_start_frame = frame_count
                else:
                    # Reset consecutive eye contact frames if no eye contact
                    if current_eye_contact_frames >= min_eye_contact_frames and eye_contact_start_frame is not None:
                        total_eye_contact_frames += current_eye_contact_frames

                    current_eye_contact_frames = 0
                    eye_contact_start_frame = None

                # Convert back to BGR for display
                display_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Display frame information
                self.display_frame_info(
                    display_frame,
                    smoothed_eye_contact and current_eye_contact_frames >= min_eye_contact_frames,
                    total_eye_contact_frames,
                    frame_count,
                    frame_rate,
                    results,
                    patient_region,
                    eye_center_3d,
                    gaze_vector_3d,
                    show_landmarks=False,
                    circle_params=circle_params
                )

                cv2.imshow('Eye Contact Detection', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    break

        finally:
            # Ensure cleanup happens even if an exception occurs
            cap.release()
            cv2.destroyAllWindows()

        # Add remaining eye contact duration if meets threshold
        if current_eye_contact_frames >= min_eye_contact_frames:
            total_eye_contact_frames += current_eye_contact_frames

        # Calculate total duration
        total_eye_contact_duration = total_eye_contact_frames / frame_rate

        return total_eye_contact_duration


def process_folder(base_folder, detector=None):
    """
    Process videos in folder structure with performance optimizations

    :param base_folder: Base folder containing videos
    :param detector: Optional EyeContactDetector instance
    :return: Dictionary of processing results with video duration information
    """
    # Create detector if not provided
    if detector is None:
        detector = EyeContactDetector(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            frame_skip=1,
            resize_width=1500
        )

    results = {}

    # Process each Face Angle folder
    for angle_folder in glob(os.path.join(base_folder, "Face Angle *")):
        angle_folder_name = os.path.basename(angle_folder)
        print(f"\nProcessing {angle_folder_name}")

        rotation_angle = get_rotation_angle_from_folder(angle_folder_name)

        print(f"Using rotation angle: {rotation_angle}°")

        # Process all numbered folders
        for video_folder in glob(os.path.join(angle_folder, "*")):
            if os.path.isdir(video_folder):
                folder_number = os.path.basename(video_folder)
                print(f"\nProcessing folder: {folder_number}")

                # Find Face videos
                face_videos = glob(os.path.join(video_folder, "Face*.mp4"))

                for video_file in face_videos:
                    video_name = os.path.basename(video_file)
                    print(f"Processing video: {video_name}")

                    # Get total video duration
                    cap = cv2.VideoCapture(video_file)
                    if not cap.isOpened():
                        print(f"Error: Could not open video file: {video_file}")
                        continue

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_duration = total_frames / fps if fps > 0 else 0
                    cap.release()

                    # Process video with performance optimizations
                    eye_contact_duration = detector.process_video_with_display(
                        video_file,
                        rotation_angle,
                        max_frames=None,
                        min_eye_contact_frames=30
                    )

                    if eye_contact_duration is not None:
                        # Calculate eye contact percentage
                        eye_contact_percentage = 0
                        if total_duration > 0:
                            eye_contact_percentage = (eye_contact_duration / total_duration) * 100

                        # Create a unique key that includes more info
                        key = f"{angle_folder_name}_{folder_number}_{video_name}"

                        results[key] = {
                            'folder': angle_folder_name,
                            'subfolder': folder_number,
                            'video_name': video_name,
                            'rotation': f"{rotation_angle}°",
                            'video_duration': total_duration,
                            'eye_contact_duration': eye_contact_duration,
                            'eye_contact_percentage': eye_contact_percentage
                        }

                        # Print intermediate results
                        print(f"Results for {video_name}:")
                        print(f"  Video duration: {total_duration:.2f} seconds")
                        print(f"  Eye contact duration: {eye_contact_duration:.2f} seconds")
                        print(f"  Eye contact percentage: {eye_contact_percentage:.2f}%")

    return results


def main():
    """Main function to process videos in folder structure and export to CSV"""
    import csv
    import datetime

    base_folder = r'C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\Trims Videos'

    if not os.path.exists(base_folder):
        print("Error: The specified folder does not exist!")
        return

    # Create optimized detector
    detector = EyeContactDetector(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        frame_skip=1,
        resize_width=1500
    )

    # Process all folders and collect results
    results = process_folder(base_folder, detector)

    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = os.path.join(base_folder, f'eye_contact_results_{timestamp}.csv')

    # Write results to CSV - one row per video
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'video_id',
            'folder',
            'rotation_angle',
            'video_duration_seconds',
            'eye_contact_duration_seconds',
            'eye_contact_percentage'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for video_id, data in results.items():
            writer.writerow({
                'video_id': video_id,
                'folder': data['folder'],
                'rotation_angle': data['rotation'],
                'video_duration_seconds': f"{data['video_duration']:.2f}",
                'eye_contact_duration_seconds': f"{data['eye_contact_duration']:.2f}",
                'eye_contact_percentage': f"{data['eye_contact_percentage']:.2f}"
            })

    print(f"\nResults exported successfully to CSV:")
    print(f"File: {csv_file_path}")
    print(f"Total videos processed: {len(results)}")

    # Print a summary of the findings
    total_videos = len(results)
    total_eye_contact = sum(data['eye_contact_duration'] for data in results.values())
    total_duration = sum(data['video_duration'] for data in results.values())

    print(f"\nSummary:")
    print(f"Total videos analyzed: {total_videos}")
    print(f"Total video duration: {total_duration:.2f} seconds")
    print(f"Total eye contact duration: {total_eye_contact:.2f} seconds")
    if total_duration > 0:
        overall_percentage = (total_eye_contact / total_duration) * 100
        print(f"Overall eye contact percentage: {overall_percentage:.2f}%")


if __name__ == "__main__":
    main()
