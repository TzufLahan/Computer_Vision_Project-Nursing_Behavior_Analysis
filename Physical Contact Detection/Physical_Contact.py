import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import datetime
import argparse


def select_circular_exclusion_zones(frame, num_zones=0):
    """
    Allows user to select multiple circular exclusion zones (areas to ignore hand detection)
    Returns a list of exclusion zone circles represented as (center_x, center_y, radius)
    """
    circular_zones = []

    if num_zones <= 0:
        # Ask user how many exclusion zones they want to define
        print("How many circular exclusion zones do you want to define? (Enter a number, 0 for none)")
        try:
            num_zones = int(input())
        except ValueError:
            print("Invalid input. No exclusion zones will be defined.")
            num_zones = 0

    for i in range(num_zones):
        # Create a copy of the frame for drawing
        zone_img = frame.copy()

        # Circle parameters
        center = None
        radius = 0
        drawing = False

        def on_mouse_event(event, x, y, flags, param):
            nonlocal center, radius, drawing, zone_img

            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing - set center
                center = (x, y)
                radius = 0
                drawing = True
                # Draw a dot at the center
                cv2.circle(zone_img, center, 2, (0, 0, 255), -1)

            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                # Update radius as mouse moves
                temp_img = frame.copy()
                # Calculate radius from center to current mouse position
                dx = x - center[0]
                dy = y - center[1]
                radius = int(np.sqrt(dx * dx + dy * dy))
                # Draw the circle
                cv2.circle(temp_img, center, radius, (0, 255, 0), 2)
                cv2.circle(temp_img, center, 2, (0, 0, 255), -1)
                # Add semi-transparent fill
                overlay = temp_img.copy()
                cv2.circle(overlay, center, radius, (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.2, temp_img, 0.8, 0, temp_img)
                zone_img = temp_img

            elif event == cv2.EVENT_LBUTTONUP and drawing:
                # Finalize the circle
                drawing = False
                # Calculate final radius
                dx = x - center[0]
                dy = y - center[1]
                radius = int(np.sqrt(dx * dx + dy * dy))

        # Set up window for circle selection
        window_name = f'Select Circular Exclusion Zone {i + 1}/{num_zones}'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse_event)

        # Instructions on the image
        prompt_img = frame.copy()
        cv2.putText(prompt_img, f"Click and drag to define circle {i + 1}/{num_zones}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(prompt_img, "Press ENTER when done or ESC to skip", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        zone_img = prompt_img

        # Display instructions
        cv2.imshow(window_name, zone_img)
        print(f"Define circle for exclusion zone {i + 1}/{num_zones}. Press ENTER when done or ESC to skip.")

        # Wait for user to draw and confirm
        while True:
            cv2.imshow(window_name, zone_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER key
                break
            elif key == 27:  # ESC key
                center = None
                break

        cv2.destroyWindow(window_name)

        if center is not None and radius > 0:
            circular_zones.append((center, radius))
            print(f"Added circular exclusion zone at center={center}, radius={radius}")
        else:
            print(f"Circular exclusion zone {i + 1} was not properly defined. Skipping.")

    return circular_zones


def select_crop_area(frame):
    """
    Allows user to select a rectangular area to crop from the video
    Returns the x, y, width, height of the crop area
    """
    # Create a copy of the frame to avoid modifying the original
    crop_img = frame.copy()

    # Selected points for crop rectangle (top-left and bottom-right)
    crop_points = []
    rect_roi = None

    def on_mouse_click(event, x, y, flags, param):
        nonlocal crop_points, crop_img, rect_roi

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start point
            crop_points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if len(crop_points) == 1:
                # Show the rectangle while dragging
                temp_img = frame.copy()
                cv2.rectangle(temp_img, crop_points[0], (x, y), (0, 255, 0), 2)
                cv2.imshow('Select Crop Area - Click and drag', temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            # End point
            crop_points.append((x, y))

            # Draw the final rectangle
            cv2.rectangle(crop_img, crop_points[0], crop_points[1], (0, 255, 0), 2)
            cv2.imshow('Select Crop Area - Click and drag', crop_img)

            # Calculate ROI from the two points
            x1, y1 = crop_points[0]
            x2, y2 = crop_points[1]

            # Ensure x1,y1 is top-left and x2,y2 is bottom-right
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Store the ROI
            rect_roi = (x1, y1, x2 - x1, y2 - y1)

    cv2.namedWindow('Select Crop Area - Click and drag')
    cv2.setMouseCallback('Select Crop Area - Click and drag', on_mouse_click)
    cv2.imshow('Select Crop Area - Click and drag', crop_img)

    print("Click and drag to select crop area. Press ENTER when done or ESC to skip cropping.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER key
            break
        elif key == 27:  # ESC key
            rect_roi = None
            break

    cv2.destroyWindow('Select Crop Area - Click and drag')

    return rect_roi


def select_area(frame, prompt="Select 4 points to define the area"):
    """
    Allows user to select 4 points to define an area on a video frame
    Returns the points of the polygon area
    """
    # Create a copy of the frame to avoid modifying the original
    area_img = frame.copy()

    # Selected points
    points = []

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the point
            points.append((x, y))

            # Draw a point at the click location
            cv2.circle(area_img, (x, y), 5, (0, 0, 255), -1)

            # If we have 2 or more points, draw lines between them
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(area_img, points[i], points[i + 1], (0, 255, 0), 2)

                # If we have 4 points, connect the last point to the first
                if len(points) == 4:
                    cv2.line(area_img, points[3], points[0], (0, 255, 0), 2)

                    # Fill the polygon with semi-transparent color
                    overlay = area_img.copy()
                    cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0, 128))
                    cv2.addWeighted(overlay, 0.3, area_img, 0.7, 0, area_img)

            cv2.imshow('Select Area - Click 4 points', area_img)

    window_name = 'Select Area - Click 4 points'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    # Show the prompt on the image
    prompt_img = area_img.copy()
    cv2.putText(prompt_img, prompt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(window_name, prompt_img)

    print(f"{prompt}. Press ESC after selection.")

    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)

    if len(points) == 4:
        return points
    else:
        return None


def is_point_in_area(point, area_points):
    """
    Checks if a point is inside a polygon defined by area_points
    Returns True if the point is inside the area, False otherwise
    """
    x, y = point

    # Convert area points to numpy array
    polygon = np.array(area_points, np.int32)

    # Use OpenCV's pointPolygonTest to check if point is inside polygon
    # Returns positive value if inside, negative if outside, and zero if on the edge
    result = cv2.pointPolygonTest(polygon, (x, y), False)

    return result >= 0


def is_point_in_circular_zone(point, center, radius):
    """
    Checks if a point is inside a circular zone
    Returns True if the point is inside the circle, False otherwise
    """
    x, y = point
    center_x, center_y = center

    # Calculate distance from point to center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # If distance is less than radius, point is inside circle
    return distance <= radius


def is_point_in_any_circular_exclusion_zone(point, circular_zones):
    """
    Checks if a point is inside any of the defined circular exclusion zones
    Returns True if the point is inside any exclusion zone, False otherwise
    """
    for center, radius in circular_zones:
        if is_point_in_circular_zone(point, center, radius):
            return True
    return False


def find_rooftop_videos(root_dir):
    """
    Find all Rooftop_*.mp4 videos in the directory structure
    Returns a list of full paths to the videos
    """
    all_videos = []

    # Search for videos with pattern "Rooftop*.mp4" in the entire directory structure
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file starts with "Rooftop" and ends with ".mp4"
            if file.startswith("Rooftop") and file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                all_videos.append(full_path)

    # Sort the videos alphabetically
    all_videos.sort()

    return all_videos


def process_video(video_path, output_folder, csv_writer, use_existing_settings=False,
                  crop_settings=None, area_points=None, exclusion_zones=None):
    """
    Process a single video and write results to CSV
    Returns the crop, area, and zone settings used
    """
    print(f"\n{'=' * 60}")
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Full path: {video_path}")
    print(f"{'=' * 60}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video {video_path}")
        return None, None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_duration = total_frames / fps

    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Duration: {total_video_duration:.2f} seconds")
    print(f"FPS: {fps}")

    # Read first frame for setup
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}")
        cap.release()
        return None, None, None

    # If no existing settings, set up crop region, area and exclusion zones
    if not use_existing_settings or crop_settings is None:
        # Select crop region
        print("Select crop region (or press ESC to use full frame):")
        crop_roi = select_crop_area(first_frame)
        if crop_roi:
            x, y, width, height = crop_roi
        else:
            x, y = 0, 0
            width, height = frame_width, frame_height

        crop_settings = (x, y, width, height)

        # Apply crop to first frame for area selection
        cropped_first_frame = first_frame[y:y + height, x:x + width]

        # Select patient area
        print("Select patient area (4 points):")
        area_points = select_area(cropped_first_frame, "Select 4 points for patient area")
        if not area_points or len(area_points) != 4:
            print("Patient area not properly selected. Exiting.")
            cap.release()
            return None, None, None

        # Select exclusion zones
        print("Select exclusion zones for patient hands:")
        exclusion_zones = select_circular_exclusion_zones(cropped_first_frame)
    else:
        # Use existing settings
        x, y, width, height = crop_settings

    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=4)

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Variables for tracking touch time
    frame_count = 0
    total_contact_time = 0
    contact_start_time = None
    current_contact = False

    # State stability variables
    stability_counter = 0
    pending_state = None
    stability_frames = 3

    # Hand landmarks for checking
    key_landmarks = [0, 8, 20]  # wrist, index fingertip, pinky fingertip
    landmark_names = {
        0: "Wrist",
        8: "Index Fingertip",
        20: "Pinky Fingertip"
    }

    # Video filename info (for display only)
    video_filename = os.path.basename(video_path)

    print(f"Analysis started. Processing {video_filename}...")

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply cropping
        cropped_frame = frame[y:y + height, x:x + width]

        frame_count += 1
        video_time = frame_count / fps

        # Create display frame
        display_frame = cropped_frame.copy()

        # Process image with MediaPipe
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Track hand points inside area
        any_hand_in_area = False
        contact_landmarks = []

        # Draw area outline and fill based on contact state
        if not current_contact:
            cv2.polylines(display_frame, [np.array(area_points)], True, (0, 0, 255), 2)
        else:
            cv2.polylines(display_frame, [np.array(area_points)], True, (0, 255, 0), 2)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [np.array(area_points)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

        # Draw exclusion zones
        for center, radius in exclusion_zones:
            cv2.circle(display_frame, center, radius, (255, 0, 0), 2)
            overlay = display_frame.copy()
            cv2.circle(overlay, center, radius, (255, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
            cv2.putText(display_frame, "Patient Hand",
                        (center[0] - 40, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add patient area label
        area_center_x = sum(p[0] for p in area_points) // len(area_points)
        area_min_y = min(p[1] for p in area_points)
        label_y = area_min_y - 15
        cv2.putText(display_frame, "Patient Area",
                    (area_center_x - 40, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Process hands if detected
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get wrist position
                wrist = hand_landmarks.landmark[0]
                h, w, _ = cropped_frame.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Check if in exclusion zone
                in_exclusion_zone = is_point_in_any_circular_exclusion_zone(
                    (wrist_x, wrist_y), exclusion_zones)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # If in exclusion zone, mark and skip
                if in_exclusion_zone:
                    cv2.circle(display_frame, (wrist_x, wrist_y), 15, (255, 0, 0), 2)
                    cv2.putText(display_frame, "Patient Hand",
                                (wrist_x - 40, wrist_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    continue

                # Otherwise, process as nurse hand
                handedness = results.multi_handedness[i]
                hand_label = handedness.classification[0].label
                nurse_hand = "Right Hand" if hand_label == "Left" else "Left Hand"

                cv2.putText(display_frame, f"Nurse {nurse_hand}",
                            (wrist_x - 30, wrist_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Check key landmarks for area containment
                for idx in key_landmarks:
                    landmark = hand_landmarks.landmark[idx]
                    px, py = int(landmark.x * w), int(landmark.y * h)

                    # Skip if in exclusion zone
                    if is_point_in_any_circular_exclusion_zone((px, py), exclusion_zones):
                        continue

                    # Check if inside patient area
                    if is_point_in_area((px, py), area_points):
                        any_hand_in_area = True
                        contact_landmarks.append(landmark_names[idx])
                        cv2.circle(display_frame, (px, py), 10, (0, 0, 255), -1)

        # Determine contact state
        potential_contact = any_hand_in_area

        # Apply stability logic
        if potential_contact != current_contact:
            if pending_state is None:
                pending_state = potential_contact
                stability_counter = 1
            elif pending_state == potential_contact:
                stability_counter += 1
                if stability_counter >= stability_frames:
                    previous_contact = current_contact
                    current_contact = pending_state
                    pending_state = None
                    stability_counter = 0

                    # Handle state transition
                    if current_contact and not previous_contact:
                        contact_start_time = video_time
                        print(f"Contact started at {video_time:.2f}s")
                    elif not current_contact and previous_contact:
                        contact_duration = video_time - contact_start_time
                        total_contact_time += contact_duration
                        print(f"Contact ended. Duration: {contact_duration:.2f}s")
            else:
                pending_state = potential_contact
                stability_counter = 1
        else:
            if potential_contact == current_contact:
                pending_state = None
                stability_counter = 0

        # Display information on frame
        contact_status = "Contact Detected" if current_contact else "No Contact Detected"
        contact_color = (0, 255, 0) if current_contact else (0, 0, 255)
        cv2.putText(display_frame, contact_status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, contact_color, 2)

        # Add video name
        cv2.putText(display_frame, os.path.basename(video_path),
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # Display timing information
        cv2.putText(display_frame, f"Time In Area: {total_contact_time:.2f}s",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Video Time: {video_time:.2f}s",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Show frame
        cv2.imshow("Analysis", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # If still in contact at the end, add final time
    if current_contact:
        contact_duration = video_time - contact_start_time
        total_contact_time += contact_duration

    # Clean up
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    # Calculate contact percentage
    contact_percentage = (total_contact_time / total_video_duration * 100) if total_video_duration > 0 else 0

    # Display summary
    print(f"\n--- Contact Summary for {video_filename} ---")
    print(f"Total Contact Time: {total_contact_time:.2f} seconds")
    print(f"Video Duration: {total_video_duration:.2f} seconds")
    print(f"Contact Percentage: {contact_percentage:.2f}%")

    # Extract folder structure information
    folder_path = os.path.dirname(video_path)
    folder_parts = folder_path.split(os.path.sep)

    # Try to extract angle info from filename
    angle_info = "Unknown"
    if "Angle" in video_filename:
        try:
            angle_info = video_filename.split("Angle")[1].split("_")[0].strip()
        except:
            pass

    # Extract additional information from folder structure if available
    face_angle = "Unknown"
    angle_number = "Unknown"

    # Look for Face Angle folder
    for part in folder_parts:
        if "Face Angle" in part:
            face_angle = part
        # Try to find a number folder
        elif part.isdigit():
            angle_number = part

    # Write results to CSV
    csv_writer.writerow([
        os.path.basename(video_path),
        face_angle,
        angle_number,
        angle_info,
        total_contact_time,
        total_video_duration,
        contact_percentage,
        len(exclusion_zones),
        "Not saved",  # No video path since we're not saving
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

    return crop_settings, area_points, exclusion_zones


def process_all_rooftop_videos(root_dir, output_folder="touch_analysis_results"):
    """
    Find and process all Rooftop videos in the directory structure
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Create CSV file for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    touch_data_csv = os.path.join(output_folder, f"hand_contact_analysis_{timestamp}.csv")

    # Write CSV header - updated to reflect that we're not saving videos
    with open(touch_data_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Video Name',
            'Face Angle',
            'Angle Number',
            'Angle Info',
            'Contact Time (s)',
            'Video Duration (s)',
            'Contact Percentage (%)',
            'Exclusion Zones Count',
            'Analysis Status',
            'Timestamp'
        ])

    # Find all Rooftop videos
    video_files = find_rooftop_videos(root_dir)

    if not video_files:
        print(f"No Rooftop videos found in {root_dir}")
        return

    print(f"Found {len(video_files)} Rooftop videos to analyze.")

    # Ask user if they want to use the same settings for all videos
    print("\nDo you want to set up crop area and analysis settings once and use for all videos? (y/n)")
    use_same_settings = input().strip().lower() == 'y'

    # Variables to store settings
    crop_settings = None
    area_points = None
    exclusion_zones = None

    # Process videos
    with open(touch_data_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for i, video_path in enumerate(video_files):
            print(f"\n[{i + 1}/{len(video_files)}] Processing {os.path.basename(video_path)}")

            try:
                # Process first video and get settings
                if i == 0 or not use_same_settings:
                    crop_settings, area_points, exclusion_zones = process_video(
                        video_path, output_folder, csv_writer, False)
                else:
                    # Use existing settings for subsequent videos
                    process_video(video_path, output_folder, csv_writer, True,
                                  crop_settings, area_points, exclusion_zones)

            except Exception as e:
                print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
                # Write error to CSV
                csv_writer.writerow([
                    os.path.basename(video_path),
                    "Unknown",  # Face Angle
                    "Unknown",  # Angle Number
                    "Unknown",  # Angle Info
                    "Error",  # Contact Time
                    "Error",  # Video Duration
                    "Error",  # Percentage
                    0,  # Exclusion Zones Count
                    "N/A",  # Processed Video Path
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

    print(f"\nAll videos processed. Results saved to {touch_data_csv}")
    print(f"Processed videos saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hand contact in Rooftop videos')
    parser.add_argument('-d', '--dir', type=str, help='Root directory containing videos', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output directory', default='touch_analysis_results')

    args = parser.parse_args()

    if args.dir:
        # Process all Rooftop videos in directory structure
        process_all_rooftop_videos(args.dir, args.output)
    else:
        print("Please specify a directory with videos using the -d option")


if __name__ == "__main__":
    main()
