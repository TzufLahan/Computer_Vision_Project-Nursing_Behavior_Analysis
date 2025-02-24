import cv2
import os
import re
from progress.bar import Bar
import numpy as np
from glob import glob
from datetime import datetime
import pandas as pd


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_video(image_folder, video_name, fps=30.0):
    print(f"\nStarting video creation from frames in {image_folder}")
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if not images:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(images)} frames to process")
    images.sort(key=natural_keys)

    # Read first frame to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print(f"Error reading first frame from {image_folder}")
        return

    height, width, layers = frame.shape
    print(f"Video dimensions: {width}x{height}")
    size = (width, height)

    print(f"Initializing video writer with {fps} fps")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    bar = Bar('Creating Video', max=len(images))
    frames_processed = 0

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed}/{len(images)} frames")
        else:
            print(f"Error reading frame: {img_path}")
        bar.next()

    bar.finish()
    video.release()
    print(f"Video creation completed. Saved to: {video_name}")

    print("\nCleaning up frames...")
    files_removed = 0
    for file in os.listdir(image_folder):
        try:
            os.remove(os.path.join(image_folder, file))
            files_removed += 1
        except Exception as e:
            print(f"Error removing file {file}: {e}")
    print(f"Cleanup completed. Removed {files_removed} files")


def get_reference_frames():
    """
    Define reference frames and videos for each Face Angle to ensure the most likely HeatMap
    """
    return {
        'Face Angle 1': {
            'folder': '101',  # Specify the folder number containing the reference video
            'frame': 1  # Specify the starting frame
        },
        'Face Angle 2': {
            'folder': '105',
            'frame': 350
        },
        'Face Angle 3': {
            'folder': '106',
            'frame': 480
        },
        'Face Angle 4': {
            'folder': '110',
            'frame': 480
        },
        'Face Angle 5': {
            'folder': '115',
            'frame': 1
        },
        'Face Angle 6': {
            'folder': '124',
            'frame': 1
        },
        'Face Angle 7': {
            'folder': '138',
            'frame': 1
        }
    }


def get_reference_frame_image(angle_name, folder_number, frame_number):
    """
    Get the reference frame image from the original videos folder
    """
    # Original videos folder
    original_folder = r"C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\סרטים ניסוי"

    # Construct path to the reference video
    angle_folder_path = os.path.join(original_folder, angle_name)
    video_folder = os.path.join(angle_folder_path, folder_number)

    # Find Upper Angle video in the reference folder
    upper_videos = glob(os.path.join(video_folder, "Upper Angle*.mp4"))
    if not upper_videos:
        print(f"Warning: No Upper Angle videos found in original folder {video_folder}")
        return None

    video_path = upper_videos[0]

    # Open video and extract frame
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Error: Could not open original video: {video_path}")
        return None

    # Check if frame number is valid
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"Warning: Frame {frame_number} exceeds total frames {total_frames} in original video")
        capture.release()
        return None

    # Get the frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = capture.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number} from original video")
        capture.release()
        return None

    # Create a deep copy of the frame to ensure it's not modified later
    reference_frame = frame.copy()
    capture.release()

    return reference_frame


def preview_reference_frames():
    """
    Preview reference frames from the original videos folder
    """
    # Original videos folder
    original_folder = r"C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\סרטים ניסוי"

    # Create directory for preview images
    preview_dir = os.path.join(os.path.dirname(original_folder), "frame_previews")
    os.makedirs(preview_dir, exist_ok=True)

    # Get reference frames
    reference_frames = get_reference_frames()
    preview_results = []

    print("\n===== PREVIEWING REFERENCE FRAMES FROM ORIGINAL VIDEOS =====")

    # Process each angle
    for angle_name, ref_config in reference_frames.items():
        print(f"\nChecking {angle_name}:")

        folder_number = ref_config['folder']
        start_frame = ref_config['frame']

        # Get reference frame using the helper function
        frame = get_reference_frame_image(angle_name, folder_number, start_frame)
        if frame is None:
            continue

        # Save preview image
        preview_path = os.path.join(preview_dir, f"{angle_name}_frame_{start_frame}.jpg")
        cv2.imwrite(preview_path, frame)

        # Add info text
        info_frame = frame.copy()
        cv2.putText(info_frame, f"{angle_name}: Frame {start_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Original Folder: {folder_number}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        window_name = f"Original Preview: {angle_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, info_frame)
        print(f"Showing frame {start_frame} from original video in folder {folder_number}")
        print("Press any key to continue to the next angle...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        # Record result
        preview_results.append({
            'angle_name': angle_name,
            'folder': folder_number,
            'frame': start_frame,
            'preview_image': preview_path
        })

    # Close all windows
    cv2.destroyAllWindows()

    # Save results
    if preview_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_df = pd.DataFrame(preview_results)
        preview_csv = os.path.join(preview_dir, f'original_frame_previews_{timestamp}.csv')
        preview_df.to_csv(preview_csv, index=False)
        print(f"\nPreview information saved to: {preview_csv}")
        print(f"Preview images saved to: {preview_dir}")

    # Ask user to continue
    response = input("\nDo you want to continue with processing the trimmed videos? (y/n): ")
    return response.lower() == 'y'


def process_video(video_path, output_base_dir, angle_name, ref_folder, start_frame=0):
    """Process a single video using the original algorithm but with specified start frame"""
    # Use reference frame from original videos
    reference_frame = get_reference_frame_image(angle_name, ref_folder, start_frame)
    if reference_frame is None:
        print(f"Failed to get reference frame. Defaulting to first frame of trimmed video.")
        use_reference_frame = False
    else:
        print(f"Successfully loaded reference frame {start_frame} from folder {ref_folder}")
        use_reference_frame = True

    # Fix video name extraction
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]

    output_dir = os.path.join(output_base_dir, video_name)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\nProcessing video: {video_path}")
    print(f"Using start frame: {start_frame}")
    print(f"Output directory: {output_dir}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    # Set starting position to the frame from reference
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read first frame to get dimensions
    ret, first_frame = capture.read()
    if not ret:
        print("Failed to read first frame")
        return None

    # If we couldn't get reference frame, use the first frame from video
    if not use_reference_frame:
        reference_frame = first_frame.copy()

    height, width = first_frame.shape[:2]
    accum_image = np.zeros((height, width), np.uint8)
    print(f"Frame dimensions: {width}x{height}")

    print("Initializing background subtractor...")
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {length}")

    bar = Bar('Processing Frames', max=length - start_frame)
    frames_processed = 0
    frames_saved = 0

    # Reset to the start frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("\nStarting frame processing...")
    for i in range(start_frame, length):
        ret, frame = capture.read()
        if not ret:
            print(f"Error reading frame {i}")
            break

        try:
            # Remove the background
            filter = background_subtractor.apply(frame)

            threshold = 3
            maxValue = 0.51
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            # Add to accumulated image
            accum_image = cv2.add(accum_image, th1)

            normalized_accum = np.clip(accum_image * 0.8, 0, 255).astype(np.uint8)
            color_image_video = cv2.applyColorMap(normalized_accum, cv2.COLORMAP_JET)
            video_frame = cv2.addWeighted(frame, 1, color_image_video, 0.4, 0)

            # Save frame with error checking
            frame_path = os.path.join(frames_dir, f"f{i:04d}.jpg")
            success = cv2.imwrite(frame_path, video_frame)
            if success:
                frames_saved += 1
                if frames_saved % 100 == 0:
                    print(f"Successfully saved frame {i} to {frame_path}")
            else:
                print(f"Failed to save frame {i}")

            frames_processed += 1
            if frames_processed % 500 == 0:
                print(f"Processed {frames_processed}/{length - start_frame} frames")
                print(f"Successfully saved {frames_saved} frames")

        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")

        bar.next()

    bar.finish()
    print(f"\nFrame processing completed.")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total frames saved: {frames_saved}")

    # Create output video
    output_video_path = None
    if frames_saved > 0:
        print("\nCreating final video...")
        output_video_path = os.path.join(output_dir, video_name + "_analyzed.avi")
        make_video(frames_dir, output_video_path)
    else:
        print("No frames were saved, cannot create video")

    # Create heatmap on the reference frame (from original video)
    print("\nCreating heatmap using the reference frame...")
    try:
        normalized_accum = np.clip(accum_image * 0.8, 0, 255).astype(np.uint8)
        color_image = cv2.applyColorMap(normalized_accum, cv2.COLORMAP_JET)

        # Use the reference frame with higher weight for better quality
        result_overlay = cv2.addWeighted(reference_frame, 0.8, color_image, 0.4, 0)
        heatmap_path = os.path.join(output_dir, video_name + "_heatmap.jpg")

        if cv2.imwrite(heatmap_path, result_overlay):
            print(f"Heatmap saved successfully to: {heatmap_path}")
        else:
            print("Failed to save heatmap")

        # Also save pure heatmap for reference
        pure_heatmap_path = os.path.join(output_dir, video_name + "_pure_heatmap.jpg")
        if cv2.imwrite(pure_heatmap_path, color_image):
            print(f"Pure heatmap saved to: {pure_heatmap_path}")

    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")

    capture.release()

    return {
        'video_name': video_name,
        'frames_processed': frames_processed,
        'frames_saved': frames_saved,
        'start_frame': start_frame,
        'reference_frame_used': use_reference_frame,
        'output_video': output_video_path,
        'heatmap_path': heatmap_path if 'heatmap_path' in locals() else None
    }


def main():
    print("Starting video processing with reference frames...")

    # Define paths
    process_folder = r"C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\Trims Videos"
    base_dir = r"C:\temp\video_process"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Preview reference frames first
    if not preview_reference_frames():
        print("Processing cancelled by user")
        return

    # Get reference frames configuration
    reference_frames = get_reference_frames()
    all_results = []

    print("\n===== PROCESSING TRIMMED VIDEOS WITH REFERENCE FRAMES =====")

    # Process each angle folder
    for angle_folder in glob(os.path.join(process_folder, "Face Angle *")):
        angle_name = os.path.basename(angle_folder)
        print(f"\nProcessing {angle_name}")

        # Get reference frame for this angle
        if angle_name not in reference_frames:
            print(f"No reference frame defined for {angle_name}, skipping...")
            continue

        ref_config = reference_frames[angle_name]
        start_frame = ref_config['frame']
        ref_folder = ref_config['folder']
        print(f"Using frame {start_frame} from reference folder {ref_folder} for all videos in {angle_name}")

        # Process all videos in this angle folder with the reference frame
        for video_folder in glob(os.path.join(angle_folder, "*")):
            if os.path.isdir(video_folder):
                folder_number = os.path.basename(video_folder)
                upper_videos = glob(os.path.join(video_folder, "Upper Angle*.mp4"))

                for video_path in upper_videos:
                    print(f"\nProcessing video: {os.path.basename(video_path)}")
                    # Pass angle_name and ref_folder to the process_video function
                    result = process_video(video_path, base_dir, angle_name, ref_folder, start_frame)

                    if result:
                        result['angle_folder'] = angle_name
                        result['folder_number'] = folder_number
                        result['reference_folder'] = ref_folder
                        all_results.append(result)

    # Save analysis results to CSV
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(base_dir, f'motion_analysis_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

    print("\nProcessing completed!")


if __name__ == '__main__':
    main()
