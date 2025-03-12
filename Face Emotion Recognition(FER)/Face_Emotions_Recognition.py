from fer import FER
import cv2
from Angles_Per_Video import get_rotation_angle_from_folder, resize_frame_to_fit_screen, rotate_frame
import os
from glob import glob
import pandas as pd
from datetime import datetime


def process_video_for_happiness(video_path, rotation_angle, angle_name, start_frame=0, happiness_threshold=0.8):
    """
    Process video focusing only on happiness detection
    Parameters:
        video_path: Path to the video file
        rotation_angle: Angle to rotate the frame
        angle_name: Name of the angle folder
        start_frame: Frame to start processing from
        happiness_threshold: Threshold for considering an expression as happy (0-1)
    """
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_path)
    frame_skip = 2
    process_width = 640
    video_name = os.path.basename(video_path)

    if not cap.isOpened():
        return None

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    happiness_data = []
    frame_count = 0

    while cap.isOpened():
        if frame_count % frame_skip != 0:
            ret = cap.grab()
            frame_count += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame = cv2.resize(frame, (process_width, int(process_width * frame.shape[0] / frame.shape[1])))
            rotated_frame = rotate_frame(frame, rotation_angle)
            display_frame = resize_frame_to_fit_screen(rotated_frame)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            emotions = detector.detect_emotions(frame_rgb)

            if emotions:
                happiness_score = emotions[0]['emotions']['happy']
                happiness_data.append({
                    'video_name': video_name,
                    'angle': angle_name,
                    'frame': frame_count,
                    'timestamp': frame_count / frame_rate,
                    'happiness_score': happiness_score,
                    'is_happy': happiness_score > happiness_threshold
                })

                # Draw on frame
                x, y, w, h = [int(v) for v in emotions[0]['box']]
                color = (0, 255, 0) if happiness_score > happiness_threshold else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"Happy: {happiness_score:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Happiness Detection', display_frame)
            frame_count += 1

        except Exception as e:
            print(f"Error on frame {frame_count}: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if happiness_data:
        df = pd.DataFrame(happiness_data)
        analysis_results = {
            'video_name': video_name,
            'angle': angle_name,
            'total_frames_processed': len(df),
            'video_duration': total_frames / frame_rate,
            'average_happiness_score': df['happiness_score'].mean(),
            'max_happiness_score': df['happiness_score'].max(),
            'happy_frames_count': df['is_happy'].sum(),
            'happy_duration': (df['is_happy'].sum() * frame_skip) / frame_rate,
            'percent_happy': (df['is_happy'].sum() / len(df)) * 100
        }
        return df, analysis_results
    return None


def main_happiness_detection():
    base_folder = r'C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\Trims Videos'

    # Create lists to store all data
    all_happiness_data = []
    all_analysis_results = []

    for angle_folder in glob(os.path.join(base_folder, "Face Angle *")):
        angle_folder_name = os.path.basename(angle_folder)
        rotation_angle = get_rotation_angle_from_folder(angle_folder_name)

        for video_folder in glob(os.path.join(angle_folder, "*")):
            if os.path.isdir(video_folder):
                for video_file in glob(os.path.join(video_folder, "Face*.mp4")):
                    video_name = os.path.basename(video_file)
                    print(f"\nProcessing {video_name} from {angle_folder_name}")

                    results = process_video_for_happiness(video_file, rotation_angle, angle_folder_name)

                    if results:
                        df, analysis = results
                        all_happiness_data.append(df)
                        all_analysis_results.append(analysis)

                        # Print analysis results
                        print(f"Average Happiness Score: {analysis['average_happiness_score']:.2f}")
                        print(f"Time Spent Happy: {analysis['happy_duration']:.2f} seconds")
                        print(f"Percentage of Time Happy: {analysis['percent_happy']:.1f}%")

    # Combine all data
    if all_happiness_data:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed frame-by-frame data
        combined_df = pd.concat(all_happiness_data, ignore_index=True)
        detailed_output_file = f"happiness_analysis_detailed_{timestamp}.csv"
        combined_df.to_csv(detailed_output_file, index=False)
        print(f"\nDetailed frame-by-frame data saved to: {detailed_output_file}")

        # Save summary analysis
        summary_df = pd.DataFrame(all_analysis_results)
        summary_output_file = f"happiness_analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_output_file, index=False)
        print(f"Summary analysis saved to: {summary_output_file}")


if __name__ == "__main__":
    main_happiness_detection()
