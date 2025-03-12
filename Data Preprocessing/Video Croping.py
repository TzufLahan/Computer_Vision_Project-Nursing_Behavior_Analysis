import cv2
import os


def trim_video(input_path, output_path, start_time, end_time):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

        if current_frame % fps == 0:
            print(f"Processing {os.path.basename(input_path)}: {(current_frame - start_frame) / (end_frame - start_frame) * 100:.1f}%")

    cap.release()
    out.release()
    print(f"Trimming completed for {os.path.basename(input_path)}!")


def process_videos(base_path, trim_times):
    """
    Process videos in Face Angle folders structure
    """
    # Iterate through Face Angle folders
    for angle_folder in os.listdir(base_path):
        if not angle_folder.startswith("Face Angle"):
            continue

        angle_path = os.path.join(base_path, angle_folder)
        print(f"\nProcessing Face Angle folder: {angle_folder}")

        # Iterate through numbered folders (101, 102, etc.)
        for num_folder in os.listdir(angle_path):
            if not num_folder.isdigit():
                continue

            # Check if we have timing data for this folder
            if int(num_folder) not in trim_times:
                print(f"Skipping folder {num_folder} - no timing data")
                continue

            start_time, end_time = trim_times[int(num_folder)]
            folder_path = os.path.join(angle_path, num_folder)

            print(f"\nProcessing folder {num_folder}")
            print(f"Using start time: {start_time}s, end time: {end_time}s")

            # Process all MP4 files in this folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.mp4'):
                    input_path = os.path.join(folder_path, filename)
                    output_filename = os.path.splitext(filename)[0] + "_trimmed.mp4"
                    output_path = os.path.join(folder_path, output_filename)

                    print(f"\nProcessing: {filename}")
                    trim_video(input_path, output_path, start_time, end_time)


if __name__ == "__main__":
 #    base_path = input("Enter base path containing all Face Angle folders: ").strip().strip('"')
 #
 #    # Dictionary mapping numbered folders to their trim times
 #    trim_times = {
 #        # folder_number: (start_time, end_time)
 #        101: (32, 365), 102: (15, 394), 103: (240, 392), 104: (17, 230), 105: (25, 380),106: (26,150 ), 107: (20,235), 108: (30, 294),
 #        109: (95, 255), 110: (130, 330), 111: (104, 442), 112: (52, 315), 113: (99, 404), 114:(130,480),  115: (50, 250), 116: (93, 450), 119: (36, 322),
 #        120: (25, 255), 121: (15, 255),122: (14, 380), 123: (29, 256), 124: (35, 146), 125: (16, 244), 126: (60, 213), 127: (41, 177),
 #        128: (58, 184), 129: (50, 190), 130: (30, 172), 131: (68, 322), 132: (13, 229), 133: (60, 190), 134: (37, 230), 135: (54, 285),
 #        136: (48, 190),  137: (47, 247), 138: (23, 240), 140: (14, 178),
 # }
 #
 #
 #
 #    # Process all videos
 #    process_videos(base_path, trim_times)
 #
    print("\nAll videos have been processed!")
    inputt = r"C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\Trims Videos\Face Angle 1\101\Face 101_trimmed.mp4"
    output = r"C:\Users\zupl1\OneDrive - post.bgu.ac.il\שולחן העבודה\כריית נתונים במאגרים גדולים\פרויקט סופי\סרטונים -סימולציה\Trims Videos\Face Angle 1\101\Face 101_trimmed_SHORT.mp4"
    trim_video(inputt,output , 0 ,5)