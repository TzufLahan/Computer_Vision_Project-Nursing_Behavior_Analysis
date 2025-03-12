import cv2
def get_rotation_angle_from_folder(folder_name):
    """
    Get the default rotation angle based on the Face Angle folder
    """
    # Define rotation angles for each folder
    rotation_angles = {
        'face angle 1': -56.2,
        'face angle 2': 3.4,
        'face angle 3': -8.1,
        'face angle 4': -18.3,
        'face angle 5': -18.3,
        'face angle 6': -2.8,
        'face angle 7': -11.3,
    }

    # Get the base folder name in lowercase
    folder_name = folder_name.lower()

    # Return the rotation angle or 0 if folder not found
    return rotation_angles.get(folder_name, 0)


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
    """
    Resize frame if it's larger than specified dimensions while maintaining aspect ratio
    """

    height, width = frame.shape[:2]

    # Calculate ratio to fit within max dimensions
    width_ratio = max_width / width if width > max_width else 1
    height_ratio = max_height / height if height > max_height else 1

    # Use the smaller ratio to ensure frame fits in both dimensions
    scale_ratio = min(width_ratio, height_ratio)

    if scale_ratio < 1:
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return frame
