import os
import re
from pathlib import Path


def is_valid_number(number_str):
    """
    Check if the number is between 101 and 140
    """
    try:
        number = int(number_str)
        return 101 <= number <= 140
    except ValueError:
        return False


def convert_filename(original_name):
    """
    Convert Hebrew filename patterns to English according to specified rules
    """
    # Remove file extension if present
    name_part = os.path.splitext(original_name)[0]
    extension = os.path.splitext(original_name)[1]

    # Define patterns - handle both directions and no spaces
    panim_pattern = r'(?:פנים\s*(\d+)|(\d+)\s*פנים|פנים(\d+)|(\d+)פנים)'
    upper_pattern = r'(?:עליון\s*(\d+)|(\d+)\s*עליון|עליון(\d+)|(\d+)עליון)'
    roof_pattern = r'(?:תיקרה\s*(\d+)|(\d+)\s*תיקרה|תיקרה(\d+)|(\d+)תיקרה|(\d+)\s*תיקרה)'

    # Helper function to get first valid number from groups
    def get_first_valid_number(match):
        if match:
            for group in match.groups():
                if group and is_valid_number(group):
                    return group
        return None

    # Check each pattern
    panim_match = re.search(panim_pattern, name_part)
    number = get_first_valid_number(panim_match)
    if number:
        return f"Face {number}{extension}"

    upper_match = re.search(upper_pattern, name_part)
    number = get_first_valid_number(upper_match)
    if number:
        return f"Upper Angle {number}{extension}"

    roof_match = re.search(roof_pattern, name_part)
    number = get_first_valid_number(roof_match)
    if number:
        return f"Rooftop Angle {number}{extension}"

    # Return None if no valid pattern is found
    return None


def process_directory(directory_path):
    """
    Process all directories and files recursively
    """
    try:
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        renamed_files = []
        skipped_files = []

        # First, process files in current directory
        for item in dir_path.iterdir():
            if item.is_file():
                original_name = item.name
                new_name = convert_filename(original_name)

                if new_name:
                    try:
                        new_path = item.parent / new_name
                        # Handle duplicates
                        counter = 1
                        base_name = os.path.splitext(new_name)[0]
                        extension = os.path.splitext(new_name)[1]
                        while new_path.exists():
                            new_name = f"{base_name}_{counter}{extension}"
                            new_path = item.parent / new_name
                            counter += 1

                        item.rename(new_path)
                        renamed_files.append((original_name, new_name))
                    except Exception as e:
                        print(f"Error renaming {original_name}: {str(e)}")
                else:
                    skipped_files.append(original_name)

        # Then process subdirectories
        for item in dir_path.iterdir():
            if item.is_dir():
                # Process the subdirectory
                sub_renamed, sub_skipped = process_directory(item)
                renamed_files.extend(sub_renamed)
                skipped_files.extend(sub_skipped)

        return renamed_files, skipped_files

    except Exception as e:
        print(f"An error occurred processing directory {directory_path}: {str(e)}")
        return [], []


def main():
    """
    Main function to run the rename operation
    """
    print("This script will rename files according to these patterns:")
    print("1. 'פנים XXX' -> 'Face XXX'")
    print("2. 'עליון XXX' -> 'Upper Angle XXX'")
    print("3. 'תקרה XXX' -> 'Rooftop Angle XXX'")
    print("\nIt will process all subdirectories recursively.")
    print("\nWarning: This will rename files directly. Make sure you have a backup.")

    root_dir = input("\nEnter the root directory path: ")



    renamed_files, skipped_files = process_directory(root_dir)

    # Print summary
    print("\nRename operation completed!")
    if renamed_files:
        print(f"\nRenamed {len(renamed_files)} files:")
        for old_name, new_name in renamed_files:
            print(f"  {old_name} → {new_name}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files (no matching pattern):")
        for filename in skipped_files:
            print(f"  {filename}")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()