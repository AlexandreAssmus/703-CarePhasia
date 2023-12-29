import os
import pandas as pd

def process_csv_file(file_path, diagnosis, target_root, source_root):
    """
    Reads a CSV file, adds a 'person_type' column, and writes it to the target directory.

    Parameters:
    file_path (str): The path to the CSV file.
    person_type (str): The type of person to tag the file with ('control' or 'patient').
    target_root (str): The root directory where the processed file will be saved.
    source_root (str): The root directory where the original file is located.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Add the 'person_type' column with the appropriate tag
    df['diagnosis'] = diagnosis
    # Determine the new file path based on the target root and the relative path from the source root
    relative_path = os.path.relpath(file_path, source_root)
    new_file_path = os.path.join(target_root, relative_path)
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    # Save the updated DataFrame back to a CSV file in the target directory
    df.to_csv(new_file_path, index=False)

def process_directory(source_root, person_type, target_root):
    """
    Walks through a directory and its subdirectories to process all CSV files found,
    and saves the processed files to the target directory.

    Parameters:
    source_root (str): The root directory to start processing from.
    person_type (str): The type of person to tag the files with ('control' or 'patient').
    target_root (str): The root directory where the processed files will be saved.
    """
    for subdir, dirs, files in os.walk(source_root):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                # Get the full file path
                file_path = os.path.join(subdir, file)
                # Process the CSV file and save it to the target directory
                process_csv_file(file_path, person_type, target_root, source_root)

# Define the root directories
source_control_dir = 'New_clean_code\Data\CSV_clean\Control_csv_data_clean'
source_patient_dir = 'New_clean_code\Data\CSV_clean\Patient_csv_data_clean'

# Define the target directory
target_dir = 'New_clean_code\Data\Tagged_diagnosis_data_csv'

# Process the control directory CSV files and save them to the target directory
process_directory(source_control_dir, 'control', target_dir)

# Process the patient directory CSV files and save them to the target directory
process_directory(source_patient_dir, 'patient', target_dir)
