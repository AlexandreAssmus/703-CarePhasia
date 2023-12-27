import csv
import re
import os

def clean_csv_text_comprehensive(input_csv_path, output_csv_path):
    """
    Cleans the text column in a CSV file by:
    1. Removing specific markers and annotations.
    2. Removing all symbols and numbers except apostrophes.
    3. Eliminating lines that consist of only one word, specifically 'INV' or 'www'.

    Parameters:
    input_csv_path (str): The file path to the input CSV file.
    output_csv_path (str): The file path for the output cleaned CSV file.
    """
    with open(input_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        texts = [row['text'] for row in reader]

    clean_texts = []
    for text in texts:
        # First cleaning step: Remove specific markers and annotations
        text = re.sub(r'\x15.*?\x15|\[\+.*?\]', '', text).strip()

        # Second cleaning step: Remove all symbols and numbers except apostrophes
        text = re.sub(r"[^\w\s']", '', text)

        # Check the length and content of the line
        words = text.split()
        if len(words) == 1 and words[0].lower() in ['inv', 'www']:
            continue  # Skip this line
        clean_texts.append(text)

    # Write the cleaned texts to the output CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])  # Write the header
        for text in clean_texts:
            writer.writerow([text])

def process_csv_directories(input_root_directory, output_root_directory):
    """
    Processes all CSV files in a directory and its subdirectories, applying comprehensive text cleaning,
    and saves the cleaned data to new CSV files, mirroring the input directory structure.

    Parameters:
    input_root_directory (str): The root directory containing the input CSV files.
    output_root_directory (str): The root directory where the cleaned CSV files will be saved.
    """
    for dirpath, dirnames, files in os.walk(input_root_directory):
        for file in files:
            if file.endswith('.csv'):
                input_csv_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(dirpath, input_root_directory)
                output_dir = os.path.join(output_root_directory, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_csv_path = os.path.join(output_dir, file)

                clean_csv_text_comprehensive(input_csv_path, output_csv_path)
                print(f"Processed {input_csv_path} to {output_csv_path}")

# Example usage
input_root_directory = r'New_clean_code\Data\CSV_text_only'  # Replace with the path to your directory containing CSV files
output_root_directory = r'New_clean_code\Data\CSV_clean'  # Replace with the path to where you want to save the cleaned CSV files
process_csv_directories(input_root_directory, output_root_directory)