
import csv
import json
import os

def json_to_csv(json_directory, csv_directory):
    """
    Converts JSON files to CSV files, including only text from 'PAR' type entries.
    Maintains the directory structure of the input JSON files for the output CSV files.

    Parameters:
    json_directory (str): The directory containing the JSON files.
    csv_directory (str): The directory where the CSV files will be saved, mirroring the input structure.
    """
    # Recreate the directory structure and convert JSON files
    for dirpath, dirnames, files in os.walk(json_directory):
        for file in files:
            if file.endswith('.json'):
                # Construct the full file paths
                json_file_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(dirpath, json_directory)
                output_dir = os.path.join(csv_directory, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                csv_file_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.csv')

                # Read the JSON file
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                
                # Filter out the objects where type is 'PAR' and extract the 'text'
                texts = [entry['text'] for entry in data if entry['type'] == 'PAR']
                
                # Write the texts to a CSV file
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['text'])  # Write the header
                    for text in texts:
                        writer.writerow([text])  # Write the text entries

                print(f"Converted {json_file_path} to {csv_file_path}")
                
# Example usage
json_directory = r'New_clean_code\Data\Full_Json'  # Replace with the path to your JSON files directory
csv_directory = r'New_clean_code\Data\CSV_text_only'    # Replace with the path to where you want to save the CSV files
json_to_csv(json_directory, csv_directory)
