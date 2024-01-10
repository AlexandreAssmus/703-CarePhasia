import json
import csv
import os

def process_json_to_csv(json_file_path, csv_file_path):
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Filter out only the blocks with the tag "PAR"
    par_texts = [block['text'] for block in data if block['type'] == 'PAR']

    # Write the text to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(['text'])
        # Write the text
        for text in par_texts:
            writer.writerow([text])

# Example usage
json_file_path = r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Json_files\Controls\Wright.json'  # Replace with your JSON file path
csv_file_path = r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Control_Wright.csv'    # Replace with your desired CSV file path
process_json_to_csv(json_file_path, csv_file_path)