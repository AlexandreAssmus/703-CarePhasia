import os
import re
import json

# Function to process a .cha file
def process_cha_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_object = None
        for line in file:
            line = line.strip()
            if line.startswith('*INV') or line.startswith ('*PAR'):
                if current_object:
                    data.append(current_object)
                current_object = {
                    "type": line.split(':')[0][1:],
                    "text": line.split(':')[1].strip(),
                    "mor": "",
                    "gra": "",
                    "wor": ""
                }
            elif current_object:
                if line.startswith('%mor'):
                    current_object["mor"] = line[5:]
                elif line.startswith('%gra'):
                    current_object["gra"] = line[5:]
                elif line.startswith('%wor'):
                    current_object["wor"] = line[5:]
        if current_object:
            data.append(current_object)
    return data

# Process all .cha files in a directory
def process_cha_directory(directory_path):
    json_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".cha"):
            file_path = os.path.join(directory_path, filename)
            data = process_cha_file(file_path)
            json_data.extend(data)
    return json_data

# Convert the data to JSON and save to a file
def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# Example usage
input_directory = r"C:\\Usersbelen\\Desktop\\Aphasia\\Patients"
output_json_file = 'output.json'

data = process_cha_directory(input_directory)
save_to_json(data, output_json_file)

