import json
import re
import os

def parse_cha_file(cha_file):
    """
    Parses a .cha file and converts it to a structured JSON format.

    Parameters:
    cha_file (str): Path to the .cha file.

    Returns:
    list: A list of dictionaries, each representing a line in the .cha file.
    """
    data = []
    current_entry = None

    with open(cha_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('*'):
                if current_entry:
                    data.append(current_entry)
                current_entry = {'type': '', 'text': '', 'mor': '', 'gra': '', 'wor': ''}
                current_entry['type'] = line.split(':')[0].strip('*')
                current_entry['text'] = line.split(':')[1].strip()
            elif line.startswith('%mor:'):
                current_entry['mor'] = line.split(':', 1)[1].strip()
            elif line.startswith('%gra:'):
                current_entry['gra'] = line.split(':', 1)[1].strip()
            elif line.startswith('%wor:'):
                current_entry['wor'] = line.split(':', 1)[1].strip()

        # Add the last entry if exists
        if current_entry:
            data.append(current_entry)

    return data

def save_to_json(data, json_file):
    """
    Saves the parsed data to a JSON file.

    Parameters:
    data (list): Parsed data to be saved.
    json_file (str): Path for the output JSON file.

    Returns:
    None
    """
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_directories(root_directory, output_root):
    """
    Walks through the directories starting from root_directory, processes all .cha files,
    and saves the structured data to JSON files mirroring the input directory structure in output_root.

    Parameters:
    root_directory (str): The root directory to start processing .cha files.
    output_root (str): The root directory where the JSON files will be saved.

    Returns:
    None
    """
    for dirpath, dirnames, files in os.walk(root_directory):
        # Skip directories that do not contain .cha files
        if not any(file.endswith('.cha') for file in files):
            continue

        # Create a corresponding directory structure in the output_root
        relative_path = os.path.relpath(dirpath, root_directory)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.endswith('.cha'):
                cha_file = os.path.join(dirpath, file)
                parsed_data = parse_cha_file(cha_file)
                
                # Create a JSON filename that corresponds to the .cha file
                json_filename = os.path.splitext(file)[0] + '.json'
                json_file = os.path.join(output_dir, json_filename)
                
                save_to_json(parsed_data, json_file)
                print(f"Processed {cha_file} to {json_file}")

# Example usage
root_directory = r'New_clean_code\Data\Raw_data'  # Replace with the path to your 'raw_data' directory
output_root = r'New_clean_code\Data\Full_Json'       # Replace with the path to where you want to save the JSON files
process_directories(root_directory, output_root)




