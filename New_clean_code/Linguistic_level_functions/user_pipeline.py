from colorama import Fore, Style
import os
import re
import pandas as pd 
import spacy
import csv
from calculate_tree_depth import process_csv_file_for_tree_depth, calculate_tree_depth
from calculate_lexical_density import calculate_lexical_density, lexical_density_process_csv_file
from stutter_detection_function import word_stutter, find_unknown_words, syllable_stutter, add_stutter_metrics_to_single_file
from ..Model.interpretation_metrics import *


#### Initial input file ####
 
def request_txt_file():
    """
    Prompts the user to input the path of a .txt file for analysis.

    The function checks if the provided file path exists and whether the file is a .txt file.
    If the file is not found or is not a .txt file, the user is informed accordingly.
    If a .txt file is provided, its path is returned for further processing.

    Returns:
        str: The path to the .txt file if the file is valid.
        None: If the file does not exist or is not a .txt file.

    Note:
        If the file is not a .txt, the user is asked to convert it to .txt format manually.
    """
    
    # Prompt the user to enter the file path
    file_path = input(f'''
                        Welcome to {Fore.GREEN}CarePhasia{Style.RESET_ALL}, your virtual assistant to assess if a 
                        person has linguistic patterns compatible with Aphasia. Remember, {Fore.RED}this is not a diagnosis{Style.RESET_ALL}
                        A diagnosis needs to be performed by health professionals. Some important tips to achieve a good execution:
                        1. The text will be segmentated into {Fore.LIGHTMAGENTA_EX} sentences {Style.RESET_ALL}. For that reason, is important
                        that you separate your text or transcription according to ortographic symbols. At this satage of development, that separation
                        will ensure {Fore.LIGHTMAGENTA_EX} a good analysis {Style.RESET_ALL}
                        2. This algorithm takes into account three main linguistic patterns: {Fore.YELLOW} depth of the syntactic tree, lexical density 
                        and presence of sttutering {Style.RESET_ALL}. That's why we only need you to use the ortographic symbols to divide sentences
                        and clauses, but not perform any adittional normalizing process of speech
                        3. With this algorithm we will also provide {Fore.GREEN} sentiment analysis performance {Style.RESET_ALL} as a suplementary indicative 
                        of the mental state of the person in the speech

                        
                        Please, write the path to a {Fore.BLUE}.txt file{Style.RESET_ALL}
                        with the speech you would like to analyze: ''')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print("The file does not exist. Please ensure the correct path was entered.")
        return None
    
    # Check if the file has a .txt extension
    if not file_path.lower().endswith('.txt'):
        print("The file is not a .txt. Please convert it to TXT format and upload again.")
        return None
    
    # If everything is correct, return the file path
    print("TXT file path successfully obtained.")
    return file_path



#### Clean symbols ####

def clean_text(text):
    """
    Removes numbers and symbols from the text, leaving only alphabetic characters and whitespace.
    
    Parameters:
        text (str): The input text string that needs to be cleaned.

    Returns:
        str: A cleaned version of the input text with numbers and symbols removed.
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

#### Clause segmentation ####

nlp = spacy.load("en_core_web_sm")

def segment_and_save_sentences(input_path, output_path):
    """
    This function segments sentences from a given text file using SpaCy's NLP model 
    and saves the segmented sentences into a CSV file.

    Args:
    input_path (str): The file path to the input text file.
    output_path (str): The file path to the output CSV file.

    Returns:
    str: The file path of the created CSV file.
    """
    
    # Load the English SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Read the content of the .txt file
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Process the text with the NLP model to segment into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the sentences in a CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])  # Write the header
        for sentence in sentences:
            writer.writerow([sentence])

    return output_path

#### First step: Data preprocessing of .txt file ####

user_file_path = request_txt_file()

sentence_segmentated_raw = segment_and_save_sentences(user_file_path, r'New_clean_code\User_pipeline\raw_sentences.csv')

#### Second step: Cleaning and normalizing of .csv file ####

def clean_csv(input_csv_path, output_csv_path):
    """
    This function reads a CSV file, removes empty rows and quotes from sentences,
    and saves the cleaned data to a new CSV file.

    Args:
    input_csv_path (str): The file path to the input CSV file.
    output_csv_path (str): The file path to the output cleaned CSV file.
    """
    
    cleaned_data = []

    # Read the CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            # Ensure the row is not empty and remove quotes and extra whitespace
            cleaned_sentence = row[0].replace('"', '').replace("'", "").strip()
            # Add to the list if the cleaned sentence is not empty
            if cleaned_sentence:
                cleaned_data.append([cleaned_sentence])

    # Write the cleaned data to a new CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])  # Write the header
        writer.writerows(cleaned_data)

    return output_csv_path

ready_to_process_text = clean_csv(sentence_segmentated_raw, r'New_clean_code\User_pipeline\clean_text.csv')

#### Third step: Linguistic level functions processing ####

lexical_density_process_csv_file(ready_to_process_text, r'New_clean_code\User_pipeline\tagged_user_data.csv', calculate_lexical_density)
process_csv_file_for_tree_depth(r'New_clean_code\User_pipeline\tagged_user_data.csv\tagged_user_data.csv', calculate_tree_depth)

#### Fourth step: Calculate averages #### 

def calculate_averages_and_save(csv_path, output_file):
    """
    Calculate the average values of 'lexical_density' and 'depth_tree' columns from a given CSV file
    and save these averages in a new CSV file.

    Args:
    - csv_path (str): Path to the input CSV file.
    - output_file (str): Path to the output CSV file where averages will be saved.

    The new CSV file will contain two columns: 'average_lexical_density' and 'average_depth_tree'.
    """

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Calculate the average values
    average_lexical_density = df['lexical_density'].mean()
    average_tree_depth = df['tree_depth'].mean()

    # Create a new DataFrame with average values
    avg_df = pd.DataFrame({
        'average_lexical_density': [average_lexical_density],
        'average_tree_depth': [average_tree_depth]
    })

    # Save the averages to a new CSV file
    avg_df.to_csv(output_file, index=False)
    print(f"Averages saved to {output_file}")

calculate_averages_and_save(r'New_clean_code\User_pipeline\tagged_user_data.csv\tagged_user_data.csv', r'New_clean_code\User_pipeline\general_averages_user.csv')

#### Fifth step: Add stutter metrics (they work at a whole text level instead of single lines) ####

add_stutter_metrics_to_single_file(r'New_clean_code\User_pipeline\general_averages_user.csv', r'New_clean_code\User_pipeline\clean_text.csv')
