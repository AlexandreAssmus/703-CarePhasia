from colorama import Fore, Style

import os
import re
import pandas as pd 
import spacy

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

def process_text_file_to_csv(file_path, csv_output_path):
    """
    Processes a text file using spaCy to extract sentences and save them in a CSV file.

    Parameters:
        file_path (str): Path to the .txt file.
        csv_output_path (str): Path where the CSV file will be saved.
    """
    # Leer el contenido del archivo .txt
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Procesar el texto con spaCy
    doc = nlp(text)

    # Extraer las oraciones
    sentences = [sent.text.strip() for sent in doc.sents]

    # Crear un DataFrame con las oraciones
    df = pd.DataFrame({'sentence': sentences})

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(csv_output_path, index=False)

#### First step: Data preprocessing of .txt file ####

user_file_path = request_txt_file()

if user_file_path:
    with open(user_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        cleaned_content = clean_text(content)
        # Guarda directamente en un archivo CSV
        process_text_file_to_csv(cleaned_content, r'New_clean_code\User_pipeline\clause_file.csv')