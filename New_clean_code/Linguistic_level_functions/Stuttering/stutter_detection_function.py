import pandas as pd
import re
from tqdm import tqdm
import os 


# WORD REPETITION -------------------------------------------------------------------------------------------

def word_stutter(data):
    """ counts the number of sentences that have a consecutive repetition of one same word (twice or more)"""

    # regex definition for consecutive repetition of word
    pattern = r'\b(\w+)\s+(\1\s+)*\1\b'

    # setting iterator of sentences with stutter
    nb = 0

    # going through dataset
    for line in data['text']:
        if isinstance(line, (str, bytes)):

            # checking wether pattern appears
            if re.search(pattern, line, flags=re.IGNORECASE):
                nb+=1
    return nb


# SYLLABLE REPETITION ----------------------------------------------------------------------------------------


with open("New_clean_code\Linguistic_level_functions\Stuttering\en_US-large.txt",'r') as file:
    english_words = file.read().splitlines()

def find_unknown_words(data):
    """ returns all the words that aren't recognized as English words in the dataset data"""
    nb_words = 0
    unknown_words = []
    for line in tqdm(data['text']):
        line = str(line)
        words = line.split()
        nb_words += len(words)
        for word in words:
            if word not in english_words:
                unknown_words.append(word)
    return unknown_words,nb_words

def find_syllable_repetition(words):
    """ returns, out of the list of words passed as argument, the ones in which a set of letters is consecutively repeated at least twice"""
    words_with_repetition = []
    pattern = r"(\w+)\1+"
    for word in words:
        if re.search(pattern, word):
            words_with_repetition.append(word)
    return words_with_repetition

def syllable_stutter(data):
    list_of_unknown_words, total_nb_words = find_unknown_words(data)
    list_stutter_words = find_syllable_repetition(list_of_unknown_words)
    nb_stutter_words = len(list_stutter_words)
    nb_unknown_words = len(list_of_unknown_words)
    nb_of_known_words = total_nb_words - nb_unknown_words

    # Verify if the number of unknown words is zero
    if nb_unknown_words == 0:
        return 0  # If it is, return 0.

    return nb_stutter_words / nb_unknown_words


# ADD METRICS TO MAIN FILE -------------------------------------------------------------------------------------

def add_stutter_metrics_to_thresholds(thresholds_file, source_directory):
    """
    Add word stutter count and syllable stutter ratio metrics to the existing thresholds_per_file.csv.

    Parameters:
    - thresholds_file (str): Path to the thresholds_per_file.csv file.
    - source_directory (str): Root directory containing the original CSV files to be analyzed.
    """
    # Read the existing thresholds file
    thresholds_df = pd.read_csv(thresholds_file)

    # Prepare a dictionary to store results of new metrics
    stutter_metrics = {}

    # Traverse through each CSV file in the source directory
    for subdir, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                # Calculate new metrics
                word_stutter_count = word_stutter(df)
                syllable_stutter_ratio = syllable_stutter(df)

                # Store the results in the dictionary
                stutter_metrics[file] = {
                    'word_stutter_count': word_stutter_count,
                    'syllable_stutter_ratio': syllable_stutter_ratio
                }

    # Add the results to the thresholds DataFrame
    for index, row in thresholds_df.iterrows():
        file_name = row['file_name']
        if file_name in stutter_metrics:
            thresholds_df.loc[index, 'word_stutter_count'] = stutter_metrics[file_name]['word_stutter_count']
            thresholds_df.loc[index, 'syllable_stutter_ratio'] = stutter_metrics[file_name]['syllable_stutter_ratio']

    # Save the updated DataFrame
    thresholds_df.to_csv(thresholds_file, index=False)

# Example usage
add_stutter_metrics_to_thresholds(r'New_clean_code\Data\thresholds_per_file.csv', r'New_clean_code\Data\Tagged_full_data')