
import pandas as pd
import re
import csv
from textblob import TextBlob

# import of the data as csv files
#df_control = pd.read_csv('../control_data_clean.csv')
df_control = pd.read_csv('C:/Users/Etudiant/M1_TAL/703-CarePhasia/semantic_analysis/control_data_clean.csv')
#df_aphasic = pd.read_csv('../patient_data_clean.csv')
df_aphasic = pd.read_csv('C:/Users/Etudiant/M1_TAL/703-CarePhasia/semantic_analysis/patient_data_clean.csv')

#print(df_aphasic.head())




def nb_word_repetition(data):
    """ counts the number of sentences that have a consecutive repetition of one same word (twice or more)"""

    # regex definition for consecutive repetition of word
    pattern = r'\b(\w+)\s+(\1\s+)*\1\b'

    # setting iterator of sentences with stutter
    nb = 0

    # going through dataset
    for line in data['text']:
        if isinstance(line, (str, bytes)):

            # cheching wether pattern appears
            if re.search(pattern, line, flags=re.IGNORECASE):
                nb+=1
    return nb


print('number of sentences with word stuttering in control dataset:', nb_word_repetition(df_control))
print('number of sentences with word stuttering in aphasic dataset:', nb_word_repetition(df_aphasic))

percentage_control = nb_word_repetition(df_control)/df_control.shape[0] * 100
percentage_aphasic = nb_word_repetition(df_aphasic)/df_aphasic.shape[0] * 100
print('control:' , percentage_control, '%')
print('aphasic', percentage_aphasic, '%')





def nb_syllable_repetition(data):
    blob = TextBlob(data)
    blob_words_full_count = 0
    blob_words_full_list = []
    unknown_words = []
    for line in data['text']:
        str_line = str(line)
        blob = TextBlob(str_line)
        blob_words_full_list += blob.words
        blob_words_full_count += blob.word_counts

    for word in blob_words_full_list:
        if word not in blob_words_full_count:
            unknown_words.append(word)
    return(unknown_words)

print(nb_syllable_repetition(df_control))
print(nb_syllable_repetition(df_aphasic))