# importing the data
import pandas as pd

df_control = pd.read_csv('../control_data_clean.csv')
df_aphasic = pd.read_csv('../patient_data_clean.csv')

print(df_aphasic.head())

# defining the regular expression describing stuttering
import re

pattern = r'\b(\w+)\s+\1\b'

# counting how many occurrences of that pattern found in each dataset

rep_control = 0
rep_aphasic = 0





# # Function to check if a sentence contains consecutive repeated words
# def contains_repeated_words(sentence):
#     pattern = r'\b(\w+)\s+\1\b'
#     return bool(re.search(pattern, sentence, flags=re.IGNORECASE))

# # Filter and count sentences with consecutive repeated words
# sentences_with_repeated_words = df_control[df_control.apply(contains_repeated_words)]
# count_sentences_with_repeated_words = len(sentences_with_repeated_words)

# # Display sentences and count
# print("Sentences with consecutive repeated words:")
# for index, row in sentences_with_repeated_words.iterrows():
#     print(row['Sentence'])

# print("\nTotal count of sentences with consecutive repeated words:", count_sentences_with_repeated_words)

