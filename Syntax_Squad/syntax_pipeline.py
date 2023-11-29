from clause_length import *
from depth_tree import *
from verb_phrases import *
from word_order import *
import spacy
import pandas as pd 

'''
Patient data syntax processing

'''

#Process inicial dataframe
df_raw = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Cleaned_files\patient_data_clean.csv')

processed_df = split_into_clauses(df_raw, 'text')

#Clause_length metric
processed_df['Word_Counts'] = processed_df['Split_Clauses'].apply(count_words_in_clause_list_and_return_list)

#Depth_tree metric
processed_df['Max_Syntax_Tree_Depth'] = processed_df['Split_Clauses'].apply(calculate_max_tree_depth)

#Word_order metric
processed_df['Word_Order'] = processed_df['Split_Clauses'].apply(get_word_order_for_clauses)

#Verb_phrases metric
processed_df['Verb_Phrases'] = processed_df['Split_Clauses'].apply(get_verb_phrase_counts_for_clauses)

print(processed_df.head())

#Save data
path= r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\syntax_data_processed.csv'

processed_df.to_csv(path, index=False)
