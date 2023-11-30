import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df_semantic_similarity_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_similarity_processed_patient.csv')
df_semantic_similarity_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_similarity_processed_control.csv')


import pandas as pd

def calculate_column_average(df, column):
    # Filter the column to include only numeric values (int or float)
    numeric_values = df[column].apply(lambda x: x if isinstance(x, (int, float)) else None)

    # Calculate and return the average, excluding non-numeric values
    return numeric_values.mean()

semantic_similarity_patients = calculate_column_average(df_semantic_similarity_patients, 'intra_sentence_similarity')
semantic_similarity_controls = calculate_column_average(df_semantic_similarity_controls, 'intra_sentence_similarity')

'''
Sentiment_analysis
'''

df_sentiment_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_data_processed_patients.csv')
df_sentiment_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_data_processed_control.csv')

sentiment_patients = calculate_column_average(df_sentiment_patients, 'sentiment')
sentiment_controls = calculate_column_average(df_sentiment_controls, 'sentiment')

print(sentiment_patients)
print(sentiment_controls)

'''
Sttutering
'''