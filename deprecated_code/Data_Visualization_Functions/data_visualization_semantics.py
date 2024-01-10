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

'''
Visualization
'''
data = {
    'Metric': ['Intra_sentence_similarity', 'sentiment'],
    'Patients': [sentiment_patients, semantic_similarity_patients],
    'Controls': [sentiment_controls, semantic_similarity_controls]
}

df = pd.DataFrame(data)

df_long = pd.melt(df, id_vars=['Metric'], value_vars=['Patients', 'Controls'], var_name='Group', value_name='Value')


plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', hue='Group', data=df_long)
plt.title('Comparison of Linguistic Metrics Between Patients and Controls')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.tight_layout()
plt.show()