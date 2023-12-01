import pandas as pd 
import seaborn as sns 
from data_visualization_semantics import calculate_column_average
import matplotlib.pyplot as plt

df_morphology_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\morphology_data_processed_patients.csv')
df_morphology_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\morphology_data_processed_controls.csv')

morphology_patients = calculate_column_average(df_morphology_patients, 'Lexical_Density')
morphology_controls = calculate_column_average(df_morphology_controls, 'Lexical_Density')

data = {
    'Metric': ['Lexical_Density'],
    'Patients': [morphology_patients],
    'Controls': [morphology_controls]
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
