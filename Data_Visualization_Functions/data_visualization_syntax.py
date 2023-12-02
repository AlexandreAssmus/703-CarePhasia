import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns 


'''
Dataset importation
'''

df_syntax_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\syntax_data_processed.csv')

#print(df_syntax_patients.head())

df_syntax_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\syntax_data_processed_control.csv')

#print(df_syntax_controls.head())

'''
Data Analysis
'''

def average_clauses(column):
    """
    Calculate the average length of lists within a pandas Series (column).
    
    Args:
    column (pd.Series): A pandas Series where each row contains a list.
    
    Returns:
    float: The average length of the lists in the column.
    """
    # Check that the input is a pandas Series
    if not isinstance(column, pd.Series):
        raise ValueError("The input must be a pandas Series.")
        
    # Calculate the average length of lists in the column
    average_length = column.apply(len).mean()
    return average_length

import ast
# Function to count the frequencies of word order patterns and calculate percentages
def calculate_frequencies(df_column):
    
    evaluated_lists = df_column.apply(ast.literal_eval)
    
    
    all_elements = [element for sublist in evaluated_lists for element in sublist]
    
    
    unique_elements = set(all_elements)
    total = len(all_elements)
    frequencies = {element: all_elements.count(element) / total for element in unique_elements}
    
    return frequencies

def calculate_average_max_syntax_depth_tree(df, column):
    # Check if the specified column contains only integer values
    if df[column].dtype == 'int64':
        # Calculate and return the mean (average) of the column
        return df[column].mean()
    else:
        # Return a message if the column does not contain only integers
        return "The column does not contain only integer values"
    
def calculate_average_from_list_column(df, column_name):

    all_numbers = []


    for sublist in df[column_name]:
        for item in sublist:

            try:
                number = int(item)
                all_numbers.append(number)
            except ValueError:

                pass


    return sum(all_numbers) / len(all_numbers) if all_numbers else None

'''
Functions execution and presentation
'''

average_clauses_patients = average_clauses(df_syntax_patients['Word_Counts'])
average_clauses_control = average_clauses(df_syntax_controls['Word_Counts'])
word_frequency_patients = calculate_frequencies(df_syntax_patients['Word_Order'])
word_frequency_controls = calculate_frequencies(df_syntax_controls['Word_Order'])
depth_tree_patients = calculate_average_max_syntax_depth_tree(df_syntax_patients, 'Max_Syntax_Tree_Depth')
depth_tree_controls = calculate_average_max_syntax_depth_tree(df_syntax_controls, 'Max_Syntax_Tree_Depth')
verb_phrases_patients = calculate_average_from_list_column(df_syntax_patients, 'Verb_Phrases')
verb_phrases_controls = calculate_average_from_list_column(df_syntax_controls, 'Verb_Phrases')
clause_length_patients = calculate_average_from_list_column(df_syntax_patients, 'Word_Counts')
clause_length_controls = calculate_average_from_list_column(df_syntax_controls, 'Word_Counts')

data = {
    'Metric': ['Average Clauses', 'Depth Tree', 'Verb Phrases', 'Clause Length'],
    'Patients': [average_clauses_patients, depth_tree_patients, verb_phrases_patients, clause_length_patients],
    'Controls': [average_clauses_control, depth_tree_controls, verb_phrases_controls, clause_length_controls]
}

df = pd.DataFrame(data)

df_long = pd.melt(df, id_vars=['Metric'], value_vars=['Patients', 'Controls'], var_name='Group', value_name='Value')


plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Metric', y='Value', hue='Group', data=df_long)
plt.title('Comparison of Linguistic Metrics Between Patients and Controls')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend(title='Group')

for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 10), 
                      textcoords = 'offset points')

plt.tight_layout()
plt.show()

print(f'The average word order distribution for controls is: {word_frequency_controls}')
print(f'The average word order dsitribution for patients is: {word_frequency_patients}')