import pandas as pd 
from collections import Counter


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
# Function to calculate the average number of clauses per text
def average_clauses_per_text(dataframe):
    # Count the number of clauses in each row and then calculate the average
    clause_counts = dataframe['Split_Clauses'].apply(len)
    return clause_counts.mean()


# Function to count the frequencies of word order patterns and calculate percentages
def word_order_frequencies(dataframe):
    # Flatten the list of lists in 'Word_Order' to count all occurrences
    all_word_orders = [order for sublist in dataframe['Word_Order'] for order in sublist]
    counts = Counter(all_word_orders)

    # Calculate the percentage for each type of word order
    total = sum(counts.values())
    percentages = {order: (count / total) * 100 for order, count in counts.items()}

    return percentages


# Function to perform statistical analysis on columns with integer values
def analyze_numeric_columns(dataframe, column_name):
    # Flatten the column if it contains lists, otherwise use the column as is
    if isinstance(dataframe[column_name].iloc[0], list):
        flattened_values = [item for sublist in dataframe[column_name] for item in sublist]
    else:
        flattened_values = dataframe[column_name]

    # Convert to pandas Series to use its statistical functions
    series = pd.Series(flattened_values)

    # Perform statistical calculations
    stats = {
        'mean': series.mean(),
        'median': series.median(),
        'std_dev': series.std(),
        'min': series.min(),
        'max': series.max()
    }

    return stats


#Most prevalent word_order per clauses
word_order_percentages_patients = word_order_frequencies(df_syntax_patients)

#Analysis for average lenght of clauses
average_clauses_patients = average_clauses_per_text(df_syntax_patients)

# Analysis for 'Word_Counts'
#word_counts_stats_patients = analyze_numeric_columns(df_syntax_patients, 'Word_Counts')

# Analysis for 'Max_Syntax_Tree_Depth'
#max_syntax_depth_stats_patients = analyze_numeric_columns(df_syntax_patients, 'Max_Syntax_Tree_Depth')

# Analysis for 'Verb_Phrases'
#verb_phrases_stats_patients = analyze_numeric_columns(df_syntax_patients, 'Verb_Phrases')

print(f'''
The word order analyzed in the text is: {word_order_percentages_patients}
The average amount of clauses analyzed in the text is: {average_clauses_patients}
The average length of the clauses in the text is: 
The average depth of syntax trees in the text is: 
The average amount of verb phrases founded in the text is: 

''')
