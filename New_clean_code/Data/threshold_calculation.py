import os
import pandas as pd

def calculate_averages_with_file_name(source_directory, destination_directory):
    """
    Calculate the average lexical density and tree depth for each diagnosis category
    across all CSV files in the specified source directory and its subdirectories, and include
    the file name from which each average is calculated. Save the result in the destination directory.

    Parameters:
    - source_directory (str): The root directory containing the CSV files.
    - destination_directory (str): The directory where the result file will be saved.

    The function saves the averages in a file named 'thresholds_per_file.csv' in the destination directory.
    """
    averages = []

    for subdir, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                # Calculate the averages for each diagnosis category and include the file name
                for diagnosis in df['diagnosis'].unique():
                    diagnosis_df = df[df['diagnosis'] == diagnosis]
                    avg_tree_depth = diagnosis_df['tree_depth'].mean()
                    avg_lexical_density = diagnosis_df['lexical_density'].mean()

                    averages.append({
                        'file_name': file,
                        'metric': 'average',
                        'diagnosis': diagnosis,
                        'average_tree_depth': avg_tree_depth,
                        'average_lexical_density': avg_lexical_density
                    })

    # Create a DataFrame from the averages and save it to the specified destination directory
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    averages_df = pd.DataFrame(averages)
    output_file = os.path.join(destination_directory, 'thresholds_per_file_1.csv')
    averages_df.to_csv(output_file, index=False)

# Example usage
#calculate_averages_with_file_name('New_clean_code\Data\Tagged_full_data', 'New_clean_code\Data')

def calculate_general_averages(source_csv_file, destination_directory):
    """
    Calculate the general average of 'average_tree_depth' and 'average_lexical_density'
    for 'patient' and 'control' groups from the provided CSV file. Save the result in the destination directory.

    Parameters:
    - source_csv_file (str): The path to the CSV file containing the data.
    - destination_directory (str): The directory where the result file will be saved.

    The function saves the general averages in a file named 'general_thresholds.csv' in the destination directory.

    This function will be used to generate the threshold of our model.
    """
    # Read the CSV file
    df = pd.read_csv(source_csv_file)

    # Initialize a dictionary to store the general averages
    general_averages = {'patient': {}, 'control': {}}

    # Calculate the general averages for 'patient' and 'control'
    for diagnosis in ['patient', 'control']:
        diagnosis_df = df[df['diagnosis'] == diagnosis]
        general_averages[diagnosis]['average_tree_depth'] = diagnosis_df['average_tree_depth'].mean()
        general_averages[diagnosis]['average_lexical_density'] = diagnosis_df['average_lexical_density'].mean()
        general_averages[diagnosis]['word_stutter_count'] = diagnosis_df['word_stutter_count'].mean()
        general_averages[diagnosis]['syllable_sttuter_ratio'] = diagnosis_df['syllable_stutter_ratio'].mean()

    # Convert the dictionary to a DataFrame and save it to the specified destination directory
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    general_averages_df = pd.DataFrame.from_dict(general_averages, orient='index').reset_index()
    general_averages_df.rename(columns={'index': 'diagnosis'}, inplace=True)
    output_file = os.path.join(destination_directory, 'general_thresholds.csv')
    general_averages_df.to_csv(output_file, index=False)

# Example usage
calculate_general_averages(r'New_clean_code\Data\thresholds_per_file.csv', 'New_clean_code\Data')
