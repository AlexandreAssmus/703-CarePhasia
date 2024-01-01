import pandas as pd
import os

def consolidate_and_preprocess_data(directory):
    """
    Consolidates individual CSV files containing patient and control data into a single DataFrame.
    It ensures that required columns are present, handles missing values, and saves the 
    cleaned DataFrame to a new CSV file.

    Parameters:
    - directory: str. The path to the directory containing subdirectories of CSV files.

    Returns:
    - full_data: DataFrame. The consolidated data.

    Saves the processed data to 'unify_tagged_data.csv' in the given directory.
    """
    
    # Initialize a list to store DataFrames
    dataframes_list = []

    # Walk through the directory to access all CSV files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Read the CSV file and store it in a DataFrame
                df = pd.read_csv(file_path)
                # Append the DataFrame to the list
                dataframes_list.append(df)

    # Concatenate all the DataFrames in the list to create a full DataFrame
    full_data = pd.concat(dataframes_list, ignore_index=True)

    # Ensure the necessary columns are present
    required_columns = ['text', 'diagnosis', 'lexical_density', 'tree_depth']
    full_data = full_data[required_columns]

    # Handle missing values if any
    full_data.dropna(inplace=True)

    # Save the cleaned DataFrame to a new CSV file
    output_path = os.path.join(directory, 'unify_tagged_data.csv')
    full_data.to_csv(output_path, index=False)
    
    return full_data

# Usage example
# Replace 'your_directory_path' with the actual path to your directory
consolidated_data = consolidate_and_preprocess_data('New_clean_code\Data\Tagged_full_data')

