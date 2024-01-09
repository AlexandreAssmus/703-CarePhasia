import os
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

def calculate_tree_depth(text):
    """
    Calculate the maximum depth of the syntactic tree of a given text.

    Parameters:
    - text (str): A single string representing a clause or sentence.

    Returns:
    - int: The maximum depth of the syntactic tree.
    """
    # Process the text with spaCy
    doc = nlp(text)

    # Function to calculate the depth of a node
    def node_depth(node):
        if not list(node.children):
            return 0
        else:
            return 1 + max(node_depth(child) for child in node.children)

    # Find the root of the clause and calculate its depth
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is None:
        return 0

    return node_depth(root)

def process_csv_files_for_tree_depth(directory):
    """
    Process all CSV files in the specified directory and its subdirectories,
    adding a 'tree_depth' column that contains the tree depth of each text entry.

    Parameters:
    - directory (str): The root directory containing the CSV files to be processed.

    This function updates each CSV file, adding a 'tree_depth' column.
    """
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                # Apply the tree depth calculation to the 'text' column
                df['tree_depth'] = df['text'].apply(calculate_tree_depth)

                # Save the updated DataFrame to the same file
                df.to_csv(file_path, index=False)


#process_csv_files_for_tree_depth('New_clean_code\Data\Tagged_full_data')

#### Function for user only ####

def process_csv_file_for_tree_depth(csv_path, calculate_tree_depth):
    """
    Read the specified CSV file and add a 'tree_depth' column that contains the tree depth
    of each text entry.

    Parameters:
    - csv_path (str): The file path of the CSV file to be processed.
    - calculate_tree_depth (function): The function to be applied to calculate the depth
      of the parse tree for each sentence.

    This function updates the specified CSV file, adding a 'tree_depth' column.
    """

    # Check if the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Apply the tree depth calculation to the 'text' column
    df['tree_depth'] = df['text'].apply(calculate_tree_depth)

    # Save the updated DataFrame to the same file
    df.to_csv(csv_path, index=False)

    print(f"Updated file with 'tree_depth' column saved to {csv_path}")