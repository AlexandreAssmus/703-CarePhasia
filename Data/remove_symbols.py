#Import libraries
import pandas as pd

'''
The following functions were used to process the data in their final format, removing all symbols 
that were non alphabetical. 
'''

def process_text_column(df, column_name):
    '''
    This function processes a specified text column in a DataFrame.
    It first removes all symbols that are not letters or white spaces
    from the specified column. Then, it removes repeated spaces.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the text column.
    column_name (str): The name of the column to be processed.

    Returns:
    pandas.DataFrame: The DataFrame with the processed text column.
    '''
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].str.replace('[^a-zA-Z\s]', '', regex=True)

    def remove_repeated_spaces(text):
        return ' '.join(text.split())

    df[column_name] = df[column_name].apply(remove_repeated_spaces)

    return df

def process_text_column(df, column_name):
    '''
    This function eliminates blank rows in the target column. At the same time,
    eliminates double spaces between words for each row.
    Parameters:
    df(pandas.DataFrame): The dataframe containing the text column.
    column_bame(str): The name of the column to be processed
    Returns:
    pandas.DataFrame: The dataFrame with the processed rows.
    '''
    # Check for NaN values and replace them with an empty string
    df[column_name] = df[column_name].apply(lambda x: '' if pd.isna(x) else x)

    # Fonction pour éliminer les espaces répétés
    def remove_repeated_spaces(text):
        return ' '.join(text.split())

    # Aplicar la función para eliminar espacios repetidos en cada fila
    df[column_name] = df[column_name].apply(remove_repeated_spaces)

    return df

def download_csv(df, filename):
    '''
    Save a DataFrame as a CSV file in the local machine.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.
    filename (str): The name of the file, should end in '.csv'.
    '''
    df.to_csv(filename, index=False)