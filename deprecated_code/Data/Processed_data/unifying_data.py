import pandas as pd 

df_morphology_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\morphology_data_processed_patients.csv')
df_morphology_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\morphology_data_processed_controls.csv')
df_semantic_similarity_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_similarity_processed_patient.csv')
df_semantic_similarity_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_similarity_processed_control.csv')
df_sentiment_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_data_processed_patients.csv')
df_sentiment_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\semantic_data_processed_control.csv')
df_syntax_patients = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\syntax_data_processed.csv')
df_syntax_controls = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Processed_data\syntax_data_processed_control.csv')


def unify_dataframes(df_list):

    df_unificado = df_list[0]


    for df in df_list[1:]:
        df_unificado = pd.merge(df_unificado, df, on='text', how='outer')

    return df_unificado

#df_unified_patients = unify_dataframes([df_morphology_patients,df_semantic_similarity_patients,df_sentiment_patients,df_syntax_patients])
#print(df_unified_patients.head()) #The dataframes are to big to be unified this way. 


