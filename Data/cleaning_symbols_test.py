import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Individual_groupPatient_data\Control_Baycrest copy.csv')

# Aplicar la expresión regular para eliminar las anotaciones
data['text'] = data['text'].str.replace(r'&[^ ]*', '', regex=True)

data['text'] = data['text'].str.findall(r"[A-Za-z']+(?:'[A-Za-z]+)?").str.join(' ')

# Guardar el archivo limpio si es necesario
data.to_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\test\test_clean_baycrest.csv', index=False)