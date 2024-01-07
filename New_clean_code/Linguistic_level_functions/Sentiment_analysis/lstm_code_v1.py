import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

'''
DISCLAIMER:

To use the embeddings download the file 'glove.6B.100d.txt'
Link:  https://www.kaggle.com/datasets/anindya2906/glove6b?select=glove.6B.100d.txt

'''


"""**Create combined csv files**

def process_folder(main_folder_path, group_label):
   Processes all CSV files in a folder and returns a merged DataFrame.
    frames = []  # List to store the DataFrames of each file
    for subdir, dirs, files in os.walk(main_folder_path):
      for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(subdir, filename)
            df = pd.read_csv(file_path)
            df['conversation_id'] = filename.split('.')[0]  # Use file name as ID
            df['group'] = group_label
            df = df[['text', 'conversation_id', 'group']]
            frames.append(df)
    return pd.concat(frames)

# Folder paths for control and patient CSV files
control_main_folder_path = "Control_csv_data_clean"
patient_main_folder_path = "Patient_csv_data_clean"

# Processing files and merging DataFrames
control_df = process_folder(control_main_folder_path, 'control')
patient_df = process_folder(patient_main_folder_path, 'patient')
combined_df = pd.concat([control_df, patient_df])

# Saving the merged DataFrame to a new CSV file
combined_df.to_csv('combined_control_patient_data.csv', index=False)

**Step 1 : Data preprocessing and split**
"""

# load the data
data = pd.read_csv('New_clean_code\Data\CSV_clean\combined_control_patient_data.csv')
data_df = pd.DataFrame(data) # DataFrame

# Text data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9']", " ", text) # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text)  # Remove extra white spaces
    return text

# Text cleaning
data_df['text'] = data_df['text'].apply(clean_text)

# Function to calculate sentiment score
def sentiment_score(text):
    analysis = TextBlob(text) # Creating a TextBlob object from the given text
    return analysis.sentiment.polarity  # Returns the sentiment polarity calculated by TextBlob, the polarity is a floating-point score ranging from -1 (negative) to 1 (positive)

# Apply the sentiment score calculation function to each text in the 'text' column of the DataFrame
#The result is stored in a new column 'sentiment' in the DataFrame
data_df['sentiment'] = data_df['text'].apply(sentiment_score)

print(data_df)

# Grouping data by conversation_id
conversation_ids = data['conversation_id'].unique() # Extract unique conversation IDs from the data
np.random.shuffle(conversation_ids) # Shuffle the conversation IDs for randomness

# Splitting conversation IDs for training, validation, and testing
train_size = int(0.8 * len(conversation_ids)) # 80 % of data for training
#val_size = int(0.1 * len(conversation_ids)) # 10% of data for validation
test_size = int(0.2 * len(conversation_ids))  # 20% of data for testing

# Assigning conversation IDs to each dataset
train_ids = conversation_ids[:train_size] # IDs for training set
#val_ids = conversation_ids[train_size:train_size + val_size] #IDs for validation set
test_ids = conversation_ids[train_size:] #IDS for test set

# Selecting the training, validation, and test sets based on conversation IDs
train_df = data[data['conversation_id'].isin(train_ids)] # Training data frame
#val_df = data[data['conversation_id'].isin(val_ids)] # Validation data frame
test_df = data[data['conversation_id'].isin(test_ids)] #test data frame

# Tokenization and Padding
tokenizer = Tokenizer(num_words=12385) # Initialize tokenizer with a maximum number of words
tokenizer.fit_on_texts(data['text']) # Fit tokenizer on the text data

# Function to tokenize and pad a DataFrame
max_seq_length = 15 # Setting the maximum sequence length for padding
def prepare_data(df):
    sequences = tokenizer.texts_to_sequences(df['text']) # Tokenize the text
    return pad_sequences(sequences, maxlen=max_seq_length) # Return padded sequences

"""**How to get num_words and max_seq_length**"""

# Initializing a tokenizer without limiting the number of words
tokenizer = Tokenizer() # Create a new Tokenizer instance
tokenizer.fit_on_texts(data_df['text']) # Fit the tokenizer on the text data

# Calculating the total number of unique words in the dataset
word_count = len(tokenizer.word_index)  # Get the length of the word_index
print(f"Nombre total de mots uniques dans le jeu de données : {word_count}") # Print the total number of unique words

text_lengths = [len(text.split()) for text in data_df['text']]
print(f"Moyenne: {np.mean(text_lengths)}")
print(f"Médiane: {np.median(text_lengths)}")
print(f"Percentile 90: {np.percentile(text_lengths, 90)}")

"""**Step 2 : Word Embeddings preparation**"""

# Loading GloVe embeddings
EMBEDDING_DIM = 100  # for GloVe 6B with 100-dimensional vectors
embedding_index = {}
with open('New_clean_code\Linguistic_level_functions\Sentiment_analysis\glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Preparing the embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Creating the embedding layer using the embedding matrix
embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_seq_length,
                            trainable=False)

# control the embedding matrix
print("Embeddingx matrix dimension :", embedding_matrix.shape)

"""**Coverage Verification**"""

# Calculating the percentage of words in the tokenizer that are covered by GloVe embeddings
total_words = len(tokenizer.word_index) # Get the total number of unique words in the tokenizer
words_in_glove = sum(1 for word in tokenizer.word_index if word in embedding_index) # Count how many of these words are present in the GloVe embeddings
coverage_percentage = (words_in_glove / total_words) * 100 # Calculate the coverage percentage: the ratio of words in GloVe to the total words in the tokenizer
print(f"Percentage of tokenizer words covered by GloVe : {coverage_percentage}%")
# 89 % --> good coverage

"""**Step 3 : LSTM implementation, training and Cross-Validation Methodology**"""

# Initialize fold number and number of folds for cross-validation
fold_no = 1
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True) # Create KFold object with 3 splits and shuffling
# Initialize early stopping callback
#early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# Loop through each fold for cross-validation
for fold_no, (train_indices, test_indices) in enumerate(kfold.split(conversation_ids), start=1): # Extract train and test IDs for the current fold
    # creating an LSTM model and reinitialize it for each fold
    model = Sequential() # Initialize a sequential model
    model.add(embedding_layer)  # Use the previously prepared embedding layer
    model.add(LSTM(64, return_sequences=True))  # First LSTM layer with fewer units and return sequences
    model.add(LSTM(128, return_sequences=False))  # return_sequences=False for the last LSTM layer, interested in a single output (overall sentiment score)
    model.add(Dropout(0.5)) # Add a Dropout layer to prevent overfitting
    model.add(Dense(1, activation='tanh'))  # Sentiment score between -1 and 1
    model.compile(optimizer='adam', loss='mean_squared_error') # adam is a popular choice and the loss function is mean squared error, typical for regression problems

    # Extract train and test IDs for the current fold
    train_ids = conversation_ids[train_indices]
    test_ids = conversation_ids[test_indices]
    #val_ids = conversation_ids[val]

    # Select train and test DataFrame based on the conversation IDs
    train_df = data[data['conversation_id'].isin(train_ids)]
    test_df = data[data['conversation_id'].isin(test_ids)]
    #val_df = data[data['conversation_id'].isin(val_ids)]

    # Prepare data for training and testing
    X_train_padded = prepare_data(train_df)
    y_train = train_df['sentiment'].values

    X_test_padded = prepare_data(test_df)
    y_test = test_df['sentiment'].values

    #X_val_padded = prepare_data(val_df)
    #y_val = val_df['sentiment'].values

    # Model training for the current fold
    print(f'Training for fold {fold_no} ...')
    model.fit(X_train_padded, y_train, epochs=5, batch_size=64)

    # Model Evaluation on the test set
    test_loss = model.evaluate(X_test_padded, y_test)
    print(f'Perte sur l\'ensemble de test pour le fold {fold_no}: {test_loss}')

    # Increment the fold number
    fold_no = fold_no + 1

"""**Dimensions verification**"""

#  dimensions of various data frame
print(f"Dimensions of X_train_padded: {X_train_padded.shape}")
print(f"Dimensions of y_train: {y_train.shape}")
print(f"Dimensions of X_test_padded: {X_test_padded.shape}")
print(f"Dimensions of y_test: {y_test.shape}")
print(f"Dimensions of train_ids: {train_ids.shape}")
print(f"Dimensions of test_ids: {test_ids.shape}")

"""**Model evaluation**"""

# Predicting sentiments on the test set
pred_sentiment = model.predict(X_test_padded) # Use the model to predict sentiments for the padded test data

# Calculating regression metrics
pred_sentiment = pred_sentiment.flatten()
mse = mean_squared_error(y_test, pred_sentiment) # Calculate Mean Squared Error between actual and predicted values
mae = mean_absolute_error(y_test, pred_sentiment) # Calculate Mean Absolute Error between actual and predicted values
rmse = np.sqrt(mse) # Calculate Root Mean Squared Error

print(f"MSE(Mean Squared Error): {(mse)}, MAE(Mean Absolute Error): {(mae)}, RMSE(Root Mean Squared Error): {(rmse)}")
#low MSE --> the model, on average, predicts values ​​quite close to the actual values.
#low MAE --> low average error per prediction
#low RMSE --> low dispersion of errors in the model predictions
#the model gives relatively accurate predictions with low errors on the test set.

"""**Contextualization of the metrics**"""

# Calculating the standard deviation of the actual sentiment score
std_dev_sentiment = np.std(data_df['sentiment'])
print(f"Standard deviation of actual sentiment scores: {std_dev_sentiment}")

# Calculating the mean absolute value of the actual sentiment scores
mean_abs_sentiment = np.mean(np.abs(data_df['sentiment']))
print(f"Mean absolute of actual sentiment scores: {mean_abs_sentiment}")

# Comparing RMSE with the standard deviation
print(f"Le RMSE représente { (rmse / std_dev_sentiment) * 100:.2f}% de l'écart-type des scores réels.")
# This indicates how significant the errors of your sentiment prediction model are compared to the variability of the real data
# the errors of your sentiment prediction model are quite significant compared to the variability of the real data

# Comparing MSE with the standard deviation
print(f"The MSE represents {(mse / std_dev_sentiment**2) * 100:.2f}% de la variance des scores réels.")
# a small part of the variability in sentiment scores is due to your model's prediction errors --> the model is quite precise

# Comparing MAE with the mean absolute
print(f"The MAE represents {(mae / mean_abs_sentiment) * 100:.2f}% de la moyenne absolue des scores réels.")
# the prediction errors, on average, are quite significant compared to the average deviation of the actual scores

"""**Graphic visualization**"""

# Graphique de dispersion
plt.scatter(y_test, pred_sentiment)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual vs Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # ideal lign
plt.show()

# Creating a DataFrame to associate predictions with conversations
test_df['predicted_sentiment'] = pred_sentiment.flatten() # Add predicted sentiment scores to the test DataFrame

# Grouping by conversation_id
conversations = test_df.groupby('conversation_id') # Group the DataFrame by conversation_id
evolution_sentiments = {} # Initialize a dictionary to store the sentiment evolution for each conversation

for conversation_id, group in conversations:
    sentiment_debut = group['predicted_sentiment'].iloc[0] # Get the first (starting) sentiment prediction of the conversation
    sentiment_fin = group['predicted_sentiment'].iloc[-1] # Get the last (ending) sentiment prediction of the conversation
    evolution_sentiments[conversation_id] = sentiment_fin - sentiment_debut # Calculate the change in sentiment from start to end and store it in the dictionary

test_df.to_csv('test_df.csv', index="False")

# Initialize variables to accumulate variations
total_variation = {}
total_variation_patient = []
total_variation_control = []
num_conversations_patient = 0
num_conversations_control = 0

for conversation_id, group in conversations:
    # Calculate the sentiment variation for the conversation
    variation = group['predicted_sentiment'].iloc[-1] - group['predicted_sentiment'].iloc[0]
    total_variation[conversation_id] = variation

    # Count the number of conversations for each group
    if group['group'].iloc[0] == 'patient':
        num_conversations_patient += 1
        total_variation_patient.append(variation)
    elif group['group'].iloc[0] == 'control':
        num_conversations_control += 1
        total_variation_control.append(variation)

# Calculate averages
average_variation_patient = np.mean(total_variation_patient)
average_variation_control = np.mean(total_variation_control)

# Print average variations and conversation counts
print(f"Average variation for patients: {average_variation_patient}")
print(f"Average variation for controls: {average_variation_control}")

print("Total Variations by Conversation:", total_variation)
print("Number of Conversations - Patient:", num_conversations_patient)
print("Number of Conversations - Control:", num_conversations_control)

# Histogram of sentiment variations
plt.figure(figsize=(12, 6))

# Histogram for patients
plt.subplot(1, 2, 1)
plt.hist(total_variation_patient, bins=20, color='blue', alpha=0.7)
plt.title("Distribution of Variations - Patients")
plt.xlabel("Sentiment Variation")
plt.ylabel("Number of Conversations")

# Histogram for control group
plt.subplot(1, 2, 2)
plt.hist(total_variation_control, bins=20, color='green', alpha=0.7)
plt.title("Distribution of Variations - Controls")
plt.xlabel("Sentiment Variation")
plt.ylabel("Number of Conversations")

plt.tight_layout()
plt.show()

# Bar Chart
# Comparison of the average variations
plt.figure(figsize=(6, 4))
groups = ['Patient', 'Control']
averages = [average_variation_patient, average_variation_control]

# Creating a bar chart to compare the averages
plt.bar(groups, averages, color=['blue', 'green'])
plt.xlabel('Group')
plt.ylabel('Average Sentiment Variation')
plt.title('Comparison of Average Sentiment Variations')
plt.show()

# check the normal distribution
# Shapiro-Wilk to check if both groups follow a normal distribution
# patient group
shapiro_test_patient = stats.shapiro(total_variation_patient)
print(f"Shapiro Test for patient group: Statistic={shapiro_test_patient[0]}, p-value={shapiro_test_patient[1]}")

# control group
shapiro_test_control = stats.shapiro(total_variation_control)
print(f"Shapiro Test for control group : Statistic={shapiro_test_control[0]}, p-value={shapiro_test_control[1]}")

#check the homogeneity of variances
from scipy.stats import levene

# Levene's test to check for homogeneity of variances
stat, p_value = levene(total_variation_patient, total_variation_control)

print(f"Levene's Test : Statistic={stat}, p-value={p_value}")

# If p-value > 0.05, it suggests homogeneity of variances

from scipy.stats import ttest_ind

# Performing a t-test for independent samples
t_stat, p_value = ttest_ind(total_variation_patient, total_variation_control)

print(f"T-test for independent samples :  t-Statistic={t_stat}, p-value={p_value}")

#p-value > 0.05 --> the difference between the means of the two groups is not statistically significant
# If p-value > 0.05, it suggests that the difference between the means of the two groups is not statistically significant

from scipy.stats import mannwhitneyu

# Mann-Whitney U Test
u_statistic, p_value = mannwhitneyu(total_variation_patient, total_variation_control, alternative='two-sided')

print(f"TMann-Whitney U Test: U Statistic={u_statistic}, p-value={p_value}")
#-->  p-value >0.05 not significative statistic difference between the two groups
# If p-value > 0.05, it indicates no statistically significant difference between the two groups