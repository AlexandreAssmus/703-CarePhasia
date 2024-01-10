import pandas as pd
import nltk
nltk.download('punkt')

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load CSV files into DataFrames
# Replace 'control_group.csv' and 'aphasic_group.csv' with the actual file paths
df_control = pd.read_csv('../control_data_clean.csv')
df_aphasic = pd.read_csv('../patient_data_clean.csv')



# Example text with unknown words
text = "This is an example sentence with unknown words: abcd xyz."

# Create a TextBlob object
blob = TextBlob(text)

# Customized sentiment calculation
unknown_word_score = -0.5  # Assign a negative sentiment score to unknown words

# Calculate polarity by considering unknown words as negative
polarity = sum([unknown_word_score if word not in blob.word_counts else blob.word_counts[word] * TextBlob(word).sentiment.polarity for word in blob.words]) / len(blob.words)
print(f"Custom Polarity: {polarity}")

# Function for sentiment analysis using TextBlob
def analyze_sentiment(sentence):
    # Convert the input to a string
    sentence_str = str(sentence)

    # Create a TextBlob object for sentiment analysis
    analysis = TextBlob(sentence_str)

    # Assigning sentiment polarity to one of three categories: 'positive', 'negative', 'neutral'
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to the 'text' column and store results in a new column 'sentiment'
df_control['sentiment'] = df_control['text'].apply(analyze_sentiment)
df_aphasic['sentiment'] = df_aphasic['text'].apply(analyze_sentiment)

# Display the DataFrames with the new 'sentiment' columns
print("Control Group:")
print(df_control)

print("\nAphasic Group:")
print(df_aphasic)


# Function for sentiment analysis using TextBlob

def analyze_sentiment(sentence):
    sentence_str = str(sentence)
    analysis = TextBlob(sentence_str)
    print(f"Sentence: {sentence}, Polarity: {analysis.sentiment.polarity}")
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to a subset of the 'text' column and print the results
print ("control group :")
sample_texts = df_control['text'].head(10)
df_control['sentiment'] = sample_texts.apply(analyze_sentiment)
print(df_control[['text', 'sentiment']])

print ("----------------")

print ("patient group :")
sample_texts = df_aphasic['text'].head(10)
df_aphasic['sentiment'] = sample_texts.apply(analyze_sentiment)
print(df_aphasic[['text', 'sentiment']])

# Function for sentiment analysis using TextBlob
def analyze_sentiment(sentence):
    sentence_str = str(sentence)
    analysis = TextBlob(sentence_str)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the 'text' column and store results in a new column 'sentiment_score'
print ("control group :")
df_control['sentiment_score'] = df_control['text'].apply(analyze_sentiment)
print(df_control[['text', 'sentiment_score']])
# Afficher les éléments 5 et 6 du DataFrame
print(df_control[['text', 'sentiment_score']].iloc[4:6])

print ("---------")

print ("patient group :")
df_aphasic['sentiment_score'] = df_aphasic['text'].apply(analyze_sentiment)
print(df_aphasic[['text', 'sentiment_score']])
# Afficher les éléments 5 et 6 du DataFrame
print(df_aphasic[['text', 'sentiment_score']].iloc[4:6])

# Function for sentiment analysis using TextBlob
def analyze_sentiment(sentence):
    analysis = TextBlob(str(sentence))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to the 'text' column and store results in a new column 'sentiment'
df_control['sentiment'] = df_control['text'].apply(analyze_sentiment)
df_aphasic['sentiment'] = df_aphasic['text'].apply(analyze_sentiment)

# Statistiques descriptives pour chaque groupe
control_sentiment_stats = df_control['sentiment'].value_counts()
aphasic_sentiment_stats = df_aphasic['sentiment'].value_counts()

# Afficher les statistiques descriptives
print("Statistiques du groupe contrôle :")
print(control_sentiment_stats)


print("\nStatistiques du groupe aphasique :")
print(aphasic_sentiment_stats)

# Visualisation des statistiques
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

control_sentiment_stats.plot(kind='bar', ax=axes[0], title='Groupe Contrôle')
aphasic_sentiment_stats.plot(kind='bar', ax=axes[1], title='Groupe Aphasique')

plt.show()

# Comparaison des moyennes de polarité

"""
Calculez la moyenne de la polarité du sentiment pour chaque groupe et comparez-les.
Cela peut donner une indication générale de la tendance sentimentale.
"""

# Calcul des moyennes de polarité pour chaque groupe
control_mean_polarity = df_control['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
aphasic_mean_polarity = df_aphasic['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()

# Afficher les moyennes de polarité
print("Moyenne de polarité pour le groupe contrôle :", control_mean_polarity)
print("Moyenne de polarité pour le groupe aphasique :", aphasic_mean_polarity)


# Comparaison statistique
"""
Utilisez des tests statistiques pour déterminer si les différences observées entre les groupes sont statistiquement significatives.
La manière de procéder dépend de la distribution de vos données.
Des tests tels que le test t de Student peuvent être utilisés si les conditions sont remplies.
"""
from scipy.stats import ttest_ind

# Effectuer un test t de Student pour comparer les moyennes de polarité
t_stat, p_value = ttest_ind(df_control['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity),
                             df_aphasic['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity))

# Afficher le résultat du test
print("Résultat du test t de Student :")
print("Statistique de test :", t_stat)
print("Valeur de p :", p_value)

# Interpréter le résultat
if p_value < 0.05:
    print("Différence statistiquement significative entre les groupes.")
else:
    print("Pas de différence statistiquement significative entre les groupes.")
