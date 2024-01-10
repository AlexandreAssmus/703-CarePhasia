from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

#### Data preprocessing ####

data = pd.read_csv(r'New_clean_code\Data\thresholds_per_file.csv')


#Division according to data and labels

X = data[['average_tree_depth', 'average_lexical_density','word_stutter_count','syllable_stutter_ratio']]
y = data['diagnosis']

#Division according to training and evaluation data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#### Model creation ####
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#### Model evaluation ####
y_pred = rf_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
categories = ["Patient", "Control"]
# Plotting the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show() 

#feature importance
importance = rf_model.feature_importances_
indices = np.argsort(importance)[::-1]

# Converting to data frames
feature_names = ['average_tree_depth', 'average_lexical_density','word_stutter_count','syllable_stutter_ratio']
df = pd.DataFrame({'Feature': np.array(feature_names)[indices], 'Importance': importance[indices]})

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), df['Importance'], color="r", align="center")
plt.xticks(range(X.shape[1]), df['Feature'], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()

#### Save model ####
joblib.dump(rf_model, r'New_clean_code\Model\random_forest_model.pkl')


### CLassification using TF-IDF vectorization

print("Started")
data_texts = pd.read_csv(r'New_clean_code\Data\CSV_clean\combined_control_patient_data.csv')
texts = data_texts['text']
labels = data_texts['group']


# TF-IDF Transformation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print("Going into training")
# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Random Forest Model
model_2 = RandomForestClassifier()
model_2.fit(X_train, y_train)

print("training finished")

#### Model evaluation ####
y_pred = model_2.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Generating the confusion matrix
cm_2 = confusion_matrix(y_test, y_pred)
categories = ["Patient", "Control"]
# Plotting the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm_2/np.sum(cm_2), annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show() 
# Get Feature Importance
feature_importance = model_2.feature_importances_

# Creating a DataFrame for better visualization
feature_names = vectorizer.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sorting by importance
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# print(importance_df)
top_n = 10
top_features = importance_df.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top {} Feature Importances in TF-IDF'.format(top_n))
plt.gca().invert_yaxis()  # To display the highest importance on top
plt.show()



# joblib.dump(model_2, r'New_clean_code\Model\random_forest_model_with_TF-IDF.pkl')