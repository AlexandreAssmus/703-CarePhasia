from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from scipy.sparse import hstack
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn import tree

import shap
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
#### Data preprocessing ####

data = pd.read_csv(r'New_clean_code\Data\thresholds_per_file.csv')


#Division according to data and labels

X = data[['average_tree_depth', 'average_lexical_density','word_stutter_count','syllable_stutter_ratio']]
y = data['diagnosis']

#Division according to training and evaluation data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# #### Model creation - Random Forest ####
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# #### Model evaluation ####
# y_pred = rf_model.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
# print(classification_report(y_test, y_pred))

# # # Generating the confusion matrix
# # cm = confusion_matrix(y_test, y_pred)
categories = ["Patient", "Control"]
# # # Plotting the confusion matrix
# plt.figure(figsize=(10,7))
# sns.heatmap(cm/np.sum(cm), annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
# plt.xlabel('Predicted')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show() 

# # #feature importance - Built-in Gini Importance
# importance = rf_model.feature_importances_
# sorted_indices = np.argsort(importance)[::-1]

# # # Converting to data frames
# feature_names = ['average_tree_depth', 'average_lexical_density','word_stutter_count','syllable_stutter_ratio']
# df = pd.DataFrame({'Feature': np.array(feature_names)[sorted_indices], 'Importance': importance[sorted_indices]})

# # Plotting the feature importance
# plt.figure(figsize=(10, 6))
# plt.title('Feature Importance')
# plt.bar(range(X.shape[1]), df['Importance'], align="center")
# plt.xticks(range(X.shape[1]), df['Feature'], rotation=45)
# plt.xlim([-1, X.shape[1]])
# plt.show()

# ##feature importance - Mean Decrease Accuracy

# permutation_imp = permutation_importance(rf_model, X_test, y_test)
# sorted_indices_p = permutation_imp.importances_mean.argsort()

# df_perm = pd.DataFrame({'Feature': np.array(feature_names)[sorted_indices_p], 'Importance': importance[sorted_indices_p]})
# plt.bar(range(X.shape[1]), df['Importance'], align="center")
# plt.xticks(range(X.shape[1]), df['Feature'], rotation=45)
# plt.xlim([-1, X.shape[1]])
# plt.xlabel("Permutation Importance")
# plt.show()

# #feature importance - with SHAP values

# explainer = shap.TreeExplainer(rf_model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=categories)

# # #### Save model ####
# joblib.dump(rf_model, r'New_clean_code\Model\random_forest_model.pkl')



#CLassification using Decision Trees

#### Model creation - Decision Tree ####
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

#### Model evaluation  ####
y_pred_dt = dt_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_dt)}')
print(classification_report(y_test, y_pred_dt))

# # Generating the confusion matrix

cm_dt = confusion_matrix(y_test, y_pred_dt)
categories = ["Patient", "Control"]
# Plotting the confusion matrix

plt.figure(figsize=(10,7))
sns.heatmap(cm_dt/np.sum(cm_dt), annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Decision Tree')
plt.show() 


# #Graph visualization - default + graphviz
# plt.figure(figsize=(35, 28))
# plot_tree(dt_model, feature_names=feature_names, class_names=categories, filled=True, rounded=True)
# plt.show()

# # Export as dot file
# dot_data = export_graphviz(dt_model, out_file=None, 
#                            feature_names=feature_names,  
#                            class_names=categories,
#                            filled=True, rounded=True,  
#                            special_characters=True)

# # Use graphviz to visualize the tree
# graph = graphviz.Source(dot_data) 
# graph.render("decision_tree", format="png")  # Saves the tree as a PNG file
# graph.view()  # Opens the tree in a viewer


# ## CLassification using TF-IDF vectorization

# data_texts = pd.read_csv(r'New_clean_code\Data\CSV_clean\combined_control_patient_data.csv')
# texts = data_texts['text']
# labels = data_texts['group']


# # # TF-IDF Transformation
# vectorizer = TfidfVectorizer()
# X_tf = vectorizer.fit_transform(texts)
# # print(tf_vectors)
# # X_tf = hstack([tf_vectors, X])

# # print("Going into training")
# # # Splitting the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_tf, labels, test_size=0.3, random_state=42)

# # # Train Random Forest Model
# model_tf = RandomForestClassifier()
# model_tf.fit(X_train, y_train)

# # print("training finished")

# # #### Model evaluation ####
# y_pred_tf = model_tf.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred_tf)}')
# print(classification_report(y_test, y_pred_tf))

# # # Generating the confusion matrix
# cm_tf = confusion_matrix(y_test, y_pred_tf)
# categories = ["Patient", "Control"]


# # Plotting the confusion matrix
# plt.figure(figsize=(10,7))
# sns.heatmap(cm_tf/np.sum(cm_tf), annot=True, fmt='g', cmap='Blues', xticklabels=categories, yticklabels=categories)
# plt.xlabel('Predicted')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show() 


# # # Get Feature Importance - Extracting the important words
# feature_importance = model_tf.feature_importances_

# # # Creating a DataFrame for better visualization
# feature_names = vectorizer.get_feature_names_out()
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# # # Sorting by importance
# importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# # print(importance_df)
# top_n = 10
# top_features = importance_df.head(top_n)
# plt.figure(figsize=(10, 6))
# plt.barh(top_features['Feature'], top_features['Importance'])
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.title('Top {} Feature Importances in TF-IDF'.format(top_n))
# plt.gca().invert_yaxis()  # To display the highest importance on top
# plt.show()

# joblib.dump(model_tf, r'New_clean_code\Model\random_forest_model_with_TF-IDF.pkl')

