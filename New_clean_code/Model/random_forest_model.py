import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#### Data preprocessing ####

data = pd.read_csv(r'New_clean_code\Data\thresholds_per_file.csv')

#Division according to data and labels

X = data[['average_tree_depth', 'average_lexical_density']]
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

#### Save model ####
joblib.dump(rf_model, 'random_forest_model.pkl')

