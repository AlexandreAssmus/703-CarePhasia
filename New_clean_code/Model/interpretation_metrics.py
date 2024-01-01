import pandas as pd
import joblib
from random_forest_model import X_test, y_test
import json


#### Thresholds for interpretation ####

thresholds_df = pd.read_csv(r'New_clean_code\Data\general_thresholds.csv')

thresholds = {
    'control': {
        'average_tree_depth': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'average_tree_depth'].values[0],
        'average_lexical_density': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'average_lexical_density'].values[0]
    },
    'patient': {
        'average_tree_depth': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'average_tree_depth'].values[0],
        'average_lexical_density': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'average_lexical_density'].values[0]
    }
}

import json

### Interpretation of results, more granular for the user ####

def interpret_metrics(row, thresholds, model):
    probabilities = model.predict_proba([[row['average_tree_depth'], row['average_lexical_density']]])[0]
    patient_probability = probabilities[1]  # Assuming '1' represents 'patient'
    explanations = []
    
    for metric, value in zip(['average_tree_depth', 'average_lexical_density'], [row['average_tree_depth'], row['average_lexical_density']]):
        if value > thresholds['patient'][metric]:
            explanations.append(f"has a high value ({value:.2f}) in {metric.replace('_', ' ')}")
        elif value < thresholds['control'][metric]:
            explanations.append(f"has a low value ({value:.2f}) in {metric.replace('_', ' ')}")
    
    explanation_str = ' and '.join(explanations)
    return f"There is a {patient_probability*100:.2f}% probability that this speech corresponds to aphasic speech because it {explanation_str}."


# Load the model and use it to interpret new entries
loaded_rf_model = joblib.load(r'New_clean_code\Model\random_forest_model.pkl')


# Add columns for predictions and probabilities to the test DataFrame to facilitate interpretation
X_test_with_predictions = X_test.copy()
X_test_with_predictions['predicted'] = loaded_rf_model.predict(X_test)
X_test_with_predictions['probability_patient'] = loaded_rf_model.predict_proba(X_test)[:, 1]  # Probability of 'patient'

# List to store interpretations
interpretations = []

# Loop to generate interpretations
for index, row in X_test_with_predictions.iterrows():
    explanation = interpret_metrics(row, thresholds, loaded_rf_model)
    interpretations.append({
        'entry_index': index,
        'interpretation': explanation
    })

# Save interpretations to a JSON file
with open('interpretations.json', 'w') as f:
    json.dump(interpretations, f, indent=4)

# Print interpretations to the console
#for interpretation in interpretations:
    #print(f"Interpretation for entry {interpretation['entry_index']}: {interpretation['interpretation']}")

