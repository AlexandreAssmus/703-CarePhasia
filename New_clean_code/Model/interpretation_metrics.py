import pandas as pd
import joblib
from random_forest_model import X_test, y_test
import json


#### Thresholds for interpretation ####

thresholds_df = pd.read_csv(r'New_clean_code\Data\general_thresholds.csv')
print(thresholds_df)

thresholds = {
    'control': {
        'average_tree_depth': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'average_tree_depth'].values[0],
        'average_lexical_density': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'average_lexical_density'].values[0],
        'word_stutter_count': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'word_stutter_count'].values[0],
        'syllable_stutter_ratio': thresholds_df.loc[thresholds_df['diagnosis'] == 'control', 'syllable_stutter_ratio'].values[0]  # corrected typo here
    },
    'patient': {
        'average_tree_depth': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'average_tree_depth'].values[0],
        'average_lexical_density': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'average_lexical_density'].values[0],
        'word_stutter_count': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'word_stutter_count'].values[0],
        'syllable_stutter_ratio': thresholds_df.loc[thresholds_df['diagnosis'] == 'patient', 'syllable_stutter_ratio'].values[0]  # corrected typo here
    }
}


### Interpretation of results, more granular for the user ####

def interpret_metrics(row, thresholds, model):
    # Get the prediction and the probability for the 'patient' class
    prediction = model.predict([[row['average_tree_depth'], row['average_lexical_density'], row['word_stutter_count'], row['syllable_stutter_ratio']]])[0]
    patient_probability = model.predict_proba([[row['average_tree_depth'], row['average_lexical_density'], row['word_stutter_count'], row['syllable_stutter_ratio']]])[0][1]
    
    # Interpretation strings for each metric
    explanations = []
    
    # Define the expected direction for each metric (True if higher values indicate aphasia, False if lower values indicate aphasia)
    expected_direction = {
        'average_tree_depth': False,  # Lower values indicate aphasia
        'average_lexical_density': False,  # Lower values indicate aphasia
        'word_stutter_count': True,  # Higher values indicate aphasia
        'syllable_stutter_ratio': False  # Lower values indicate aphasia
    }
    
    # Check each metric against the thresholds
    for metric, direction in expected_direction.items():
        value = row[metric]
        threshold = thresholds['control'][metric]  # Using 'control' thresholds as the reference for non-aphasic speech
        comparison = 'higher' if value > threshold else 'lower'
        supports_or_contradicts = 'supports' if ((value > threshold) == direction) else 'contradicts'
        
        explanations.append(f"The {metric} of {value:.2f} {supports_or_contradicts} the aphasia pattern as it is {comparison} than the average non-aphasic threshold of {threshold:.2f}.")

    # Combine the explanations
    explanation_str = " ".join(explanations)
    
    # Final interpretation message
    interpretation = f"There is a {patient_probability*100:.2f}% probability that this speech corresponds to a patient with aphasia. Interpretation: {explanation_str}"
    
    return interpretation

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
with open(r'New_clean_code\Model\interpretations.json', 'w') as f:
    json.dump(interpretations, f, indent=4)

# Print interpretations to the console
#for interpretation in interpretations:
    #print(f"Interpretation for entry {interpretation['entry_index']}: {interpretation['interpretation']}")

