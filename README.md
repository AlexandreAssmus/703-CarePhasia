# Overview
CarePhasia is a research prototype that explores how linguistic patterns can help flag the risk of aphasia.
It extracts several language‑level metrics from raw transcripts and feeds them to a machine‑learning model that estimates whether the speech resembles that of a patient or a control subject.

# Features
- Sentence segmentation & cleaning – converts a plain text transcript into a clean CSV file.
- Linguistic metrics
- Syntactic tree depth 
- Lexical density (ratio of content words to total words)
- Stuttering detection for repeated words and syllables
- Sentiment analysis of each sentence (LSTM‑based prototype)
- Modeling – Random Forest classifier (with optional Decision Tree and TF‑IDF experiments) to distinguish patient vs. control speech.
- Interpretability – thresholds and SHAP/permutation importance plots describe how each feature influences predictions.
- User pipeline – interactive CLI that guides users through uploading a text file and receiving an interpretive report.

# Repository structure
New_clean_code/

├── Data/                     # Datasets, preprocessing utilities, and threshold files

├── Linguistic_level_functions/

│   ├── calculate_lexical_density.py

│   ├── calculate_tree_depth.py

│   ├── stutter_detection_function.py

│   └── user_pipeline.py      # Main entry point for end users

├── Model/

│   ├── random_forest_model.py

│   ├── interpretation_metrics.py

│   └── random_forest_model.pkl

└── User_pipeline/            # Intermediate files generated during user runs

Result Graphs/                # Confusion matrices, feature importance plots, word clouds, …

deprecated_code/              # Archived experiments

requirements.txt


# Installation 

```bash
git clone <this-repo-url>
```
```bash
cd CarePhasia
```
```bash
python -m venv venv
```
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```
# Quick start

* Prepare a .txt transcript where sentences are separated by standard punctuation.

* Run the user pipeline:

```bash 
python New_clean_code/Linguistic_level_functions/user_pipeline.py
```

Follow the prompts to select your text file. The script will:

- segment the transcript into sentences,

- compute the linguistic metrics,

- average them and compare against reference thresholds,

- run the Random Forest classifier,

- output a JSON interpretation and store intermediate CSVs in *New_clean_code/User_pipeline/*.

# Training your own model

To retrain or experiment with new features:

```bash
python New_clean_code/Model/random_forest_model.py
```

This script uses *New_clean_code/Data/thresholds_per_file.csv* as training data and saves a new *random_forest_model.pkl*.

The script *interpretation_metrics.py* loads this model along with user metrics to generate explanatory text.

# Visualizations 

Performance graphs, feature‑importance plots, and word clouds illustrating patient vs. control speech appear under *Result Graphs/*. These assets are produced by the modeling scripts and may serve as inspiration for further analysis.
