
import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def count_verb_phrases(clause):
    # Process the clause with spaCy
    doc = nlp(clause)

    # Initialize a counter for verb phrases
    verb_phrase_count = 0

    # Iterate over the tokens
    for token in doc:
        # Check for main verbs; in spaCy, these are often labeled as 'ROOT' or have auxiliaries ('aux', 'auxpass')
        if token.pos_ == 'VERB' and (token.dep_ == 'ROOT' or any(child.dep_ in ['aux', 'auxpass'] for child in token.children)):
            verb_phrase_count += 1

    return verb_phrase_count

def get_verb_phrase_counts_for_clauses(clauses_list):
    # Return a list of verb phrase counts for the list of clauses
    return [count_verb_phrases(clause) for clause in clauses_list]
