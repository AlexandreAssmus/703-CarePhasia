import spacy
import pandas as pd 

nlp = spacy.load('en_core_web_sm')

def analyze_word_order(clause):
    # Process the clause with spaCy
    doc = nlp(clause)

    # Initialize variables to None; they will store the first subject, verb, and object we find
    subject = verb = obj = None

    # Iterate over the tokens
    for token in doc:
        # If we find a subject, verb, and object, we break out of the loop as we've got the basic word order
        if subject and verb and obj:
            break
        # Check for subject (nsubj)
        if not subject and token.dep_ == 'nsubj':
            subject = token
        # Check for verb (ROOT typically represents the main verb in a clause)
        if not verb and token.dep_ == 'ROOT':
            verb = token
        # Check for object (dobj for direct object)
        if not obj and token.dep_ == 'dobj':
            obj = token

    # Now we determine the word order based on the positions of the subject, verb, and object
    word_order = ''
    elements = [(subject, 'S'), (verb, 'V'), (obj, 'O')]
    for element, label in sorted(elements, key=lambda x: (x[0] is not None, x[0].i if x[0] else -1)):
        if element:
            word_order += label

    return word_order

def get_word_order_for_clauses(clauses_list):
    # Return a list of word order patterns for the list of clauses
    return [analyze_word_order(clause) for clause in clauses_list]
