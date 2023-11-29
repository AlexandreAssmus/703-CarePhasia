# Libraries importation
import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm') 

#There was a problem with the installation of SpaCy. I followed this solution:
#https://github.com/explosion/spaCy/issues/12659

def find_root_of_sentence(doc):
    return next((token for token in doc if token.dep_ == "ROOT"), None)

def find_other_verbs(doc, root_token):
    return [token for token in doc if token.pos_ == "VERB" and token.head == root_token]

def get_clause_token_span(verb, doc):
    if verb is None:
        return (0, 0)
    clause_tokens = list(verb.subtree)
    return (min(token.i for token in clause_tokens), max(token.i for token in clause_tokens))

def process_sentence(sentence):
    # Convert non-string input to string
    if not isinstance(sentence, str):
        sentence = str(sentence)

    doc = nlp(sentence)
    root_token = find_root_of_sentence(doc)

    if root_token is None:
        return [sentence]

    all_verbs = [root_token] + find_other_verbs(doc, root_token)
    token_spans = [get_clause_token_span(verb, doc) for verb in all_verbs]

    # Gather the unique clause text, avoiding repetitions
    unique_clauses = set()
    for start, end in token_spans:
        clause_text = doc[start:end + 1].text.strip()
        unique_clauses.add(clause_text)

    # Return the unique clauses as a list
    return list(unique_clauses) if len(unique_clauses) > 1 else [sentence]

def split_into_clauses(df, column_name):
    df['Split_Clauses'] = df[column_name].apply(process_sentence)
    return df

#Usage function clause separation:

df_clause_length = pd.read_csv(r'C:\Users\belen\Desktop\Université de Lorraine\703\Aphasia\Data\Cleaned_files\control_data_clean.csv')
processed_df = split_into_clauses(df_clause_length, 'text')

def count_words_in_clause(clause):
    doc = nlp(clause)
    # Count words in the clause, excluding punctuation and whitespace
    word_count = sum(1 for token in doc if not token.is_punct and not token.is_space)
    return word_count

def count_words_in_clause_list_and_return_list(clauses_list):
    # Use a list comprehension to apply the word count function to each clause
    return [count_words_in_clause(clause) for clause in clauses_list]
