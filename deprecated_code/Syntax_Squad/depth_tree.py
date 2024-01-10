import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def calculate_max_tree_depth(clauses_list):
    '''
    This function calculates the depth of the tree for each clause.
    Parameterers:
    clauses_lists (list of strings): list of strings purely alphabetical
    Returns:
    list: list of integers that should later be add to the dataframe
    '''
    def calculate_tree_depth(clause):
        # Process the clause with spaCy
        doc = nlp(clause)

        # Function to calculate the depth of a node
        def node_depth(node):
            # If the node has no children, its depth is 0
            if not list(node.children):
                return 0
            # Otherwise, the depth is 1 + the maximum depth of its children
            else:
                return 1 + max(node_depth(child) for child in node.children)

        # Find the root of the clause and calculate its depth
        root = next((token for token in doc if token.dep_ == "ROOT"), None)
        if root is None:
            return 0

        return node_depth(root)

    # Handle cases where there are no clauses or unexpected input
    if not clauses_list or not isinstance(clauses_list, list):
        return 0

    # Calculate the depth for each clause in the list
    depths = [calculate_tree_depth(clause) for clause in clauses_list]

    # Return the maximum depth among all clauses
    return max(depths, default=0)
