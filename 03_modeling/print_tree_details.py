# 03_modeling/print_tree_details.py

import pickle
import numpy as np

FEATURE_COLS = [
    'is_verb_3sg',
    'is_pron_subject',
    'is_preposition',
    'format_listen',
    'days_in_course',
    'client_android',
    'client_ios',
    'session_practice',
    'session_test',
    'format_reverse_translate',
    'token_order'
]

def print_tree_rules(tree, feature_names, node_id=0, depth=0, lines=[]):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    # Calculate probabilities
    value = tree.value[node_id][0]
    total_samples = tree.n_node_samples[node_id]
    prob_error = value[1] / sum(value) if sum(value) > 0 else 0.0
    
    indent = "|   " * depth
    if left_child == right_child: # leaf node
        lines.append(f"{indent}Leaf: Error Rate = {prob_error*100:.2f}% (Samples: {int(total_samples)})")
    else:
        feat = feature_names[tree.feature[node_id]]
        thresh = tree.threshold[node_id]
        
        # Check if feature is boolean (0.5 threshold)
        if thresh == 0.5:
            lines.append(f"{indent}If {feat} is False:")
            print_tree_rules(tree, feature_names, left_child, depth + 1, lines)
            lines.append(f"{indent}If {feat} is True:")
            print_tree_rules(tree, feature_names, right_child, depth + 1, lines)
        else:
            lines.append(f"{indent}If {feat} <= {thresh:.2f}:")
            print_tree_rules(tree, feature_names, left_child, depth + 1, lines)
            lines.append(f"{indent}If {feat} > {thresh:.2f}:")
            print_tree_rules(tree, feature_names, right_child, depth + 1, lines)
    return lines

def main():
    with open("03_modeling/decision_tree_model.pkl", "rb") as f:
        dt_model = pickle.load(f)
        
    tree = dt_model.tree_
    lines = print_tree_rules(tree, FEATURE_COLS)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
