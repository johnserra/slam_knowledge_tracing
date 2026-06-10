# 04_outputs/generate_curriculum_rules.py

import os
import json
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
    'token_order',
    'listen_x_prep',
    'listen_x_verb3sg',
    'listen_x_pron',
    'translate_x_prep',
    'translate_x_verb3sg',
    'translate_x_pron'
]

def traverse_tree(tree, feature_names, node_id=0, current_path=None, rules=None, min_samples_gate=1000):
    if current_path is None:
        current_path = []
    if rules is None:
        rules = []
        
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    # Calculate error probability in this node
    value = tree.value[node_id][0]
    total_samples = int(tree.n_node_samples[node_id])
    prob_error = float(value[1] / sum(value)) if sum(value) > 0 else 0.0
    
    # Check if leaf node
    if left_child == right_child:
        # We only generate rules for high-risk cohorts (exceeding baseline 12.6%) with sufficient sample size
        if prob_error >= 0.15 and total_samples >= min_samples_gate:
            # Determine CTL Errorless Teaching Scaffold Action
            if prob_error >= 0.30:
                action = "FORCE_TAP_WORD_BANK_WITH_EXPLICIT_HINT"
                rationale = "Very high error probability (>= 30%). Provide maximal scaffolding to prevent error consolidation."
            elif prob_error >= 0.20:
                action = "INJECT_POPUP_GRAMMAR_ALERT"
                rationale = "High error probability (20% - 30%). Alert learner of the specific grammatical rule prior to execution."
            else:
                action = "HIGHLIGHT_WORD_IN_PREVIEW"
                rationale = "Elevated error probability (15% - 20%). Emphasize the target token structure in lesson preview."
                
            rules.append({
                "rule_id": f"SLA_RULE_{len(rules) + 1:02d}",
                "conditions": list(current_path),
                "samples_in_cohort": total_samples,
                "predicted_error_rate": f"{prob_error*100:.2f}%",
                "scaffold_action": action,
                "rationale": rationale
            })
    else:
        feat = feature_names[tree.feature[node_id]]
        thresh = float(tree.threshold[node_id])
        
        # Branch Left (feature <= threshold)
        left_cond = {
            "feature": feat,
            "operator": "<=",
            "value": thresh,
            "description": f"{feat} is False" if thresh == 0.5 else f"{feat} <= {thresh:.2f}"
        }
        current_path.append(left_cond)
        traverse_tree(tree, feature_names, left_child, current_path, rules, min_samples_gate)
        current_path.pop()
        
        # Branch Right (feature > threshold)
        right_cond = {
            "feature": feat,
            "operator": ">",
            "value": thresh,
            "description": f"{feat} is True" if thresh == 0.5 else f"{feat} > {thresh:.2f}"
        }
        current_path.append(right_cond)
        traverse_tree(tree, feature_names, right_child, current_path, rules, min_samples_gate)
        current_path.pop()
        
    return rules

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "decision_tree_model.pkl")
    output_path = os.path.join(script_dir, "curriculum_rules.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Decision Tree model not found at {model_path}")
        return
        
    with open(model_path, "rb") as f:
        dt_model = pickle.load(f)
        
    print("Extracting high-risk pedagogical rules from Decision Tree...")
    rules = traverse_tree(dt_model.tree_, FEATURE_COLS, min_samples_gate=1000)
    
    # Save rules to JSON
    with open(output_path, "w") as f:
        json.dump(rules, f, indent=2)
        
    print(f"Successfully generated {len(rules)} rules and saved to:\n -> {output_path}\n")
    
    # Print rules overview
    print("=" * 80)
    print(" GENERATED ACQUISITION RULES FOR CAREERTALKLAB LESSON ENGINE")
    print("=" * 80)
    for r in rules:
        print(f"Rule ID: {r['rule_id']} (Error Rate: {r['predicted_error_rate']}, Cohort: {r['samples_in_cohort']} samples)")
        print(f"Action:  {r['scaffold_action']}")
        print(f"Path Conditions:")
        for cond in r['conditions']:
            print(f"  - {cond['description']}")
        print(f"Rationale: {r['rationale']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
