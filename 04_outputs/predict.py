# 04_outputs/predict.py

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Define expected features in exact order as during training
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

# RobustScaler parameters computed from the training split
SCALER_MEDIAN = 4.704
SCALER_IQR = 8.171

def load_models():
    """Loads Logistic Regression and Decision Tree models from script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lr_path = os.path.join(script_dir, "models", "logistic_regression_model.pkl")
    dt_path = os.path.join(script_dir, "models", "decision_tree_model.pkl")
    
    if not os.path.exists(lr_path) or not os.path.exists(dt_path):
        print(f"Error: Model files not found. Please ensure they exist at:\n - {lr_path}\n - {dt_path}")
        sys.exit(1)
        
    with open(lr_path, "rb") as f:
        lr_model = pickle.load(f)
    with open(dt_path, "rb") as f:
        dt_model = pickle.load(f)
        
    return lr_model, dt_model

def preprocess_exercise(exercise):
    """Processes a raw tokenized exercise dictionary into features matching model schema."""
    # Metadata parameters
    exercise_format = exercise.get("exercise_format", "")
    session_type = exercise.get("session_type", "")
    client = exercise.get("client", "")
    days_in_course_raw = float(exercise.get("days_in_course", 0.0))
    
    # Scale days in course
    days_scaled = (days_in_course_raw - SCALER_MEDIAN) / SCALER_IQR
    
    # Platform context mappings
    client_android = (client == "android")
    client_ios = (client == "ios")
    
    # Session context mappings
    session_practice = (session_type == "practice")
    session_test = (session_type == "test")
    
    # Format context mappings
    format_listen = (exercise_format == "listen")
    format_reverse_translate = (exercise_format == "reverse_translate")
    
    processed_tokens = []
    
    for tok in exercise.get("tokens", []):
        token_text = tok.get("token", "")
        pos = tok.get("part_of_speech", "")
        morph = tok.get("morphology", "")
        dep = tok.get("dependency_label", "")
        order = int(tok.get("token_order", 1))
        
        # A. 3rd person singular present verbs
        is_verb = pos in ['VERB', 'AUX']
        is_3sg = "Person=3" in morph and "Number=Sing" in morph
        is_verb_3sg = (is_verb and is_3sg)
        
        # B. Subject pronouns (pro-drop structural clash)
        is_pron_subject = (pos == "PRON" and dep == "nsubj")
        
        # C. Prepositions (adposition mapping lexical clash)
        is_preposition = (pos == "ADP")
        
        # Construct feature dictionary in exact training order
        feature_dict = {
            'is_verb_3sg': bool(is_verb_3sg),
            'is_pron_subject': bool(is_pron_subject),
            'is_preposition': bool(is_preposition),
            'format_listen': bool(format_listen),
            'days_in_course': float(days_scaled),
            'client_android': bool(client_android),
            'client_ios': bool(client_ios),
            'session_practice': bool(session_practice),
            'session_test': bool(session_test),
            'format_reverse_translate': bool(format_reverse_translate),
            'token_order': int(order),
            'listen_x_prep': bool(format_listen and is_preposition),
            'listen_x_verb3sg': bool(format_listen and is_verb_3sg),
            'listen_x_pron': bool(format_listen and is_pron_subject),
            'translate_x_prep': bool(format_reverse_translate and is_preposition),
            'translate_x_verb3sg': bool(format_reverse_translate and is_verb_3sg),
            'translate_x_pron': bool(format_reverse_translate and is_pron_subject)
        }
        
        processed_tokens.append({
            'text': token_text,
            'features': feature_dict,
            'pos': pos,
            'dep': dep,
            'morph': morph
        })
        
    return processed_tokens

def color_code_risk(prob):
    """Returns ANSI escape codes for coloring risk levels."""
    # Class baseline is 12.6% error.
    # We define:
    # - Low risk: prob < 10%
    # - Medium risk: 10% <= prob < 20%
    # - High risk: prob >= 20%
    if prob < 0.10:
        return "\033[92m", "LOW"  # Green
    elif prob < 0.20:
        return "\033[93m", "MED"  # Yellow
    else:
        return "\033[91m", "HIGH" # Red

def get_pedagogical_triggers(features):
    """Extracts pedagogical explanations for why a token is flagged as high-risk."""
    triggers = []
    
    # Format clashes
    if features['format_listen']:
        triggers.append("Auditory Spelling Recall Hazard (+325.3% error odds)")
    if features['format_reverse_translate']:
        triggers.append("Free-production Writing Constraint (+226.9% error odds)")
        
    # Grammatical clashes
    if features['is_preposition']:
        triggers.append("Preposition Mapping lexical clash (+28.8% error odds)")
    if features['is_verb_3sg']:
        triggers.append("3rd Person Singular subject-verb agreement redundancy (-32.9% error odds)")
    if features['is_pron_subject']:
        triggers.append("Explicit subject pronoun structure (-51.0% error odds)")
        
    # Sentence context
    if features['token_order'] > 3:
        triggers.append(f"Position fatigue/syntactic complexity at index {features['token_order']} (+10.9% odds/word)")
        
    return triggers

def run_predictions(lr_model, dt_model, processed_tokens):
    """Performs inference and prints a detailed analysis report."""
    results = []
    
    for tok in processed_tokens:
        # Build features DataFrame in exact order to suppress warnings and match feature names
        features_df = pd.DataFrame([tok['features']], columns=FEATURE_COLS)
        
        # Predict probability
        lr_prob = lr_model.predict_proba(features_df)[0, 1]
        dt_prob = dt_model.predict_proba(features_df)[0, 1]
        
        results.append({
            'text': tok['text'],
            'lr_prob': lr_prob,
            'dt_prob': dt_prob,
            'features': tok['features'],
            'pos': tok['pos']
        })
        
    return results

def print_report(exercise, results):
    """Prints a beautifully formatted report to the console."""
    print("=" * 80)
    print(f" SLA KNOWLEDGE TRACING INFERENCE REPORT")
    print(f" Exercise ID:  {exercise.get('exercise_id', 'unknown')}")
    print(f" Format:       {exercise.get('exercise_format', 'unknown').upper()}")
    print(f" Session Type: {exercise.get('session_type', 'unknown').upper()}")
    print(f" Platform:     {exercise.get('client', 'unknown').upper()}")
    print(f" Course Time:  {exercise.get('days_in_course', 0.0)} days")
    print(f" Full Sentence: \"{exercise.get('sentence', '')}\"")
    print("=" * 80)
    
    print(f"{'Token':<12} | {'POS':<6} | {'LR Prob':<9} | {'DT Prob':<9} | {'Risk':<6} | {'Pedagogical Interference Triggers / Explanations'}")
    print("-" * 80)
    
    reset_code = "\033[0m"
    
    for res in results:
        prob = res['lr_prob'] # We use LR as the primary calibrated probability
        color_code, risk_level = color_code_risk(prob)
        
        triggers = get_pedagogical_triggers(res['features'])
        trigger_str = " | ".join(triggers) if triggers else "Baseline acquisition risk"
        
        # Format printing
        lr_prob_str = f"{res['lr_prob']*100:.1f}%"
        dt_prob_str = f"{res['dt_prob']*100:.1f}%"
        
        print(f"{res['text']:<12} | {res['pos']:<6} | {lr_prob_str:<9} | {dt_prob_str:<9} | {color_code}{risk_level:<6}{reset_code} | {trigger_str}")
        
    print("=" * 80)
    print("\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json_path = os.path.join(script_dir, "sample_exercises.json")
    
    json_path = default_json_path
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        
    if not os.path.exists(json_path):
        print(f"Error: Target file not found at: {json_path}")
        sys.exit(1)
        
    with open(json_path, "r") as f:
        try:
            exercises = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            sys.exit(1)
            
    # If a single exercise is passed instead of a list, wrap it
    if isinstance(exercises, dict):
        exercises = [exercises]
        
    lr_model, dt_model = load_models()
    
    print(f"Loaded Logistic Regression and Decision Tree models successfully.\n")
    print(f"Running inference on {len(exercises)} exercises from '{os.path.basename(json_path)}'...\n")
    
    for ex in exercises:
        processed = preprocess_exercise(ex)
        results = run_predictions(lr_model, dt_model, processed)
        print_report(ex, results)

if __name__ == "__main__":
    main()
