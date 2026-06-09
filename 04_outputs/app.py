# 04_outputs/app.py

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="SLA Knowledge Tracing Sandbox",
    description="Interactive model visualization for L2 English acquisition analysis",
    version="1.0"
)

# Define request schema
class TokenInput(BaseModel):
    token: str
    part_of_speech: str
    morphology: str
    dependency_label: str
    token_order: int

class AnalysisRequest(BaseModel):
    sentence: str
    exercise_format: str
    session_type: str
    client: str
    days_in_course: float
    custom_tokens: Optional[List[TokenInput]] = None

# Model columns
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

# Scaler constants
SCALER_MEDIAN = 4.704
SCALER_IQR = 8.171

# Global models
LR_MODEL = None
DT_MODEL = None

# Custom Lexicon for parsing raw input sentences when NLP library is absent
LEXICON = {
    # Pronouns
    'he': {'pos': 'PRON', 'morph': 'Person=3|Number=Sing', 'dep': 'nsubj'},
    'she': {'pos': 'PRON', 'morph': 'Person=3|Number=Sing', 'dep': 'nsubj'},
    'they': {'pos': 'PRON', 'morph': 'Person=3|Number=Plur', 'dep': 'nsubj'},
    'it': {'pos': 'PRON', 'morph': 'Person=3|Number=Sing', 'dep': 'nsubj'},
    'i': {'pos': 'PRON', 'morph': 'Person=1|Number=Sing', 'dep': 'nsubj'},
    'you': {'pos': 'PRON', 'morph': 'Person=2', 'dep': 'nsubj'},
    'we': {'pos': 'PRON', 'morph': 'Person=1|Number=Plur', 'dep': 'nsubj'},
    'him': {'pos': 'PRON', 'morph': 'Person=3|Number=Sing|Case=Acc', 'dep': 'dobj'},
    'her': {'pos': 'PRON', 'morph': 'Person=3|Number=Sing|Case=Acc', 'dep': 'dobj'},
    'them': {'pos': 'PRON', 'morph': 'Person=3|Number=Plur|Case=Acc', 'dep': 'dobj'},
    # Verbs 3sg present
    'works': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'plays': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'runs': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'goes': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'eats': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'speaks': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'writes': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'reads': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    'is': {'pos': 'AUX', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'cop'},
    'has': {'pos': 'VERB', 'morph': 'Person=3|Number=Sing|Tense=Pres', 'dep': 'ROOT'},
    # Verbs non-3sg
    'work': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'play': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'run': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'go': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'eat': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'speak': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'write': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'read': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    'are': {'pos': 'AUX', 'morph': 'Tense=Pres', 'dep': 'cop'},
    'have': {'pos': 'VERB', 'morph': 'Tense=Pres', 'dep': 'ROOT'},
    # Prepositions
    'in': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'on': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'at': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'to': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'from': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'with': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'for': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'of': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'about': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'by': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    'under': {'pos': 'ADP', 'morph': 'null', 'dep': 'prep'},
    # Determiners
    'the': {'pos': 'DET', 'morph': 'null', 'dep': 'det'},
    'a': {'pos': 'DET', 'morph': 'null', 'dep': 'det'},
    'an': {'pos': 'DET', 'morph': 'null', 'dep': 'det'},
    # Nouns
    'bank': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'office': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'house': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'school': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'soccer': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'dobj'},
    'park': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'book': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'dobj'},
    'car': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'dobj'},
    'friend': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'teacher': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'student': {'pos': 'NOUN', 'morph': 'Number=Sing', 'dep': 'pobj'},
    'english': {'pos': 'PROPN', 'morph': 'Number=Sing', 'dep': 'dobj'},
}

@app.on_event("startup")
def load_serialized_models():
    global LR_MODEL, DT_MODEL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lr_path = os.path.join(script_dir, "models", "logistic_regression_model.pkl")
    dt_path = os.path.join(script_dir, "models", "decision_tree_model.pkl")
    
    if os.path.exists(lr_path) and os.path.exists(dt_path):
        with open(lr_path, "rb") as f:
            LR_MODEL = pickle.load(f)
        with open(dt_path, "rb") as f:
            DT_MODEL = pickle.load(f)
        print("Models loaded successfully.")
    else:
        print("Warning: Models not found on startup.")

def clean_sentence_to_tokens(sentence: str):
    """Simple lexicon and regex splitter to tokenize and POS-tag sentences."""
    # Clean punctuation
    cleaned = sentence.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    words = cleaned.split()
    
    tokens = []
    for idx, w in enumerate(words):
        w_lower = w.lower()
        order = idx + 1
        
        # Match lexicon
        if w_lower in LEXICON:
            lex = LEXICON[w_lower]
            pos = lex['pos']
            morph = lex['morph']
            dep = lex['dep']
        else:
            # Fallback heuristics
            dep = "dobj" if idx > 1 else "ROOT"
            morph = "null"
            if w_lower.endswith("s") and not w_lower in ['is', 'has', 'was', 'works', 'plays', 'runs', 'goes', 'eats', 'speaks', 'writes', 'reads']:
                pos = "NOUN"
                morph = "Number=Plur"
            else:
                pos = "NOUN"
                morph = "Number=Sing"
                
        tokens.append({
            'token': w,
            'part_of_speech': pos,
            'morphology': morph,
            'dependency_label': dep,
            'token_order': order
        })
    return tokens

@app.post("/api/predict")
def predict_error_probabilities(req: AnalysisRequest):
    if LR_MODEL is None or DT_MODEL is None:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
        
    # Get tokens
    if req.custom_tokens and len(req.custom_tokens) > 0:
        raw_tokens = [tok.dict() for tok in req.custom_tokens]
    else:
        raw_tokens = clean_sentence_to_tokens(req.sentence)
        
    # Scaled parameters
    days_scaled = (req.days_in_course - SCALER_MEDIAN) / SCALER_IQR
    
    # Metadata context indicators
    client_android = (req.client == "android")
    client_ios = (req.client == "ios")
    
    session_practice = (req.session_type == "practice")
    session_test = (req.session_type == "test")
    
    format_listen = (req.exercise_format == "listen")
    format_reverse_translate = (req.exercise_format == "reverse_translate")
    
    predictions = []
    
    for tok in raw_tokens:
        token_text = tok['token']
        pos = tok['part_of_speech']
        morph = tok['morphology']
        dep = tok['dependency_label']
        order = tok['token_order']
        
        # 1. Morphosyntactic clashes
        is_verb = pos in ['VERB', 'AUX']
        is_3sg = "Person=3" in morph and "Number=Sing" in morph
        is_verb_3sg = (is_verb and is_3sg)
        
        is_pron_subject = (pos == "PRON" and dep == "nsubj")
        is_preposition = (pos == "ADP")
        
        # 2. Build feature dictionary
        features = {
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
        
        # 3. Model inference
        features_df = pd.DataFrame([features], columns=FEATURE_COLS)
        lr_prob = LR_MODEL.predict_proba(features_df)[0, 1]
        dt_prob = DT_MODEL.predict_proba(features_df)[0, 1]
        
        # 4. Generate pedagogical explanations
        explanations = []
        if format_listen:
            explanations.append("Auditory Spelling Recall Hazard (+325.3% error odds)")
        if format_reverse_translate:
            explanations.append("Free-production Writing Constraint (+226.9% error odds)")
        if is_preposition:
            explanations.append("Preposition Mapping interference lexical clash (+28.8% error odds)")
        if is_verb_3sg:
            explanations.append("3rd Person Singular marker redundancy structural clash (-32.9% error odds)")
        if is_pron_subject:
            explanations.append("Explicit subject pronoun construction (-51.0% error odds)")
        if order > 3:
            explanations.append(f"Position syntactic load / fatigue at index {order} (+10.9% odds/word)")
            
        predictions.append({
            'token': token_text,
            'pos': pos,
            'order': order,
            'lr_prob': float(lr_prob),
            'dt_prob': float(dt_prob),
            'features': features,
            'explanations': explanations
        })
        
    return {
        'sentence': req.sentence,
        'format': req.exercise_format,
        'session_type': req.session_type,
        'client': req.client,
        'days_in_course': req.days_in_course,
        'predictions': predictions
    }

# Mount static files (will be created in next steps)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SLA Sandbox static files not yet initialized</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
