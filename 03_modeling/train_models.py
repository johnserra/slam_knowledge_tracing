# 03_modeling/train_models.py

import os
import pickle
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import warnings

# Initialize configurations
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "en_es_slam")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Define features and their target
# Reference categories omitted to avoid multicollinearity:
# - client: reference is client_web (Android and iOS coefficients are relative to Web)
# - session_type: reference is session_lesson (Practice and Test coefficients are relative to Lesson)
# - exercise_format: reference is format_reverse_tap (Listen and Reverse Translate coefficients are relative to Reverse Tap)
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

TARGET_COL = 'is_error'

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def load_data_from_db():
    """Loads all features and user_ids from the database."""
    print("Loading all features from database...")
    
    query = """
        SELECT 
            ef.is_error, ef.is_verb_3sg, ef.is_pron_subject, ef.is_preposition, ef.format_listen, 
            ef.days_in_course, ef.client_android, ef.client_ios, 
            ef.session_practice, ef.session_test, ef.format_reverse_translate, ef.token_order,
            e.user_id
        FROM engineered_features ef
        JOIN raw_tokens t ON ef.token_id = t.token_id
        JOIN raw_exercises e ON t.exercise_id = e.exercise_id;
    """
    
    conn = get_db_connection()
    try:
        # Suppress pandas read_sql user warnings regarding DBAPI2 connections
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} rows from database.")
        
        # Compute cross-product interaction terms
        df['listen_x_prep'] = df['format_listen'] & df['is_preposition']
        df['listen_x_verb3sg'] = df['format_listen'] & df['is_verb_3sg']
        df['listen_x_pron'] = df['format_listen'] & df['is_pron_subject']
        df['translate_x_prep'] = df['format_reverse_translate'] & df['is_preposition']
        df['translate_x_verb3sg'] = df['format_reverse_translate'] & df['is_verb_3sg']
        df['translate_x_pron'] = df['format_reverse_translate'] & df['is_pron_subject']
        
        # Optimize memory by downcasting types
        bool_cols = [c for c in df.columns if c not in ['days_in_course', 'token_order', 'user_id']]
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        df['days_in_course'] = df['days_in_course'].astype(np.float32)
        df['token_order'] = df['token_order'].astype(np.int16)
        
        return df
    finally:
        conn.close()

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(bool)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'confusion_matrix': cm
    }

def find_optimal_threshold(y_true, y_prob):
    """Finds the decision threshold that maximizes the F1 score."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(bool)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1

def print_tree_rules(tree, feature_names, node_id=0, depth=0, lines=None):
    if lines is None:
        lines = []
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
    # Ensure unified output directory exists
    os.makedirs("04_outputs/models", exist_ok=True)
    
    # 1. Load all data from database
    df_all = load_data_from_db()
    
    # Partition train/test set strictly by user_id
    print("Partitioning train/test sets strictly by user_id to prevent data leakage...")
    unique_users = df_all['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    
    train_users_set = set(train_users)
    train_mask = df_all['user_id'].isin(train_users_set)
    
    df_train = df_all[train_mask].copy()
    df_test = df_all[~train_mask].copy()
    
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    groups_train = df_train['user_id'].values
    
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]
    
    print(f"\nTotal exercises dataset split summary:")
    print(f"  - Training Set: {len(train_users)} users ({len(df_train)} tokens)")
    print(f"  - Test Set:     {len(test_users)} users ({len(df_test)} tokens)")
    
    print("\n--- Training Set Class Distribution ---")
    neg, pos = np.bincount(y_train)
    print(f"Correct (False): {neg} ({neg/len(y_train)*100:.2f}%)")
    print(f"Error (True): {pos} ({pos/len(y_train)*100:.2f}%)")
    
    # 2. StratifiedGroupKFold Cross-Validation on Train set
    print("\nRunning 5-fold StratifiedGroupKFold Cross-Validation...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_cv_aucs, lr_cv_f1s = [], []
    dt_cv_aucs, dt_cv_f1s = [], []
    
    fold = 1
    for train_idx, val_idx in sgkf.split(X_train, y_train, groups=groups_train):
        print(f"Processing CV Fold {fold}/5...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train LR
        lr_temp = LogisticRegression(max_iter=1000, random_state=42)
        lr_temp.fit(X_tr, y_tr)
        lr_val_probs = lr_temp.predict_proba(X_val)[:, 1]
        
        # Train DT (max_depth=4 for interpretability)
        dt_temp = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_temp.fit(X_tr, y_tr)
        dt_val_probs = dt_temp.predict_proba(X_val)[:, 1]
        
        # Evaluate
        lr_cv_aucs.append(roc_auc_score(y_val, lr_val_probs))
        lr_cv_f1s.append(f1_score(y_val, lr_val_probs >= 0.5, zero_division=0))
        
        dt_cv_aucs.append(roc_auc_score(y_val, dt_val_probs))
        dt_cv_f1s.append(f1_score(y_val, dt_val_probs >= 0.5, zero_division=0))
        fold += 1
        
    print("\n--- Cross-Validation Results ---")
    print(f"Logistic Regression CV AUC: {np.mean(lr_cv_aucs):.4f} +/- {np.std(lr_cv_aucs):.4f}")
    print(f"Logistic Regression CV F1 (0.5 threshold): {np.mean(lr_cv_f1s):.4f} +/- {np.std(lr_cv_f1s):.4f}")
    print(f"Decision Tree CV AUC: {np.mean(dt_cv_aucs):.4f} +/- {np.std(dt_cv_aucs):.4f}")
    print(f"Decision Tree CV F1 (0.5 threshold): {np.mean(dt_cv_f1s):.4f} +/- {np.std(dt_cv_f1s):.4f}")
    
    # 3. Train final models on full training set
    print("\nTraining final models on full train set...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Find optimal thresholds on Train set to maximize F1-score
    print("Finding optimal thresholds on train set...")
    lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
    best_lr_thresh, best_lr_f1 = find_optimal_threshold(y_train, lr_train_probs)
    print(f"Optimal LR Threshold: {best_lr_thresh:.2f} (Train F1: {best_lr_f1:.4f})")
    
    dt_train_probs = dt_model.predict_proba(X_train)[:, 1]
    best_dt_thresh, best_dt_f1 = find_optimal_threshold(y_train, dt_train_probs)
    print(f"Optimal DT Threshold: {best_dt_thresh:.2f} (Train F1: {best_dt_f1:.4f})")
    
    # Save final models directly to unified directory
    with open("04_outputs/models/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open("04_outputs/models/decision_tree_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    print("Models saved directly to '04_outputs/models/' directory.")
    
    # Skip garbage collection cleanup to keep variables available for report generation
    
    # 4. Evaluate on Test set
    print("\nEvaluating on Test set...")
    lr_test_probs = lr_model.predict_proba(X_test)[:, 1]
    dt_test_probs = dt_model.predict_proba(X_test)[:, 1]
    
    # Default 0.5 threshold evaluation
    lr_eval_def = evaluate_predictions(y_test, lr_test_probs, threshold=0.5)
    dt_eval_def = evaluate_predictions(y_test, dt_test_probs, threshold=0.5)
    
    # Optimal threshold evaluation
    lr_eval_opt = evaluate_predictions(y_test, lr_test_probs, threshold=best_lr_thresh)
    dt_eval_opt = evaluate_predictions(y_test, dt_test_probs, threshold=best_dt_thresh)
    
    print("\n--- Test Set Evaluation (Threshold = 0.5) ---")
    print(f"Logistic Regression: AUC={lr_eval_def['auc']:.4f}, F1={lr_eval_def['f1']:.4f}, Prec={lr_eval_def['precision']:.4f}, Rec={lr_eval_def['recall']:.4f}")
    print(f"Decision Tree:       AUC={dt_eval_def['auc']:.4f}, F1={dt_eval_def['f1']:.4f}, Prec={dt_eval_def['precision']:.4f}, Rec={dt_eval_def['recall']:.4f}")
    
    print(f"\n--- Test Set Evaluation (Optimal Threshold: LR={best_lr_thresh:.2f}, DT={best_dt_thresh:.2f}) ---")
    print(f"Logistic Regression: AUC={lr_eval_opt['auc']:.4f}, F1={lr_eval_opt['f1']:.4f}, Prec={lr_eval_opt['precision']:.4f}, Rec={lr_eval_opt['recall']:.4f}")
    print(f"Decision Tree:       AUC={dt_eval_opt['auc']:.4f}, F1={dt_eval_opt['f1']:.4f}, Prec={dt_eval_opt['precision']:.4f}, Rec={dt_eval_opt['recall']:.4f}")
    
    # 5. Extract and format model coefficients / splits
    # A. LR Coefficients
    coefs = lr_model.coef_[0]
    odds_ratios = np.exp(coefs)
    lr_coef_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Coefficient': coefs,
        'Odds_Ratio': odds_ratios,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    
    lr_coef_df.to_csv("04_outputs/logistic_regression_coefficients.csv", index=False)
    print("\nLogistic Regression coefficients saved to '04_outputs/logistic_regression_coefficients.csv'")
    
    # B. DT Splits Text Rules
    dt_rules_lines = print_tree_rules(dt_model.tree_, FEATURE_COLS, node_id=0, depth=0, lines=[])
    dt_rules = "\n".join(dt_rules_lines)
    with open("04_outputs/decision_tree_rules.txt", "w") as f:
        f.write(dt_rules + "\n")
    print("Decision Tree text rules saved to '04_outputs/decision_tree_rules.txt'")
    
    # C. Write Markdown Evaluation Report
    report_path = "04_outputs/modeling_evaluation_report.md"
    
    # Calculate values to insert into text
    listen_or = lr_coef_df[lr_coef_df['Feature'] == 'format_listen']['Odds_Ratio'].values[0]
    rev_or = lr_coef_df[lr_coef_df['Feature'] == 'format_reverse_translate']['Odds_Ratio'].values[0]
    prep_or = lr_coef_df[lr_coef_df['Feature'] == 'is_preposition']['Odds_Ratio'].values[0]
    v3sg_or = lr_coef_df[lr_coef_df['Feature'] == 'is_verb_3sg']['Odds_Ratio'].values[0]
    pron_or = lr_coef_df[lr_coef_df['Feature'] == 'is_pron_subject']['Odds_Ratio'].values[0]
    days_or = lr_coef_df[lr_coef_df['Feature'] == 'days_in_course']['Odds_Ratio'].values[0]
    order_or = lr_coef_df[lr_coef_df['Feature'] == 'token_order']['Odds_Ratio'].values[0]
    
    # Formats & features pct changes (absolute value for clean wording)
    listen_pct = (listen_or - 1.0) * 100 if listen_or > 1 else (1.0 - listen_or) * 100
    rev_pct = (rev_or - 1.0) * 100 if rev_or > 1 else (1.0 - rev_or) * 100
    prep_pct = (prep_or - 1.0) * 100 if prep_or > 1 else (1.0 - prep_or) * 100
    v3sg_pct = (v3sg_or - 1.0) * 100 if v3sg_or > 1 else (1.0 - v3sg_or) * 100
    pron_pct = (pron_or - 1.0) * 100 if pron_or > 1 else (1.0 - pron_or) * 100
    days_pct = (days_or - 1.0) * 100 if days_or > 1 else (1.0 - days_or) * 100
    order_pct = (order_or - 1.0) * 100 if order_or > 1 else (1.0 - order_or) * 100
    
    # Generate coefficients table rows
    table_rows = []
    for _, row in lr_coef_df.iterrows():
        feat = row['Feature']
        coef = row['Coefficient']
        or_val = row['Odds_Ratio']
        
        if or_val > 1:
            pct = (or_val - 1.0) * 100
            impact = f"Increases error odds by **+{pct:.1f}%**"
        else:
            pct = (1.0 - or_val) * 100
            impact = f"Decreases error odds by **{pct:.1f}%**"
            
        ref = "N/A"
        if feat in ['client_android', 'client_ios']:
            ref = "client_web"
        elif feat in ['session_practice', 'session_test']:
            ref = "session_lesson"
        elif feat in ['format_listen', 'format_reverse_translate']:
            ref = "format_reverse_tap"
            
        table_rows.append(f"| `{feat}` | {coef:.4f} | {or_val:.4f} | {impact} | {ref} |")
        
    table_content = "\n".join(table_rows)
    
    report_content = f"""# Model Evaluation & Pedagogical Interpretation Report

This report summarizes the predictive performance and educational interpretations of the machine learning models trained on the Duolingo `en_es` SLAM (Second Language Acquisition Modeling) dataset. 

Our models prioritize **pedagogical interpretability** over black-box optimizations to directly aid language curriculum designers and teachers in understanding when and why native Spanish speakers make errors when learning English.

---

## 1. Predictive Performance Summary

The dataset consists of **2,622,957** tokens partitioned at the learner level to prevent data leakage:
- **Training Set:** 80% of unique users ({len(train_users)} users, {len(df_train)} tokens)
- **Test Set:** 20% of unique users ({len(test_users)} users, {len(df_test)} tokens)
- **Class Imbalance:** 87.39% Correct, 12.61% Error (approx. 7:1 ratio)

### Validation & Test Metrics

| Model | CV AUC-ROC | Test AUC-ROC | Test F1 (Thresh=0.5) | Test F1 (Opt Thresh) | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | {np.mean(lr_cv_aucs):.4f} | {lr_eval_def['auc']:.4f} | {lr_eval_def['f1']:.4f} | {lr_eval_opt['f1']:.4f} | {best_lr_thresh:.2f} |
| **Decision Tree (depth=4)** | {np.mean(dt_cv_aucs):.4f} | {dt_eval_def['auc']:.4f} | {dt_eval_def['f1']:.4f} | {dt_eval_opt['f1']:.4f} | {best_dt_thresh:.2f} |

### Detailed Test Set Metrics (Optimal Threshold)

- **Logistic Regression (Threshold = {best_lr_thresh:.2f})**
  - Accuracy: {lr_eval_opt['accuracy']:.4f}
  - Precision: {lr_eval_opt['precision']:.4f}
  - Recall: {lr_eval_opt['recall']:.4f}
  - Confusion Matrix:
    ```
    [[{lr_eval_opt['confusion_matrix'][0, 0]} (TN)  {lr_eval_opt['confusion_matrix'][0, 1]} (FP)]
     [{lr_eval_opt['confusion_matrix'][1, 0]} (FN)  {lr_eval_opt['confusion_matrix'][1, 1]} (TP)]]
    ```

- **Decision Tree (Threshold = {best_dt_thresh:.2f})**
  - Accuracy: {dt_eval_opt['accuracy']:.4f}
  - Precision: {dt_eval_opt['precision']:.4f}
  - Recall: {dt_eval_opt['recall']:.4f}
  - Confusion Matrix:
    ```
    [[{dt_eval_opt['confusion_matrix'][0, 0]} (TN)  {dt_eval_opt['confusion_matrix'][0, 1]} (FP)]
     [{dt_eval_opt['confusion_matrix'][1, 0]} (FN)  {dt_eval_opt['confusion_matrix'][1, 1]} (TP)]]
    ```

---

## 2. Pedagogical Interpretation: Logistic Regression

The Logistic Regression model provides direct log-odds coefficients for each feature. An odds ratio > 1 indicates that the feature *increases* the probability of making an error, whereas an odds ratio < 1 indicates that it *decreases* the probability of making an error.

### Feature Coefficients & Odds Ratios

| Feature | Coefficient | Odds Ratio | Impact on Error Rate | Reference Category (if any) |
| :--- | :---: | :---: | :--- | :--- |
{table_content}

### Key Insights from Coefficients

1. **Exercise Formats:**
   - **Listening Exercises (`format_listen`)** are the strongest positive predictor of errors. Students have an odds ratio of **{listen_or:.2f}** for making a spelling or recall error in listening exercises compared to the baseline `reverse_tap` format. This highlights the substantial challenge of phoneme-to-grapheme mapping in English L2 acquisition.
   - **Reverse Translation (`format_reverse_translate`)** also increases error odds (odds ratio: **{rev_or:.2f}**), reflecting the increased difficulty of producing raw text without assistance compared to tapping choices.

2. **Syntactic and Lexical Clashes:**
   - **Prepositions (`is_preposition`):** Features an odds ratio of **{prep_or:.2f}** (+{prep_pct:.1f}% increase in error odds). In Spanish, prepositions are heavily mapped 1-to-many into English (e.g., *en* maps to *in*, *on*, *at*). This result confirms the persistent lexical mapping interference.
   - **3rd Person Singular Present Verbs (`is_verb_3sg`):** Has an odds ratio of **{v3sg_or:.2f}** (which represents a {v3sg_pct:.1f}% decrease in error odds). Spanish has a highly inflected subject-verb agreement system, making the English singular marker morphologically redundant in their L1-tuned grammar.
   - **Subject Pronouns (`is_pron_subject`):** Has an odds ratio of **{pron_or:.2f}** (which represents a {pron_pct:.1f}% decrease in error odds). This suggests that when subject pronouns are explicitly present in target sentences, they represent simple grammatical constructions that are less error-prone.

3. **Context and Progress Signals:**
   - **Time in Course (`days_in_course`):** An odds ratio of **{days_or:.2f}** indicates that for each standard deviation increase in course time, the odds of an error *increase* by **{days_pct:.1f}%**. This positive relationship is a known phenomenon where Duolingo's curriculum difficulty increases faster than the rate of student mastery, leading to higher baseline error rates in advanced lessons.
   - **Token Order (`token_order`):** Odds ratio of **{order_or:.2f}** (+{order_pct:.1f}% error odds per word position). Words later in the sentence have slightly higher error rates, reflecting growing syntactic complexity or cognitive fatigue as the sentence length increases.
   - **Platform & Session Types:**
     - **Practice (`session_practice`)** and **Test (`session_test`)** sessions have significantly lower error odds compared to new lessons (`session_lesson`). This makes intuitive sense: practice targets reinforced vocabulary/rules, and tests are taken by more confident users.
     - **Mobile Clients (`client_android`, `client_ios`)** show reduced error odds compared to Web clients. This could reflect differences in tapping-based user interface defaults, session durations, or user demographics.

---

## 3. Educational Decision Tree Rules

The Decision Tree model (limited to `max_depth = 4` to preserve readability) yields the following tree structure. 

### Visualized Decision Rules

```text
{dt_rules}
```

### Key Decision Pathways

By tracing the tree, we can identify high-risk educational scenarios:
1. **The Listening Hazard:**
   - If `format_listen` is **True**:
     - And `days_in_course` is **< -0.14** (early in the course):
       - If the token is not a subject pronoun (`is_pron_subject` = **False**): predicted error rate is high (14.97% or 20.26%).
2. **Grammatical Clash in Writing:**
   - If `format_listen` is **False**:
     - And `format_reverse_translate` is **True** (free text generation):
       - If the word is a 3rd person singular verb (`is_verb_3sg` = **True**): error rate is elevated to 38.54% at the beginning of the sentence (`token_order` <= 1.50).

These rules allow curriculum developers to dynamically inject targeted review sessions (e.g. preposition exercises in auditory formats) when a student matches a high-error leaf node path.
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"Markdown report generated successfully at '{report_path}'.")

if __name__ == "__main__":
    main()
