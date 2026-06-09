# SLA-Theoretic Knowledge Tracing for English L2 Learners (Spanish L1)

An interpretable, token-level machine learning model designed to predict grammatical and syntactic errors made by native Spanish speakers learning English. This repository pivots from standard black-box sequence models to prioritize **pedagogical actionability**, mapping learner mistakes directly to Second Language Acquisition (SLA) theories to guide curriculum scaffolding.

Built using the Duolingo SLAM dataset (2.62M tokens, 824K exercises).

---

## Project Overview

Traditional Knowledge Tracing (e.g., BKT, DKT) models student performance at the sequence or exercise level. This project focuses on **token-level error tracking**, allowing us to identify the exact words and grammatical structures that trigger learner errors. 

Rather than chasing marginal gains in predictive power via deep learning, we implement interpretable models (**Logistic Regression** and **Decision Trees**) to extract clear, readable rules. These rules can be integrated directly into adaptive learning systems using an **Errorless Teaching** framework (Prompt-Echo-Distractor-Transfer).

---

## Second Language Acquisition (SLA) Features

We engineer syntactic and morphological features to represent three core cross-linguistic "clashes" between Spanish (L1) and English (L2):

1. **The Dropped Pronoun Clue (`is_pron_subject`) — Pro-drop Structural Clash**
   * *SLA Theory:* Spanish is a pro-drop language where subject pronouns are morphologically optional because verb inflections communicate the subject (e.g., *"Soy un niño"*). English requires explicit subject pronouns (*"I am a boy"*). Spanish speakers frequently drop pronouns or confuse cases due to L1 transfer.
2. **The Verb Ending Clue (`is_verb_3sg`) — Morphological Redundancy**
   * *SLA Theory:* Spanish features highly inflected verbs across all person/number agreements. English only inflects the present tense in the third-person singular (adding the `-s` suffix, e.g., *"she runs"*). Spanish speakers frequently omit this morphologically redundant suffix.
3. **The Preposition Clue (`is_preposition`) — Lexical Mapping Interference**
   * *SLA Theory:* Spanish utilizes a single broad preposition (*"en"*) which maps one-to-many into three different English prepositions (*"in"*, *"on"*, *"at"*). This mapping clash makes English prepositions a persistent source of error (overall error rate of 18.4% on prepositions).
4. **Format & Cognitive Load (`format_listen`) — Auditory Recall vs. Visual Recognition**
   * *SLA Theory:* Listening exercises require active spelling recall and phoneme-to-grapheme decoding, which carry a higher cognitive load (17.52% error rate) compared to multiple-choice tapping exercises (4.59% error rate).

---

## Model Performance

The dataset was split chronologically/sequentially (80% training, 20% test) to prevent look-ahead leakage. 

| Model | CV AUC-ROC | Test AUC-ROC | Test F1 (Optimal Threshold) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (with Interaction Terms)** | **0.6617** | **0.6611** | **0.3155** |
| **Decision Tree (depth=4)** | **0.6276** | **0.6262** | **0.2882** |

### Key Odds Ratios (Logistic Regression)

An Odds Ratio (OR) > 1 indicates that a feature increases error probability; an OR < 1 indicates a decrease.

* **`format_listen` (OR: 4.31):** Listening exercises increase the odds of a token error by **+331.4%** compared to standard tapping formats.
* **`format_reverse_translate` (OR: 2.11):** Free translation exercises (writing) increase error odds by **+111.3%** compared to tapping, validating the Output Hypothesis.
* **`is_preposition` (OR: 1.54):** Prepositional tokens increase error odds by **+54.4%** due to Spanish-to-English lexical interference.
* **`is_pron_subject` (OR: 0.34):** Explicitly required subject pronouns reduce error odds by **-65.7%** when they are grammatically simple.
* **`translate_x_pron` (OR: 1.62):** Subject pronoun error odds rise significantly during free writing (reverse translation), capturing active pro-drop transfer errors.

---

## Curriculum Automation & Decision Rules

By traversing the Decision Tree (depth=4), we programmatically extract leaves where the predicted error rate exceeds 15% and map them to instructional scaffolding actions:

* **Predicted Error Rate $\ge 30\%$:** Trigger `FORCE_TAP_WORD_BANK_WITH_EXPLICIT_HINT` (Prompt-Echo tier) to prevent error consolidation.
* **Predicted Error Rate $20\% - 30\%$:** Trigger `INJECT_POPUP_GRAMMAR_ALERT` (Distractor tier) showing the contrast between SVO English and pro-drop Spanish before the user submits.
* **Predicted Error Rate $15\% - 20\%$:** Trigger `HIGHLIGHT_WORD_IN_PREVIEW` (Transfer tier) to emphasize the syntactic structure during lesson introduction.

Rules are exported automatically to [`curriculum_rules.json`](04_outputs/curriculum_rules.json).

---

## Interactive Sandbox

This project features an interactive FastAPI sandbox dashboard to visualize error rates, inspect features, and test inputs in real time:

* **Dynamic Heatmaps:** Inspect which words in a sentence carry the highest predicted error probabilities.
* **Scaffolding Triggers:** View which automated curriculum rules are triggered for each token.
* **Playground:** Input custom sentences, formats, and progress metrics to see real-time model predictions.

---

## Repository Structure

* `01_data_ingestion/`: Handles PostgreSQL database schema creation and atomic streaming ingestion of the raw `.train` data.
* `02_feature_engineering/`: Contains SQL and Python modules that engineer pro-drop, third-person verb agreement, and preposition features.
* `03_modeling/`: Performs model training, cross-validation, hyperparameter selection, and model persistence (`.pkl` exports).
* `04_outputs/`: Holds deliverables, including coefficients, JSON curriculum rules, prediction CLI (`predict.py`), and the FastAPI web sandbox (`app.py`).

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/johnserra/slam_knowledge_tracing.git
cd slam_knowledge_tracing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Command Line Predictions
You can run predictions on custom sentences directly from the terminal:
```bash
python 04_outputs/predict.py --sentence "He runs in the house" --format "reverse_translate" --days 50
```

### 3. Run the Web Sandbox
To launch the FastAPI server and interactive web interface:
```bash
python 04_outputs/app.py
```
Open your browser and navigate to `http://127.0.0.1:8000`.
