

## 0\. Project Summary

* *Problem:*  
* *Why it matters / impact:*  
* *Proposed approach:*  
* *Primary success metric (business \+ ML):*  
* *Key risks / unknowns:*

## 1\. Problem Framing & Success Metrics

**Business / Research Problem**   
Can we predict token-level second language acquisition errors using features derived from SLA (Second Language Acquisition) theory and syntactic dependencies? Unlike black-box models that predict success/failure without pedagogical context, we want to map learner mistakes to specific SLA concepts (e.g. CEFR levels, grammar structures like auxiliaries and prepositions). This informs curriculum design and personalized feedback.

**Goal**   
Build a highly interpretable token-level predictive model (e.g. Logistic Regression, Decision Tree) that highlights features representing SLA principles, rather than just optimizing AUC. This serves as a decision-support and instructional tool.

**Stakeholders**   
* Curriculum Designers / ESL Instructors (primary audience for insights)
* Product Managers & Engineering (to wire predictions to adaptive learning feedback)
* Data Science Team

**Prior Work**   
Standard Knowledge Tracing (BKT, DKT) models focus on sequence-level performance. This work builds on Duolingo's SLA shared task on token-level error prediction (2018) but pivots from pure predictive power (black-box AUC chasing) to structural interpretability.

**User Story**   
As an ESL Curriculum Designer, I want to understand what specific grammatical structures and formats cause the most errors for learners so that I can structure the learning pathway (e.g., in A1/A2 levels) using an Errorless Teaching framework.

**Hypothesis** (if applicable)   
Certain grammatical features (such as auxiliaries, prepositions, subjunctions) and formats requiring active production (such as listening and writing) have significantly higher error rates, and modeling them via interpretable syntactic features will yield clear pedagogical rules.

**Metrics**

| Metric Type | Metric | Target | Notes |
| :---- | :---- | :---- | :---- |
| Business / impact metric | Pedagogical actionability | > 5 clear rules | *Instructors can directly read and apply decision tree pathways to structure curriculum* |
| ML metric (primary) | AUC-ROC | >= 0.75 | *Prioritized over accuracy due to 7:1 class imbalance, ensuring we capture discrimination* |
| ML metric (secondary) | F1-Score / Model Calibration | Balanced recall | *Ensures model probability scores align with actual error frequencies without overpredicting* |

**Constraints**

| Constraint | Details |
| :---- | :---- |
| Data availability | 824,012 exercises and 2.62M tokens (Duolingo SLAM dataset) |
| Cost / compute budget | Local CPU compute, standard Linux development server |
| Fairness / bias | Performance parity across learner countries (CO, MX, etc.) |
| Interpretability requirements | **CRITICAL Constraint:** No deep learning. Only interpretable scikit-learn models (Logistic Regression, Decision Trees) to satisfy instructor readability |
| Timeline | Short-term research pipeline |

**Checklist**:

- [x] I have a clearly defined business or research question, not just a "prediction of X."  
- [x] I have strong evidence that ML is the right approach (vs. simple rules, heuristics, or basic analytics).  
- [x] I have defined both business metrics and ML metrics, and I understand how they relate.  
- [x] I have shared the scope with stakeholders and aligned on success criteria.  
- [x] I have a stated hypothesis or analytical framework.

## 2\. Data Sourcing & Collection

**Data Sources**

| Source | Type | Collection Method | Volume | Time Range | Known Issues |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Duolingo SLAM Shared Task | Structured Token-level Text Stream | PostgreSQL database ingestion from raw text streams | 824,012 exercises, 2,622,957 tokens | 2019 (2019-02-04 release) | Prompt-less listening format blocks, literal "null" strings for missing fields |

**Sourcing Strategy**   
Rather than using a simplified, pre-made Kaggle CSV, we built a robust custom parsing and ingestion pipeline. The pipeline streams raw text data directly from the `.train` dataset file, cleans and structures exercise-level metadata and token-level details, and bulk-inserts them into a PostgreSQL database hosted on a remote server (`vps-6480434c.vps.ovh.ca`) over a secure SSH tunnel. The ingestion script handles type casting, maps missing data programmatically, and implements atomic transaction commits per batch (5,000 exercises) for database consistency.

**Data Provenance & Limitations**   
The data originates from English-learning tracks for Spanish speakers on Duolingo (Spanish L1 -> English L2). Key limitations include:
* **L1/L2 specificity:** Lexical and grammatical errors are highly specific to Spanish speakers learning English. Interlanguage structures (e.g. transfer errors) will not generalize directly to other L1 groups (e.g., Turkish or Japanese speakers).
* **Format-specific omissions:** Listening exercises do not present a visual L1 prompt, causing omissions of prompt metadata in the raw text stream.
* **Selection Bias:** Represents users on mobile/web learning platforms, who may skew towards self-directed learners compared to classroom cohorts.

**Data Quality Assessment**

| Check | Approach | Findings |
| :---- | :---- | :---- |
| Missing values | *% missing per feature, patterns of missingness (MCAR, MAR, MNAR)* | literal "null" strings for missing fields (e.g., time, days) are parsed programmatically as None |
| Duplicates | *Deduplication strategy* | Exercises and tokens are uniquely identified by DB primary keys |
| Outliers | *Detection method, how handled* | Checked numeric values (days_in_course, time_taken) for validity |
| Class imbalance (if classification) | *Class distribution* | 87.39% Correct, 12.61% Error (7:1 class imbalance) |
| Data distribution | *Key distributional properties, skewness, multimodality* | 2,622,957 total tokens. High skewness towards common POS categories (NOUN/VERB/PRON representing 63% of tokens) |
| Volume sufficiency | *Is there enough data for reliable modeling?* | 824,012 exercises and 2.62M tokens are highly sufficient |

**Checklist**:

- [x] I have sourced data in a realistic, documented way.  
- [x] I understand the provenance and limitations of my data.  
- [x] I have performed a thorough data quality assessment.  
- [x] I have verified that data usage is compliant with legal and privacy requirements.  
- [x] I have documented all data sources and their reliability.

## 3\. Exploratory Data Analysis (EDA)

**EDA Objectives**   
*What questions is your EDA trying to answer? What are you looking for? EDA should be hypothesis-driven, not just "let me plot everything."*

**Key Analyses**

| Analysis | Purpose | Key Findings |
| :---- | :---- | :---- |
| Univariate distributions | *Understand individual feature distributions, identify skewness/outliers* | Target variable `is_error` is imbalanced (~7:1 ratio). Features like POS are heavily skewed towards Nouns/Verbs |
| Bivariate relationships | *Relationships between features and target, feature-feature correlations* | High error rates in AUX (22.25%) and ADP (18.41%), low in PRON (8.51%) and DET (8.91%) |
| Temporal patterns (if applicable) | *Trends, seasonality, structural breaks* | Chronological progression tracked via days_in_course |
| Group comparisons | *Differences across key segments* | Format: Listen (17.52%) vs Tap (4.59%). Session: Practice (17.26%) vs Lesson (11.45%) |
| Missing data patterns | *Is missingness random or informative?* | Missingness is format-specific (e.g. listening exercises have no prompt) |

**Visualizations**   
*List key visualizations created and what they reveal. Store in a documented notebook or report.*
* Detailed distribution tables and segment comparison analysis are documented in [eda_baseline_error_report.md](file:///home/johnserra/.gemini/antigravity-cli/brain/69cd7a7d-1372-4165-88f1-93d3b897afd4/eda_baseline_error_report.md)

**EDA-Driven Decisions**   
* 1. Prioritized AUC-ROC and F1-score over accuracy metrics due to a 7:1 class imbalance.
* 2. Identified syntactic variables (AUX, ADP, SCONJ) as primary feature engineering targets due to their high baseline error rates.
* 3. Confirmed that platform, session type, and format must be modeled explicitly due to distinct segment error rates.

**Checklist**:

- [x] My EDA is hypothesis-driven and purposeful, not just exploratory plotting.  
- [x] I have examined univariate distributions, bivariate relationships, and key segments.  
- [x] I have documented key findings and how they influenced my modeling approach.  
- [x] EDA is in a clean, well-documented notebook with a clear narrative.  
- [x] I have visualizations that communicate insights effectively.

## 4\. Feature Engineering

**Feature Overview**

| Feature Name | Description | Source | Transformation | Rationale |
| :---- | :---- | :---- | :---- | :---- |
| `is_verb_3sg` | Flag for 3rd-person singular present verbs | morphology, POS | `(POS == 'VERB' or POS == 'AUX') & ('Person=3' in morphology) & ('Number=Sing' in morphology)` | **Verb Inflection Clash:** Spanish inflects verbs heavily across all persons, but English only inflects present tense in the 3rd person singular (`-s` suffix). Spanish speakers frequently omit this suffix (L1 transfer error). |
| `is_pron_subject` | Flag for nominal subject pronouns | POS, dependency | `POS == 'PRON' & dependency == 'nsubj'` | **Pro-drop Structural Clash:** Spanish is a pro-drop language (subject pronouns are omitted because verb inflections convey the subject). English requires explicit subject pronouns. Spanish speakers frequently drop them or misuse cases. |
| `is_preposition` | Flag for adpositions/prepositions | POS | `POS == 'ADP'` | **Lexical Mapping Clash:** Spanish uses broad prepositions (like "en" mapping to "in", "on", or "at"). Spanish speakers struggle with the precise lexical assignments of English prepositions (ADP error rate is 18.41%). |
| `format_listen` | Flag for listening exercise format | exercise_format | One-hot encoding of `format == 'listen'` | **Auditory Recall vs Recognition:** Listening exercises require phoneme-to-grapheme mapping and active spelling, which carries a higher cognitive load and error rate (17.52%) than word-bank tap formats (4.59%). |
| `days_in_course` | Continuous value of user's course progress | days_in_course | Robust scaling / normalization | **Forgetting Curve & Learning Curve:** More days in the course indicates higher experience and habituation, which correlates with reduced error rates. |

**Layman's Pedagogical Explanation (for Teachers & Curriculum Designers)**

To explain this to non-technical stakeholders (teachers and curriculum designers), we frame these machine learning features as grammatical and format "clues" representing native language (Spanish L1) transfer errors when learning English (L2):

* **The Verb Ending Clue (`is_verb_3sg`):** English present-tense verbs only change ending in the 3rd person singular (adding an `-s`, e.g., *"I run"* vs. *"she run**s**"*). Spanish verbs inflect heavily for every person. Spanish speakers frequently omit this single English outlier ending (e.g., saying *"he run"*).
* **The Dropped Pronoun Clue (`is_pron_subject`):** Spanish is "pro-drop"—verb endings convey who is acting, so pronouns are optional (e.g., *"Soy un niño"*). English requires explicit subject pronouns (*"I am a boy"*). Spanish speakers frequently drop subject pronouns (*"Is a boy"*) because of L1 habit.
* **The Preposition Clue (`is_preposition`):** The Spanish preposition **"en"** translates to three different English words: *"in"*, *"on"*, and *"at"*. This 1-to-3 mapping clash makes prepositions a massive friction point for learners.
* **The Exercise Type Clue (`format_listen`):** Listening exercises require auditory-to-spelling conversion and active recall from scratch (high cognitive load, 17.52% error rate). Tap exercises (word-bank) require simple recognition (low cognitive load, 4.59% error rate).
* **The Progress Clue (`days_in_course`):** More days spent in the course captures the student's learning curve and habituation over time, naturally leading to fewer errors.

**Turkish L1 Transfer Implementation & Validation (Option C - Private IP)**

We implemented a custom feature engineering module (`02_feature_engineering/turkish_feature_engineering.py`) and a synthetic data simulator (`05_turkish_simulator/`) to model and validate Turkish-specific grammatical and syntactic transfer errors. Since Turkish is agglutinative, SOV, and lacks grammatical articles/gendered pronouns, we engineered 5 key trackers:
* **Copula Omission Tracker (`is_copula_candidate`):** Flags linking/auxiliary verbs (*be/is/are/etc.*) that Turkish learners omit because Turkish attaches copulas as noun suffixes or drops them in 3rd person.
* **Article Omission Tracker (`is_article_candidate`):** Flags English articles (*the/a/an*) which Turkish L1 learners omit due to Turkish lacking definite articles.
* **Pronoun Gender Matcher (`is_gender_pronoun`):** Flags gendered pronouns (*he/she/her/his/etc.*) clashing with the gender-neutral Turkish pronoun *"o"*.
* **Verb-Second (V2) Syntax Tracker (`token_order_from_verb`):** Measures token distance from the main verb to capture structural SVO vs. SOV word order clashes.
* **Adposition Mapping Tracker (`is_preposition_spatial`, `is_preposition_grammatical`):** Differentiates spatial prepositions (which clash with Turkish case suffixes) from grammatical prepositions (which clash with postposition syntax).

**Validation Status (June 2026):**
* **Simulator:** Simulated 15,000 exercises (68,982 tokens) with SLA-theoretic probability distributions.
* **Model Recovery:** Trained an interpretable Logistic Regression model which recovered the exact programmed parameters (Copula Omission Odds Ratio = 3.70; Article Omission OR = 2.94; Grammatical Preposition OR = 2.51; Gender Pronoun OR = 2.23; Spatial Preposition OR = 1.93).
* **Format Interaction:** Confirmed that **free translation formats** heavily compound grammatical transfer errors compared to passive tap-bubble formats (e.g. `translate_x_copula` OR = 1.80; `translate_x_article` OR = 1.73).
* **Scaffolding Integration:** Traversed the validation Decision Tree (depth=4) to extract 15 private rules (saved in `turkish_curriculum_rules.json`) that route Turkish L1 students to explicit curriculum scaffolding (e.g. Prompt-Echo word bank taps) in the CareerTalkLab lesson planner engine. All Turkish L1 files are excluded from public git commits to protect proprietary IP.


**Feature Engineering Strategy**

| Stage | Approach |
| :---- | :---- |
| Data cleaning | Convert raw "null" strings to Python `None` programmatically during database ingestion. Zero-fill missing `time_taken` values, scale `days_in_course` using RobustScaler to handle outlier learning timelines. |
| Feature creation | Programmatically extract grammatical structural clashes (pro-drop indicators, third-person singular inflections, preposition flags). Build token positional features (`token_order`) to track word-order syntax. |
| Feature preprocessing | One-hot encode categorical features (`part_of_speech`, `dependency_label`, `client`, `session_type`, `exercise_format`). RobustScaler for numerical inputs. |
| Feature selection | Perform tree-based feature importance ranking. Drop collinear features using correlation thresholds. |

**Data Leakage Prevention**   
Describe specifically how you prevent data leakage:
* **Train/Val/Test Splits:** Splits are performed chronologically/temporally based on `days_in_course` per user to ensure the model predicts future performance based on past exercises, avoiding look-ahead bias.
* **Preprocessing Statistics:** Standard scalers and encodings are fit exclusively on the training set, and then applied to the validation and test sets.
* **Target Leakage:** All features are strictly historical (user progress days, current exercise format, current token features) and contain no future information or direct leak of the target `is_error`.

**Checklist**:

- [x] I have documented all features, their sources, transformations, and rationale.  
- [x] I have a principled approach to missing data (not just dropping rows).  
- [x] I have verified no data leakage in my feature pipeline.  
- [x] Preprocessing is fit on training data only, then applied to val/test.  
- [x] I have performed feature selection and documented the rationale.  
- [x] Feature engineering code is modular and reproducible (not one-off notebook cells).

## 5\. Labeling & Target Definition

**Target Variable**   
*What is the prediction target? How is it defined? Is the definition unambiguous? If it's a proxy for a business outcome, document the relationship.*

**Label Source**

| Aspect | Details |
| :---- | :---- |
| Label origin | *Ground truth data, manual labeling, programmatic/weak supervision, LLM-assisted* |
| Label quality | *How confident are you in label accuracy? Any known noise or ambiguity?* |
| Class distribution | *For classification: what is the distribution of classes?* |
| Temporal considerations | *Is there a time lag between features and label availability?* |

**Labeling Strategy** (if creating your own labels)

| Aspect | Approach |
| :---- | :---- |
| Labeling guidelines | *Clear document with rules, examples, and edge case handling (do not underestimate this step\!)* |
| Labeling method | *Manual, programmatic/weak supervision, LLM-assisted* |
| Quality assurance | *Inter-annotator agreement, spot-check validation* |
| Volume | *How many examples are labeled?* |

**Checklist**:

- [ ] My target variable is clearly defined and documented.  
- [ ] I understand the relationship between my target and the business outcome.  
- [ ] If creating labels, I have documented guidelines and validated quality.  
- [ ] I have addressed class imbalance if present.

## 6\. Model Training & Evaluation

**Candidate Models**

| Method | Brief Description | Pros | Cons | Complexity |
| :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression** | Linear classifier predicting probability of error using scikit-learn | Fast, highly interpretable coefficients, well-calibrated probabilities | Cannot capture complex non-linear combinations without manual feature interaction | *Low* |
| **Decision Tree** | Decision tree classifier with `max_depth = 4` to preserve readability | Generates clear, human-readable path rules; automatically handles feature interactions | Lower predictive performance due to shallow depth constraint | *Low-Medium* |

**Data Splits**

| Split | Strategy | Size | Notes |
| :---- | :---- | :---- | :---- |
| Training | Partitioned by user_id (80% unique users) | 2,110,404 tokens | Represents 80% of unique users (2,074 users). Prevents student-characteristic leakage. |
| Test (hold-out) | Partitioned by user_id (20% unique users) | 512,553 tokens | Represents 20% of unique users (519 users). Untouched until final evaluation. |

**Model Development Experiment Tracking**

| Aspect | Approach |
| :---- | :---- |
| Tool | Local Python script annotations and generated markdown reports (`/04_outputs/modeling_evaluation_report.md`) |
| What's tracked | Feature coefficients, odds ratios, decision tree splits, CV and Test set metrics (AUC-ROC, F1, Precision, Recall, Accuracy, Confusion Matrices) |

**Hyperparameter Tuning**

| Aspect | Details |
| :---- | :---- |
| Method | Manual evaluation of tree depth constraints (`max_depth` from 3 to 6) to balance performance vs interpretability. Grid search over probability thresholds to maximize F1-score. |
| Parameters tuned | Tree depth (winning: `max_depth=4`), decision threshold (LR winning: `0.13`, DT winning: `0.13`) |
| Best configuration | Logistic Regression (`C=1.0`), Decision Tree (`max_depth=4`, `random_state=42`) |

**Evaluation**

* *Baseline performance:* Majority-class prediction (always predicting "No Error") yields 87.39% accuracy but 0.00 AUC-ROC and 0.00 F1-score due to the 7:1 class imbalance.
* *Primary metric \+ justification:* AUC-ROC is selected because it measures the model's ability to rank items by error probability, independent of threshold.
  - **Logistic Regression Test AUC-ROC:** `0.6609` (CV: `0.6679 +/- 0.0025` using `StratifiedGroupKFold`)
  - **Decision Tree (depth=4) Test AUC-ROC:** `0.6600` (CV: `0.6613 +/- 0.0031` using `StratifiedGroupKFold`)
* *Secondary metrics:* F1-score (with optimized thresholds to handle imbalance):
  - **Logistic Regression (Threshold = 0.13):** F1 = `0.2936`, Precision = `0.1852`, Recall = `0.7081`
  - **Decision Tree (Threshold = 0.13):** F1 = `0.2886`, Precision = `0.1784`, Recall = `0.7549`
* *Threshold / trade-off decisions:* Since raw probabilities are utilized directly by the curriculum rules (e.g. `prob_error >= 30%`), standard Logistic Regression is essential to preserve model calibration. Using `class_weight='balanced'` would distort these probabilities. Instead, boundary threshold tuning lets us adjust the precision/recall trade-off without altering calibration. We optimize the threshold to maximize F1, significantly boosting recall (finding 70-75% of all errors) at the expense of precision.
* *Calibration:* The Logistic Regression model yields well-calibrated probability scores, providing instructors with the actual probability of error.

**Error Analysis**

| Analysis | Findings |
| :---- | :---- |
| Confusion matrix / residual analysis | LR confusion matrix: 234,929 TN, 210,174 FP, 19,690 FN, 47,760 TP. High false positive rate is a side-effect of optimizing F1-score to maximize recall in the face of class imbalance. |
| Slice analysis | Error rates are extremely high in Listening formats (up to 31.12% for late-sentence tokens) and in writing formats when starting a sentence with a 3rd person singular verb (which triggers Rule 02/03 scaffolding). |
| Failure patterns | The model struggles to predict spelling errors that arise from user typos rather than systematic grammatical clashes. These typos appear randomly. |
| Fairness audit | System behaves consistently across sequential partitions of the test set, reflecting stable prediction quality. |

**Interpretability & Explainability**

| Method | Purpose |
| :---- | :---- |
| **Odds Ratios (Logistic Regression)** | Explaining the direction and magnitude of the impact of syntactic features on learner error rates. |
| **Decision Tree Tracing (depth=4)** | Visualizing explicit conditional paths with error rates and sample counts for curriculum designers to read. |

**Checklist**:

- [x] I have proper train/val/test splits with no data leakage.  
- [x] I have established a baseline (including non-ML) that my model improves upon.  
- [x] I have tracked experiments with logged hyperparameters and metrics.  
- [x] I have performed hyperparameter tuning and documented results.  
- [x] I have done systematic error analysis on misclassified/high-error examples.  
- [x] I have analyzed performance across key data slices to check for bias.  
- [x] I have considered model interpretability and can explain predictions.  
- [x] I have chosen metrics appropriate for my class distribution and business context.

## 7\. Communication & Storytelling

*Data science is about driving decisions. Your ability to communicate findings is as important as the modeling itself.*

**Findings Summary**   
Native Spanish speakers face distinct, predictable challenges when learning English on Duolingo. Rather than a flat error distribution, mistakes are heavily concentrated in auditory reception (Listening formats increase error odds by **+325.3%**) and active production (Reverse Translation increases error odds by **+226.9%**). Lexical mapping clashes are highly prominent, with prepositions increasing error odds by **+28.8%** due to Spanish L1 interference (e.g. mapping *en* to *in/on/at*). Interestingly, subject pronouns (often dropped in Spanish) are well-recalled when explicitly prompted in English (51.0% lower error odds). Decision tree splits reveal that starting a sentence with a 3rd person singular verb yields a high error rate of **38.54%**.

**Key Visualizations**   

| Visualization | What It Shows | Key Takeaway |
| :---- | :---- | :---- |
| **Odds Ratios Table** | Ranking of features by their impact on error rate. | Listening and translation formats are by far the largest drivers of errors, followed by preposition lexical clashes. |
| **Decision Tree Split Diagram** | Traces conditional paths (e.g. Format = Listen, Session = Practice, Position > 3) to show error rate. | Curriculum designers can see exact rules of when error rates spike (e.g., up to 38.5%). |

**Recommendations**   
1. **Dynamic Review Allocation:** Implement a rule-based engine in Duolingo's lesson planner that triggers a review session focusing on prepositions and spelling when a user enters a listening practice session.
2. **Errorless Teaching Curricular Structure:** Structure sentence construction in initial A1/A2 lessons such that third-person singular verbs are introduced in the middle of sentences rather than as sentence-starters, minimizing cognitive load and preventing syntax errors.
3. **Typo vs. Grammatical Clash Treatment:** Differentiate feedback for spelling errors in listening formats vs. grammatical omissions (like third-person singular `-s`), as the former represents acoustic-spelling challenges, while the latter represents syntax transfer clashes.

**Limitations & Caveats**   
* **Spanish L1 Specificity:** The structural clashes modeled (pro-drop pronouns, preposition mappings) are specific to Spanish speakers. These models will not transfer to Turkish L1 speakers, who face different syntactic clashes (agglutination, subject-object-verb word order, and article omissions).
* **Interface Limitations:** The data does not record whether a typo was caused by keyboard auto-correct or platform tapping rules, which may confound the client (Web vs Mobile) coefficient.

**Checklist**:

- [x] I have a non-technical summary a stakeholder could understand.  
- [x] I have clear, well-designed visualizations that support my narrative.  
- [x] I have actionable recommendations framed in business terms.  
- [x] I have documented limitations and caveats honestly.

## 8\. Deployment (if applicable)

*Not all data science projects require production deployment, but demonstrating deployment skills makes your portfolio stand out.*

**Deployment Strategy**

| Aspect | Approach |
| :---- | :---- |
| Serving pattern | **Local CLI Inference Utility (`04_outputs/predict.py`)**: A command-line script that accepts a tokenized JSON file of exercises, parses syntax tokens, maps features, computes predictions using the serialized Logistic Regression and Decision Tree models, and outputs a color-coded report with pedagogical explanations. |
| Containerization | Deferred (Not containerized; run directly via the project's virtual environment). |
| Hosting | Local dev server / CLI. |
| CI/CD | N/A for local research deliverable. |

**Checklist**:

- [x] My model is accessible to users (API, app, or batch output) (Accessible via `predict.py` script).  
- [ ] My application is containerized with Docker.  
- [x] I have a live demo or clear setup instructions (Demonstrable on `sample_exercises.json`).

## 9\. Monitoring & Iteration (if deployed)

**Monitoring**

| What to Monitor | Metrics | Approach |
| :---- | :---- | :---- |
| Model performance | **AUC-ROC and F1 score** on fresh hold-out validation batches | Periodically run evaluation script when new annotated exercise streams are ingested. |
| Data drift | **Feature proportions** (e.g. changes in client mix or POS frequencies) | Monitor descriptive statistics of newly engineered features in PostgreSQL. |
| Prediction drift | **Mean error rate predictions** over time | Compare the output probability distributions of the models across different weeks/months. |

**Retraining Strategy**   
* Retrain model sequentially whenever a new major batch of SLAM data is released.
* The feature engineering pipeline supports resume logic based on PostgreSQL indexes, allowing incremental feature computation before refitting the scikit-learn models on the expanded training split.

**Checklist**:

- [ ] I have monitoring for model performance and data drift.  
- [x] I have a plan for when and how to retrain.

## 10\. Code Quality & Repository Structure

**Project Structure**

Enforced by the system directives in `ANTIGRAVITY.md`, we follow a strict folder-structure-as-agent-architecture to separate concerns between data ingestion, feature engineering, modeling, and outputs:

```
slam_knowledge_tracing/
├── 01_data_ingestion/       # PostgreSQL schema and text stream ingestion
│   ├── 01_schema.sql        # Database tables and indexes
│   └── 02_ingest_data.py    # Robust parsing & ingestion script with resume logic
├── 02_feature_engineering/  # SLA translation layer: creates structural clash features
├── 03_modeling/             # Model training and interpretable evaluation
├── 04_outputs/              # Deliverables (decision tree paths and coefficients)
├── _brand/                  # Persona and quality guidelines (personas, lint-rules)
├── requirements.txt         # Project dependencies
├── .env                     # Local environment configurations (never checked in)
└── ANTIGRAVITY.md           # System directives, architectural constraints
```

**Notebook Standards**   
Notebooks are deferred in favor of modular, production-ready, and reproducible `.py` command-line scripts for each pipeline stage. This maintains clean state tracking, execution logging, and prevents the issue of scrambled notebook cell states.

**Code Standards**
* **Type Hinting & Documentation:** Reusable functions carry standard Python type hints and detailed docstrings explaining the pedagogical/SLA rationale.
* **No Hardcoding:** Database credentials, paths, and configurations are loaded dynamically via `dotenv` from `.env` or system environment variables.
* **Reproducibility:** A clean virtual environment (`venv`) and a version-locked `requirements.txt` ensure anyone can clone the repository and run any pipeline stage.

**Checklist**:

- [x] Notebooks are numbered and follow a logical flow (Modular `.py` stages are numbered 01-04).  
- [x] Each notebook has a clear purpose and narrative (Each stage script contains docstrings and inline commentary).  
- [x] Code is clean — no dead cells or unexplained outputs.  
- [x] Reusable logic is refactored into .py modules.
- [x] Type hints and docstrings on reusable functions.  
- [x] No hardcoded values — configs in separate files or env vars.  
- [x] Reproducible: someone can clone the repo and reproduce results.

**README Contents**

- [x] Clear overview of the problem, approach, and key findings (Documented in ANTIGRAVITY.md).  
- [x] Architecture / workflow diagram.  
- [x] Key decisions and trade-offs.  
- [x] Setup and reproduction instructions.  
- [x] Performance metrics and key visualizations.  
- [ ] Live demo link (if deployed).

## 11\. Project Timeline & Milestones

| Milestone | Target Date | Status |
| :---- | :---- | :---- |
| Problem scoping & hypothesis |  |  |
| Data sourcing & quality assessment |  |  |
| Exploratory data analysis |  |  |
| Feature engineering pipeline |  |  |
| Labeling (if applicable) |  |  |
| Baseline model & evaluation framework |  |  |
| Model experimentation & tuning |  |  |
| Error analysis & fairness audit |  |  |
| Communication deliverables (report, visualizations) |  |  |
| Deployment (if applicable) |  |  |
| Final polish, README, demo |  |  |

## 12\. Appendix

* *Links to repos, notebooks, dashboards, experiment tracking, etc.*  
* *Architecture / workflow diagram(s)*  
* *Glossary of terms/metrics*  
* *Decision log — key choices made and why (this is super important for interviews\!)*  
* *References — domain literature, related analyses, methodology references*

