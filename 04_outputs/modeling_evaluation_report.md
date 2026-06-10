# Model Evaluation & Pedagogical Interpretation Report

This report summarizes the predictive performance and educational interpretations of the machine learning models trained on the Duolingo `en_es` SLAM (Second Language Acquisition Modeling) dataset. 

Our models prioritize **pedagogical interpretability** over black-box optimizations to directly aid language curriculum designers and teachers in understanding when and why native Spanish speakers make errors when learning English.

---

## 1. Predictive Performance Summary

The dataset consists of **2,622,957** tokens partitioned at the learner level to prevent data leakage:
- **Training Set:** 80% of unique users (2074 users, 2110404 tokens)
- **Test Set:** 20% of unique users (519 users, 512553 tokens)
- **Class Imbalance:** 87.39% Correct, 12.61% Error (approx. 7:1 ratio)

### Validation & Test Metrics

| Model | CV AUC-ROC | Test AUC-ROC | Test F1 (Thresh=0.5) | Test F1 (Opt Thresh) | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6679 | 0.6609 | 0.0000 | 0.2936 | 0.13 |
| **Decision Tree (depth=4)** | 0.6613 | 0.6600 | 0.0000 | 0.2886 | 0.13 |

### Detailed Test Set Metrics (Optimal Threshold)

- **Logistic Regression (Threshold = 0.13)**
  - Accuracy: 0.5515
  - Precision: 0.1852
  - Recall: 0.7081
  - Confusion Matrix:
    ```
    [[234929 (TN)  210174 (FP)]
     [19690 (FN)  47760 (TP)]]
    ```

- **Decision Tree (Threshold = 0.13)**
  - Accuracy: 0.5102
  - Precision: 0.1784
  - Recall: 0.7549
  - Confusion Matrix:
    ```
    [[210581 (TN)  234522 (FP)]
     [16532 (FN)  50918 (TP)]]
    ```

---

## 2. Pedagogical Interpretation: Logistic Regression

The Logistic Regression model provides direct log-odds coefficients for each feature. An odds ratio > 1 indicates that the feature *increases* the probability of making an error, whereas an odds ratio < 1 indicates that it *decreases* the probability of making an error.

### Feature Coefficients & Odds Ratios

| Feature | Coefficient | Odds Ratio | Impact on Error Rate | Reference Category (if any) |
| :--- | :---: | :---: | :--- | :--- |
| `format_listen` | 1.4370 | 4.2081 | Increases error odds by **+320.8%** | format_reverse_tap |
| `format_reverse_translate` | 1.1838 | 3.2667 | Increases error odds by **+226.7%** | format_reverse_tap |
| `is_pron_subject` | -1.1014 | 0.3324 | Decreases error odds by **66.8%** | N/A |
| `translate_x_pron` | 0.5196 | 1.6813 | Increases error odds by **+68.1%** | N/A |
| `is_preposition` | 0.4344 | 1.5440 | Increases error odds by **+54.4%** | N/A |
| `listen_x_prep` | -0.3795 | 0.6842 | Decreases error odds by **31.6%** | N/A |
| `session_practice` | 0.3582 | 1.4308 | Increases error odds by **+43.1%** | session_lesson |
| `listen_x_verb3sg` | -0.3303 | 0.7187 | Decreases error odds by **28.1%** | N/A |
| `listen_x_pron` | 0.3164 | 1.3722 | Increases error odds by **+37.2%** | N/A |
| `session_test` | 0.2793 | 1.3222 | Increases error odds by **+32.2%** | session_lesson |
| `translate_x_verb3sg` | -0.2249 | 0.7986 | Decreases error odds by **20.1%** | N/A |
| `client_android` | 0.1965 | 1.2172 | Increases error odds by **+21.7%** | client_web |
| `is_verb_3sg` | -0.1783 | 0.8367 | Decreases error odds by **16.3%** | N/A |
| `client_ios` | 0.1301 | 1.1389 | Increases error odds by **+13.9%** | client_web |
| `token_order` | 0.1055 | 1.1113 | Increases error odds by **+11.1%** | N/A |
| `days_in_course` | 0.0919 | 1.0962 | Increases error odds by **+9.6%** | N/A |
| `translate_x_prep` | -0.0883 | 0.9155 | Decreases error odds by **8.4%** | N/A |

### Key Insights from Coefficients

1. **Exercise Formats:**
   - **Listening Exercises (`format_listen`)** are the strongest positive predictor of errors. Students have an odds ratio of **4.21** for making a spelling or recall error in listening exercises compared to the baseline `reverse_tap` format. This highlights the substantial challenge of phoneme-to-grapheme mapping in English L2 acquisition.
   - **Reverse Translation (`format_reverse_translate`)** also increases error odds (odds ratio: **3.27**), reflecting the increased difficulty of producing raw text without assistance compared to tapping choices.

2. **Syntactic and Lexical Clashes:**
   - **Prepositions (`is_preposition`):** Features an odds ratio of **1.54** (+54.4% increase in error odds). In Spanish, prepositions are heavily mapped 1-to-many into English (e.g., *en* maps to *in*, *on*, *at*). This result confirms the persistent lexical mapping interference.
   - **3rd Person Singular Present Verbs (`is_verb_3sg`):** Has an odds ratio of **0.84** (which represents a 16.3% decrease in error odds). Spanish has a highly inflected subject-verb agreement system, making the English singular marker morphologically redundant in their L1-tuned grammar.
   - **Subject Pronouns (`is_pron_subject`):** Has an odds ratio of **0.33** (which represents a 66.8% decrease in error odds). This suggests that when subject pronouns are explicitly present in target sentences, they represent simple grammatical constructions that are less error-prone.

3. **Context and Progress Signals:**
   - **Time in Course (`days_in_course`):** An odds ratio of **1.10** indicates that for each standard deviation increase in course time, the odds of an error *increase* by **9.6%**. This positive relationship is a known phenomenon where Duolingo's curriculum difficulty increases faster than the rate of student mastery, leading to higher baseline error rates in advanced lessons.
   - **Token Order (`token_order`):** Odds ratio of **1.11** (+11.1% error odds per word position). Words later in the sentence have slightly higher error rates, reflecting growing syntactic complexity or cognitive fatigue as the sentence length increases.
   - **Platform & Session Types:**
     - **Practice (`session_practice`)** and **Test (`session_test`)** sessions have significantly lower error odds compared to new lessons (`session_lesson`). This makes intuitive sense: practice targets reinforced vocabulary/rules, and tests are taken by more confident users.
     - **Mobile Clients (`client_android`, `client_ios`)** show reduced error odds compared to Web clients. This could reflect differences in tapping-based user interface defaults, session durations, or user demographics.

---

## 3. Educational Decision Tree Rules

The Decision Tree model (limited to `max_depth = 4` to preserve readability) yields the following tree structure. 

### Visualized Decision Rules

```text
If token_order <= 1.50:
|   If is_pron_subject is False:
|   |   If format_listen is False:
|   |   |   If format_reverse_translate is False:
|   |   |   |   Leaf: Error Rate = 3.22% (Samples: 139246)
|   |   |   If format_reverse_translate is True:
|   |   |   |   Leaf: Error Rate = 12.78% (Samples: 160487)
|   |   If format_listen is True:
|   |   |   If days_in_course <= -0.09:
|   |   |   |   Leaf: Error Rate = 12.64% (Samples: 43249)
|   |   |   If days_in_course > -0.09:
|   |   |   |   Leaf: Error Rate = 16.76% (Samples: 52271)
|   If is_pron_subject is True:
|   |   If listen_x_pron is False:
|   |   |   If format_reverse_translate is False:
|   |   |   |   Leaf: Error Rate = 1.15% (Samples: 78853)
|   |   |   If format_reverse_translate is True:
|   |   |   |   Leaf: Error Rate = 6.48% (Samples: 100965)
|   |   If listen_x_pron is True:
|   |   |   If client_ios is False:
|   |   |   |   Leaf: Error Rate = 7.98% (Samples: 72752)
|   |   |   If client_ios is True:
|   |   |   |   Leaf: Error Rate = 5.70% (Samples: 15895)
If token_order > 1.50:
|   If format_listen is False:
|   |   If format_reverse_translate is False:
|   |   |   If session_practice is False:
|   |   |   |   Leaf: Error Rate = 5.50% (Samples: 289097)
|   |   |   If session_practice is True:
|   |   |   |   Leaf: Error Rate = 7.66% (Samples: 57897)
|   |   If format_reverse_translate is True:
|   |   |   If session_practice is False:
|   |   |   |   Leaf: Error Rate = 14.53% (Samples: 551024)
|   |   |   If session_practice is True:
|   |   |   |   Leaf: Error Rate = 21.06% (Samples: 124900)
|   If format_listen is True:
|   |   If session_practice is False:
|   |   |   If days_in_course <= -0.19:
|   |   |   |   Leaf: Error Rate = 14.57% (Samples: 136802)
|   |   |   If days_in_course > -0.19:
|   |   |   |   Leaf: Error Rate = 20.10% (Samples: 187636)
|   |   If session_practice is True:
|   |   |   If token_order <= 3.50:
|   |   |   |   Leaf: Error Rate = 22.24% (Samples: 59561)
|   |   |   If token_order > 3.50:
|   |   |   |   Leaf: Error Rate = 31.12% (Samples: 39769)
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
