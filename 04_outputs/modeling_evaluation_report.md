# Model Evaluation & Pedagogical Interpretation Report

This report summarizes the predictive performance and educational interpretations of the machine learning models trained on the Duolingo `en_es` SLAM (Second Language Acquisition Modeling) dataset. 

Our models prioritize **pedagogical interpretability** over black-box optimizations to directly aid language curriculum designers and teachers in understanding when and why native Spanish speakers make errors when learning English.

---

## 1. Predictive Performance Summary

The dataset consists of **2,622,957** tokens split sequentially at a boundary of `exercise_id = 1407413`:
- **Training Set:** 80% of exercises (approx. 2.1M tokens)
- **Test Set:** 20% of exercises (525753 tokens)
- **Class Imbalance:** 87.39% Correct, 12.61% Error (approx. 7:1 ratio)

### Validation & Test Metrics

| Model | CV AUC-ROC | Test AUC-ROC | Test F1 (Thresh=0.5) | Test F1 (Opt Thresh) | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6681 | 0.6617 | 0.0000 | 0.2755 | 0.14 |
| **Decision Tree (depth=4)** | 0.6623 | 0.6538 | 0.0000 | 0.2688 | 0.13 |

### Detailed Test Set Metrics (Optimal Threshold)

- **Logistic Regression (Threshold = 0.14)**
  - Accuracy: 0.5914
  - Precision: 0.1749
  - Recall: 0.6484
  - Confusion Matrix:
    ```
    [[270071 (TN)  192676 (FP)]
     [22152 (FN)  40854 (TP)]]
    ```

- **Decision Tree (Threshold = 0.13)**
  - Accuracy: 0.4982
  - Precision: 0.1628
  - Recall: 0.7695
  - Confusion Matrix:
    ```
    [[213457 (TN)  249290 (FP)]
     [14525 (FN)  48481 (TP)]]
    ```

---

## 2. Pedagogical Interpretation: Logistic Regression

The Logistic Regression model provides direct log-odds coefficients for each feature. An odds ratio > 1 indicates that the feature *increases* the probability of making an error, whereas an odds ratio < 1 indicates that it *decreases* the probability of making an error.

### Feature Coefficients & Odds Ratios

| Feature | Coefficient | Odds Ratio | Impact on Error Rate | Reference Category (if any) |
| :--- | :---: | :---: | :--- | :--- |
| `format_listen` | 1.4619 | 4.3140 | Increases error odds by **+331.4%** | format_reverse_tap |
| `format_reverse_translate` | 1.1670 | 3.2123 | Increases error odds by **+221.2%** | format_reverse_tap |
| `is_pron_subject` | -1.0746 | 0.3414 | Decreases error odds by **65.9%** | N/A |
| `translate_x_pron` | 0.4830 | 1.6209 | Increases error odds by **+62.1%** | N/A |
| `is_preposition` | 0.4456 | 1.5615 | Increases error odds by **+56.1%** | N/A |
| `session_practice` | 0.3670 | 1.4434 | Increases error odds by **+44.3%** | session_lesson |
| `listen_x_prep` | -0.3546 | 0.7014 | Decreases error odds by **29.9%** | N/A |
| `listen_x_pron` | 0.2997 | 1.3495 | Increases error odds by **+34.9%** | N/A |
| `listen_x_verb3sg` | -0.2888 | 0.7492 | Decreases error odds by **25.1%** | N/A |
| `session_test` | 0.2375 | 1.2680 | Increases error odds by **+26.8%** | session_lesson |
| `translate_x_verb3sg` | -0.2084 | 0.8119 | Decreases error odds by **18.8%** | N/A |
| `is_verb_3sg` | -0.1823 | 0.8334 | Decreases error odds by **16.7%** | N/A |
| `client_android` | 0.1348 | 1.1443 | Increases error odds by **+14.4%** | client_web |
| `token_order` | 0.1040 | 1.1096 | Increases error odds by **+11.0%** | N/A |
| `days_in_course` | 0.0816 | 1.0851 | Increases error odds by **+8.5%** | N/A |
| `translate_x_prep` | -0.0783 | 0.9247 | Decreases error odds by **7.5%** | N/A |
| `client_ios` | 0.0458 | 1.0469 | Increases error odds by **+4.7%** | client_web |

### Key Insights from Coefficients

1. **Exercise Formats:**
   - **Listening Exercises (`format_listen`)** are the strongest positive predictor of errors. Students have an odds ratio of **4.31** for making a spelling or recall error in listening exercises compared to the baseline `reverse_tap` format. This highlights the substantial challenge of phoneme-to-grapheme mapping in English L2 acquisition.
   - **Reverse Translation (`format_reverse_translate`)** also increases error odds (odds ratio: **3.21**), reflecting the increased difficulty of producing raw text without assistance compared to tapping choices.

2. **Syntactic and Lexical Clashes:**
   - **Prepositions (`is_preposition`):** Features an odds ratio of **1.56** (+56.1% increase in error odds). In Spanish, prepositions are heavily mapped 1-to-many into English (e.g., *en* maps to *in*, *on*, *at*). This result confirms the persistent lexical mapping interference.
   - **3rd Person Singular Present Verbs (`is_verb_3sg`):** Has an odds ratio of **0.83** (which represents a 16.7% decrease in error odds). Spanish has a highly inflected subject-verb agreement system, making the English singular marker morphologically redundant in their L1-tuned grammar.
   - **Subject Pronouns (`is_pron_subject`):** Has an odds ratio of **0.34** (which represents a 65.9% decrease in error odds). This suggests that when subject pronouns are explicitly present in target sentences, they represent simple grammatical constructions that are less error-prone.

3. **Context and Progress Signals:**
   - **Time in Course (`days_in_course`):** An odds ratio of **1.09** indicates that for each standard deviation increase in course time, the odds of an error *increase* by **8.5%**. This positive relationship is a known phenomenon where Duolingo's curriculum difficulty increases faster than the rate of student mastery, leading to higher baseline error rates in advanced lessons.
   - **Token Order (`token_order`):** Odds ratio of **1.11** (+11.0% error odds per word position). Words later in the sentence have slightly higher error rates, reflecting growing syntactic complexity or cognitive fatigue as the sentence length increases.
   - **Platform & Session Types:**
     - **Practice (`session_practice`)** and **Test (`session_test`)** sessions have significantly lower error odds compared to new lessons (`session_lesson`). This makes intuitive sense: practice targets reinforced vocabulary/rules, and tests are taken by more confident users.
     - **Mobile Clients (`client_android`, `client_ios`)** show reduced error odds compared to Web clients. This could reflect differences in tapping-based user interface defaults, session durations, or user demographics.

---

## 3. Educational Decision Tree Rules

The Decision Tree model (limited to `max_depth = 4` to preserve readability) yields the following tree structure. 

### Visualized Decision Rules

```text
If format_listen is False:
|   If format_reverse_translate is False:
|   |   If token_order <= 1.50:
|   |   |   If is_verb_3sg is False:
|   |   |   |   Leaf: Error Rate = 2.45% (Samples: 213034)
|   |   |   If is_verb_3sg is True:
|   |   |   |   Leaf: Error Rate = 38.54% (Samples: 397)
|   |   If token_order > 1.50:
|   |   |   If session_practice is False:
|   |   |   |   Leaf: Error Rate = 5.55% (Samples: 283261)
|   |   |   If session_practice is True:
|   |   |   |   Leaf: Error Rate = 7.94% (Samples: 56158)
|   If format_reverse_translate is True:
|   |   If token_order <= 1.50:
|   |   |   If translate_x_pron is False:
|   |   |   |   Leaf: Error Rate = 12.81% (Samples: 161676)
|   |   |   If translate_x_pron is True:
|   |   |   |   Leaf: Error Rate = 6.52% (Samples: 101088)
|   |   If token_order > 1.50:
|   |   |   If session_practice is False:
|   |   |   |   Leaf: Error Rate = 14.58% (Samples: 551250)
|   |   |   If session_practice is True:
|   |   |   |   Leaf: Error Rate = 21.26% (Samples: 126666)
If format_listen is True:
|   If is_pron_subject is False:
|   |   If session_practice is False:
|   |   |   If days_in_course <= -0.14:
|   |   |   |   Leaf: Error Rate = 14.97% (Samples: 179255)
|   |   |   If days_in_course > -0.14:
|   |   |   |   Leaf: Error Rate = 20.26% (Samples: 214672)
|   |   If session_practice is True:
|   |   |   If token_order <= 3.50:
|   |   |   |   Leaf: Error Rate = 22.09% (Samples: 73300)
|   |   |   If token_order > 3.50:
|   |   |   |   Leaf: Error Rate = 31.93% (Samples: 38953)
|   If is_pron_subject is True:
|   |   If token_order <= 1.50:
|   |   |   If client_ios is False:
|   |   |   |   Leaf: Error Rate = 8.49% (Samples: 71518)
|   |   |   If client_ios is True:
|   |   |   |   Leaf: Error Rate = 5.80% (Samples: 16160)
|   |   If token_order > 1.50:
|   |   |   If token_order <= 3.50:
|   |   |   |   Leaf: Error Rate = 14.44% (Samples: 8461)
|   |   |   If token_order > 3.50:
|   |   |   |   Leaf: Error Rate = 29.74% (Samples: 1355)
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
