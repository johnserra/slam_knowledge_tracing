# Antigravity Context: SLA-Theoretic Knowledge Tracing

## System Directives
*   **Methodology:** Interpretable Context Methodology (ICM). The folder structure dictates the workflow. Maintain strict separation of concerns between data ingestion, feature engineering, and modeling.
*   **Prime Directive:** Function before aesthetics. Prioritize seamless operations, bulletproof error handling, and logical clarity over clever or condensed code. 
*   **Modeling Constraints:** NO black-box algorithms. NO deep learning or neural networks. Use standard, highly interpretable `scikit-learn` models (e.g., Logistic Regression, Decision Trees). 
*   **Domain Focus:** The primary goal is pedagogical interpretability, not maximizing AUC. Features must map to verifiable Second Language Acquisition (SLA) concepts (e.g., CEFR levels, Output Hypothesis, interlanguage stages).

## Architecture & Agent Roles
1.  `/01_data_ingestion`: Handles the heavy lifting. Python scripts here should exclusively interact with the local PostgreSQL database on the Linux server. Do not load massive CSVs into memory; use SQL aggregations where possible.
2.  `/02_feature_engineering`: The translation layer. Scripts here read queried data and append new programmatic columns representing SLA principles (e.g., tagging third-person singular errors).
3.  `/03_modeling`: The execution layer. Trains interpretable models to validate the pedagogical features against the actual learner error rates.
4.  `/04_outputs`: Stores the deliverables. Focus on extracting decision tree pathways and feature importance coefficients that an English instructor can actually read and understand.

## Operational Rules
*   When generating Python code, prioritize readability. Use clear, descriptive variable names.
*   Document the pedagogical reasoning behind every feature engineering script using inline comments.
*   Ensure all database queries use parameterized inputs to prevent injection and maintain stability.
