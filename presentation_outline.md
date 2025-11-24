## Presentation Outline (≈20 minutes)

### 1. Predictive task (3 minutes)
- Introduce the UCI Wine Quality dataset and motivate the low/medium/high classification framing.
- Clarify evaluation criteria (accuracy, macro F1, balanced accuracy) and what constitutes success vs. baselines.

### 2. Data context + EDA (5 minutes)
- Describe collection process (blind tastings + lab chemistry) and preprocessing (semicolons, no nulls, label bucketing).
- Show the distribution plot, summary statistics, and correlation heat map from the notebook.
- Highlight domain insights (alcohol, sulphates, volatile acidity trends).

### 3. Modeling choices (4 minutes)
- Walk through the `goob_ai.modeling` registry: dummy baseline, logistic regression, random forest, histogram gradient boosting.
- Explain how standardization + ColumnTransformer keeps pipelines clean.
- Discuss complexity trade-offs (training time vs. interpretability).

### 4. Evaluation + diagnostics (5 minutes)
- Present the cross-validation table, then the holdout leaderboard.
- Justify why random forests are preferred (macro F1, balanced accuracy gains).
- Show the confusion matrix + classification report, emphasizing that misclassifications are mostly adjacent classes.

### 5. Related work + next steps (3 minutes)
- Cite Cortez et al. (2009) and recent Kaggle analyses.
- Connect results to course material and propose future improvements (ordinal regression, conformal intervals, richer features).
- Close with clear instructions for reviewers on where to find notebook sections and slides.

