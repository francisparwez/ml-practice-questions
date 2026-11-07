# 🧠 Machine Learning Practice Questions in Python

A hands-on collection of **48 machine learning practice questions** designed to build practical understanding of eight popular machine learning algorithms using Python, from beginner to intermediate level.

## 🎯 Objectives

- Understand how common machine learning algorithms work through practical implementation.
- Practice training models and making predictions using `scikit-learn`.
- Learn regression, classification, and unsupervised clustering.
- Evaluate model performance using appropriate metrics.
- Experiment with feature scaling, hyperparameter tuning, and cross-validation.
- Develop practical machine learning skills through progressively challenging exercises.

## 🛠️ Technologies & Libraries

- **Python** — Programming language
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation and analysis
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning algorithms, preprocessing, and evaluation
- **XGBoost** — Gradient boosting
- **Jupyter Notebook** — Interactive experimentation

## 📚 Practice Questions

This repository contains 48 questions across eight machine learning algorithms, with three beginner and three intermediate tasks for each algorithm.

### 📈 1. Linear Regression — Questions 1–6

**Objective:** Learn to predict continuous numerical values and understand regression model behavior.

#### Level ML1 — Beginner

- [x] **Q1. Predict Student Exam Scores** — Trained a Linear Regression model using study hours to predict exam scores; displayed the prediction for 7 hours, coefficient, and intercept. Implementation: `01-linear-regression/a_student-exam-scores.py`.
- [ ] **Q2. Predict Employee Salaries** — Generated a 50-row synthetic dataset, split it into training and testing sets, and trained a Linear Regression model using years of experience. Added a `predict_salary()` function and predicted a salary for 4.5 years of experience; predictions for the full test set are not yet implemented. Implementation: `01-linear-regression/b_predict_employee_salaries.py`.
- [ ] **Q3. Evaluate a Regression Model** — Evaluate house price predictions using MAE, MSE, and R².

#### Level ML2 — Intermediate

- [ ] **Q4. Predict Sales Using Advertising Expenditure** — Apply multiple Linear Regression using three advertising features.
- [ ] **Q5. Investigate the Effect of Outliers** — Compare regression models before and after introducing an outlier.
- [ ] **Q6. Compare Polynomial and Linear Regression** — Compare linear and polynomial models on curved data.

**Concepts:** `LinearRegression()`, `.fit()`, `.predict()`, coefficients, intercept, MAE, MSE, R², polynomial features.

### 🎯 2. Logistic Regression — Questions 7–12

**Objective:** Understand classification, predicted probabilities, classification metrics, and imbalanced datasets.

#### Level ML1 — Beginner

- [ ] **Q7. Predict Whether a Student Passes** — Classify pass/fail outcomes using study hours.
- [ ] **Q8. Detect Spam Messages** — Classify messages using text vectorization and Logistic Regression.
- [ ] **Q9. Evaluate Classification Predictions** — Measure customer churn classification performance.

#### Level ML2 — Intermediate

- [ ] **Q10. Predict Customer Churn Using Multiple Features** — Build a classification pipeline with numerical and categorical features.
- [ ] **Q11. Experiment with Classification Thresholds** — Compare classification results at different probability thresholds.
- [ ] **Q12. Handle Imbalanced Classification** — Compare baseline and class-weighted models on fraud data.

**Concepts:** `LogisticRegression()`, `predict_proba()`, `CountVectorizer()`, precision, recall, F1-score, confusion matrix, classification thresholds, class imbalance.

### 🌳 3. Decision Trees — Questions 13–18

**Objective:** Understand tree-based predictions, decision rules, feature importance, and overfitting.

#### Level ML1 — Beginner

- [ ] **Q13. Predict Whether a Customer Purchases a Product** — Train a Decision Tree classifier using customer features.
- [ ] **Q14. Visualize a Decision Tree** — Visualize decision rules and follow prediction paths.
- [ ] **Q15. Predict Product Prices** — Train a Decision Tree regressor and evaluate its predictions.

#### Level ML2 — Intermediate

- [ ] **Q16. Investigate Overfitting** — Compare shallow and unrestricted trees.
- [ ] **Q17. Tune a Decision Tree Model** — Experiment with tree depth and sample-size parameters.
- [ ] **Q18. Analyze Feature Importance** — Rank the features used to predict employee attrition.

**Concepts:** `DecisionTreeClassifier()`, `DecisionTreeRegressor()`, `plot_tree()`, `max_depth`, `min_samples_split`, `min_samples_leaf`, feature importance, `GridSearchCV()`.

### 🌲 4. Random Forest — Questions 19–24

**Objective:** Explore ensemble learning, model stability, feature importance, and hyperparameter tuning.

#### Level ML1 — Beginner

- [ ] **Q19. Predict Flower Species** — Classify Iris flowers using Random Forest.
- [ ] **Q20. Predict Employee Salaries** — Use Random Forest regression to estimate salaries.
- [ ] **Q21. Compare One Tree with a Forest** — Compare a Decision Tree against a Random Forest.

#### Level ML2 — Intermediate

- [ ] **Q22. Investigate the Number of Trees** — Compare model performance and training time across different forest sizes.
- [ ] **Q23. Tune a Random Forest Classifier** — Optimize forest parameters using cross-validation.
- [ ] **Q24. Identify Important Customer Features** — Rank customer purchase prediction features.

**Concepts:** `RandomForestClassifier()`, `RandomForestRegressor()`, `n_estimators`, `max_depth`, `min_samples_leaf`, `RandomizedSearchCV()`, ensemble learning, feature importance.

### 🚀 5. XGBoost — Questions 25–30

**Objective:** Practice gradient boosting, model tuning, regression, classification, and imbalanced classification.

#### Level ML1 — Beginner

- [ ] **Q25. Predict Customer Churn with XGBoost** — Train an XGBoost classifier.
- [ ] **Q26. Predict Sales Using XGBoost Regression** — Predict sales revenue using an XGBoost regressor.
- [ ] **Q27. Compare XGBoost with a Decision Tree** — Compare test performance and generalization.

#### Level ML2 — Intermediate

- [ ] **Q28. Tune XGBoost Hyperparameters** — Experiment with estimator count, tree depth, and learning rate.
- [ ] **Q29. Investigate Learning Rate and Boosting Rounds** — Compare validation performance across configurations.
- [ ] **Q30. Analyze Credit Card Fraud Predictions** — Evaluate XGBoost on imbalanced fraud data.

**Concepts:** `XGBClassifier()`, `XGBRegressor()`, `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`, cross-validation, ROC-AUC.

### 📐 6. Support Vector Machine (SVM) — Questions 31–36

**Objective:** Understand decision boundaries, kernel functions, feature scaling, and parameter tuning.

#### Level ML1 — Beginner

- [ ] **Q31. Classify Breast Cancer Cases** — Train an SVM classifier using standardized features.
- [ ] **Q32. Compare SVM Kernels** — Compare linear and RBF kernels.
- [ ] **Q33. Predict Customer Categories** — Visualize SVM predictions and decision boundaries.

#### Level ML2 — Intermediate

- [ ] **Q34. Investigate the Effect of Feature Scaling** — Compare SVM performance with and without standardization.
- [ ] **Q35. Tune SVM Parameters** — Optimize `C` and `gamma` using cross-validation.
- [ ] **Q36. Classify Non-Linear Data** — Compare linear and RBF SVMs on non-linear patterns.

**Concepts:** `SVC()`, `StandardScaler()`, linear kernel, RBF kernel, `C`, `gamma`, `GridSearchCV()`, decision boundaries.

### 👥 7. K-Nearest Neighbors (KNN) — Questions 37–42

**Objective:** Understand distance-based classification and regression, the choice of neighbors, and feature scaling.

#### Level ML1 — Beginner

- [ ] **Q37. Classify Iris Flowers** — Classify flower species using KNN.
- [ ] **Q38. Predict Customer Purchase Decisions** — Compare purchase predictions using different values of k.
- [ ] **Q39. Predict House Prices Using KNN** — Predict house prices using nearest-neighbor regression.

#### Level ML2 — Intermediate

- [ ] **Q40. Find the Best Value of k** — Compare training and validation performance across different neighbor counts.
- [ ] **Q41. Investigate Distance Metrics and Weighting** — Compare Euclidean and Manhattan distances and weighting strategies.
- [ ] **Q42. Analyze the Impact of Feature Scaling** — Compare KNN performance before and after standardization.

**Concepts:** `KNeighborsClassifier()`, `KNeighborsRegressor()`, `n_neighbors`, `weights`, `metric`, `StandardScaler()`, bias-variance trade-off.

### 🔵 8. K-Means Clustering — Questions 43–48

**Objective:** Explore unsupervised learning, customer segmentation, cluster selection, and clustering evaluation.

#### Level ML1 — Beginner

- [ ] **Q43. Group Customers into Clusters** — Cluster customers by income and spending scores.
- [ ] **Q44. Cluster Flowers Without Labels** — Apply K-Means to Iris features without species labels.
- [ ] **Q45. Explore the Effect of K** — Compare cluster assignments and inertia for different cluster counts.

#### Level ML2 — Intermediate

- [ ] **Q46. Choose the Number of Clusters** — Use the elbow method to select a reasonable cluster count.
- [ ] **Q47. Evaluate Clustering Quality** — Compare clustering configurations using silhouette scores.
- [ ] **Q48. Build a Customer Segmentation Analysis** — Preprocess customer data, select clusters, and profile each segment.

**Concepts:** `KMeans()`, `n_clusters`, `fit_predict()`, centroids, inertia, elbow method, `silhouette_score()`, feature scaling, cluster profiling.

## 📊 Progress Tracker

| Algorithm           | Beginner | Intermediate |    Total |
| ------------------- | -------: | -----------: | -------: |
| Linear Regression   |      1/3 |          0/3 |      1/6 |
| Logistic Regression |      0/3 |          0/3 |      0/6 |
| Decision Trees      |      0/3 |          0/3 |      0/6 |
| Random Forest       |      0/3 |          0/3 |      0/6 |
| XGBoost             |      0/3 |          0/3 |      0/6 |
| SVM                 |      0/3 |          0/3 |      0/6 |
| KNN                 |      0/3 |          0/3 |      0/6 |
| K-Means Clustering  |      0/3 |          0/3 |      0/6 |
| **Total**           | **1/24** |     **0/24** | **1/48** |

## 📁 Project Structure

```text
ml-practice-questions/
├── 01-linear-regression/
│   ├── a_student-exam-scores.py
│   └── b_predict_employee_salaries.py
└── README.md
```

## ⚙️ Installation

Create and activate a virtual environment, then install the required libraries.

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost jupyter
```

## 🧪 Recommended Workflow

For each practice question:

1. Read the problem and identify the input features and target.
2. Prepare the dataset and handle missing or categorical values where necessary.
3. Split supervised-learning data into training and testing sets.
4. Apply preprocessing when appropriate, fitting transformations only on training data.
5. Train the model and generate predictions.
6. Evaluate results using suitable metrics.
7. Experiment with model parameters when required.
8. Record the solution, concepts practiced, and progress in this README.

For K-Means, focus on cluster assignments, inertia, silhouette scores, and the interpretation of each cluster rather than supervised prediction accuracy.

## 📈 Learning Outcomes

After completing these exercises, the goal is to be able to:

- Implement eight common machine learning algorithms in Python.
- Distinguish regression, classification, and clustering tasks.
- Train models and evaluate their performance.
- Understand overfitting, feature scaling, and class imbalance.
- Perform basic hyperparameter tuning and cross-validation.
- Interpret model coefficients, feature importance, decision boundaries, and clusters.
- Build a foundation for more advanced machine learning projects.

## 📝 Progress Log

Use this section to record completed questions and noteworthy improvements.

- **Completed questions:** 1 / 48
- **Beginner questions:** 1 / 24
- **Intermediate questions:** 0 / 24
- **Algorithms completed:** 0 / 8

Update the progress tracker and relevant checkboxes as each solution is completed.

---

**Learning approach:** Focus on understanding the logic behind each algorithm rather than simply memorizing the syntax. Build each solution yourself, compare results, and document what you learn along the way. 🧠
