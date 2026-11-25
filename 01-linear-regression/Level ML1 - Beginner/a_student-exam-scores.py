# 1. Predict student exam scores
# Given:
#     hours_studied = [1, 2, 3, 4, 5, 6]
#     exam_scores = [35, 40, 50, 55, 65, 75]
# Train a Linear Regression model to predict exam scores from study hours.
#     Fit the model.
#     Predict the score for 7 study hours.
#     Display the model's coefficient and intercept.
# Concepts: LinearRegression() , .fit() , .predict() , coefficients.

import numpy as np
from sklearn.linear_model import LinearRegression

hours_studied = [1, 2, 3, 4, 5, 6]
exam_scores = [35, 40, 50, 55, 65, 75]

# Reshape the data
X = np.array(hours_studied).reshape(-1, 1)
y = np.array(exam_scores)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict the score for 7 study hours
predicted_score = model.predict(np.array([[7]]))
print(f"Predicted Exam Score For 7 Study Hours: {predicted_score[0].round(2)}")

# Display the model's coefficient and intercept
print(f"Model Coefficient: {model.coef_[0].round(2)}")
print(f"Model Intercept: {model.intercept_.round(2)}")