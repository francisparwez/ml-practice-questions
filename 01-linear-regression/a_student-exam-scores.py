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

X = np.array(hours_studied).reshape(-1, 1)

model = LinearRegression()

model.fit(X=X, y=exam_scores)

prediction = model.predict([[7]])
print(f"Predicted exam score for 7 study hours: {prediction[0]:.2f}")
print(f"Coefficient (Slope, m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")