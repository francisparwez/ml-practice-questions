# 2. Predict employee salaries
# Given a dataset containing years of experience and annual salary, train a Linear Regression
# model to estimate salaries for new employees.
#     Split the dataset into training and testing sets.
#     Train the model.
#     Predict salaries for the test data.
# Concepts: train_test_split() , regression, predictions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 50

experience = np.random.uniform(1, 15, n_samples)

salary = 40000 + 8500 * experience + np.random.normal(0, 5000, n_samples)

data = pd.DataFrame({
    'Experience': np.round(experience, 1),
    'Salary': np.round(salary, -2)
})

# 2. Extract features and target variable
X = data[['Experience']]
y = data['Salary']

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict salary for 'n' years of experience
def predict_salary(years_of_experience):
    return model.predict(np.array([[years_of_experience]]))[0]

n_years = 4.5
predicted_salary = predict_salary(n_years)

print(f"Predicted salary for {n_years} years of experience: ${predicted_salary:.2f}")

# 6. Predict salaries for the test data
y_pred = model.predict(X_test)

results = pd.DataFrame({
    'Actual Salary': y_test,
    'Predicted Salary': y_pred
})

print("\nActual vs Predicted Salaries:")
print(results)

# 7. Display model coefficient and intercept
print(f"\nCoefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")