# 4. Predict sales using advertising expenditure
# Create a dataset containing TV, radio, and online advertising expenditure alongside product
# sales.
#     Train a multiple Linear Regression model.
#     Predict sales using all three advertising features.
#     Identify which features have positive or negative coefficients.
# Concepts:
#         Multiple Linear Regression, multiple features, feature coefficients


import numpy as np
from sklearn.linear_model import LinearRegression

# Real-world Advertising dataset subset (in thousands)
tv = [
    230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
    66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3,
    218.4, 237.4, 13.2, 228.3, 62.3, 262.9, 142.9, 240.1, 248.8, 70.6
]

radio = [
    37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6,
    5.8, 24.0, 35.1, 7.6, 32.9, 47.7, 36.6, 39.6, 20.5, 23.9,
    27.7, 5.1, 15.9, 16.9, 12.6, 3.5, 29.3, 16.7, 27.1, 16.0
]

online = [
    69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2,
    24.2, 4.0, 65.9, 7.2, 46.0, 52.9, 114.0, 55.8, 18.3, 19.1,
    53.4, 23.5, 49.6, 26.2, 18.3, 19.5, 12.6, 22.9, 22.9, 40.8
]

sales = [
    22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6,
    8.6, 17.4, 9.2, 9.7, 19.0, 22.4, 12.5, 24.4, 11.3, 14.6,
    18.0, 12.5, 5.6, 15.5, 9.7, 12.0, 15.0, 15.9, 18.9, 10.5
]

# Combine lists into a 2D feature matrix (X) and target array (y)
X = np.column_stack((tv, radio, online))
y = np.array(sales)

# Fit the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Display model coefficients and R-squared score
print("Intercept:", round(model.intercept_, 4))
print("Coefficients (TV, Radio, Online):", np.round(model.coef_, 4))
print("R-squared Score:", round(model.score(X, y), 4))

# Predict sales using all three advertising features
predictions = model.predict(X)

print("\nActual vs Predicted Sales:")

for actual, predicted in zip(y, predictions):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")
    
# Display individual feature coefficients
features = ['TV', 'Radio', 'Online']

print("\nFeature Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    if coefficient > 0:
        direction = "Positive"
    elif coefficient < 0:
        direction = "Negative"
    else:
        direction = "Zero"
    print(f"{feature}: {coefficient:.4f} ({direction})")