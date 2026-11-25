# 3. Evaluate a regression model
# Using a dataset of house sizes and prices, train a Linear Regression model and evaluate its
# predictions.
#     Calculate Mean Absolute Error (MAE).
#     Calculate Mean Squared Error (MSE).
#     Calculate the \(R^2\) score.
# Concepts: mean_absolute_error() , mean_squared_error() , r2_score() 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

n_samples = 75
house_sizes = np.random.randint(800, 4001, size=n_samples)

noise = np.random.normal(loc=0, scale=25000, size=n_samples)
house_prices = 50000 + (house_sizes * 180) + noise

house_prices = np.clip(house_prices, 60000, None).round(2)

df = pd.DataFrame({
    'Square_Feet': house_sizes,
    'Price': house_prices
})

X = df[['Square_Feet']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Actual Prices:              {y_test}")
print(f"Predicted Prices:           {np.round(y_pred, 2)}")
print("-" * 40)
print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
print(f"Mean Squared Error (MSE):   {mse:,.2f}")
print(f"Root Mean Squared Error:    ${rmse:,.2f}")
print(f"R-squared (R2) Score:       {r2:.4f}")

