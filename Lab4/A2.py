import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase_data")

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
y = df[["Payment (Rs)"]].to_numpy()

costs = np.linalg.pinv(X) @ y
print("Costs (coefficients):\n", costs)

y_pred = np.matmul(X,costs)
print(y_pred)

mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2 Score:", r2)
