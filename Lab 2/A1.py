import numpy as np
import pandas as pd
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values
rank = np.linalg.matrix_rank(X)
print(rank)

a = np.linalg.pinv(X) @ y
print(a)