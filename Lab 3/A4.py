import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

numeric_data = data.select_dtypes(include=[np.number])

x = numeric_data.iloc[0].values
y = numeric_data.iloc[1].values

def minkowski_distance(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

p_values = range(1, 11)
distances = [minkowski_distance(x, y, p) for p in p_values]

plt.figure()
plt.plot(p_values, distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.show()
