import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("train.csv")

data = data[data.iloc[:, -1].isin(["WALKING", "STANDING"])]

X = data.iloc[:, :2].to_numpy(dtype=float)
y = data.iloc[:, -1].to_numpy()

y = np.where(y == "WALKING", 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

margin = 0.1
h = 0.01

x_min, x_max = X_train[:, 0].min() - margin, X_train[:, 0].max() + margin
y_min, y_max = X_train[:, 1].min() - margin, X_train[:, 1].max() + margin

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap='bwr',
    edgecolors='k',
    label="Train Data"
)

plt.title(f"kNN Decision Boundary (k={k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
