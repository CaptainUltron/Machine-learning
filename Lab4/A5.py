import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

data = np.column_stack((X, Y))

classes = []
for i in range(len(X)):
    if X[i] + Y[i] > 11:
        classes.append(1)
    else:
        classes.append(0)

k=10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(data, classes)

x = np.arange(0, 10.1, 0.1)
y = np.arange(0, 10.1, 0.1)

grid = np.array([(i, j) for i in x for j in y])

Z = knn.predict(grid)
Z = Z.reshape(len(x), len(y)).T

plt.figure(figsize=(8,6))
plt.contourf(x, y, Z, alpha=0.3)

plt.scatter(data[:,0], data[:,1], c=classes, edgecolors='k')

plt.title(f"kNN Decision Boundary (k={k})")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
