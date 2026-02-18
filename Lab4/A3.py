import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

data = []
for i in range(len(X)):
    data.append([X[i], Y[i]])

classes = []
for i in range(len(X)):
    if X[i] + Y[i] > 11:
        classes.append(1)
    else:
        classes.append(0)

plt.figure(figsize=(6, 6))

for i in range(len(data)):
    if classes[i] == 0:
        plt.scatter(X[i], Y[i], color='blue')
    else:
        plt.scatter(X[i], Y[i], color='red')

plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data Scatter Plot (Class0 = Blue, Class1 = Red)")
plt.grid(True)

plt.show()