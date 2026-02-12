from A10 import knn
from A6 import split_data
from A9 import predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("LAB03/train.csv")
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    predictions_k1 = knn(X_train, y_train, X_test,100,105, k=11)
    predictions_k3 = knn(X_train, y_train, X_test,100,105, k=3)
    pred=predict(X_train, y_train, X_test,100,105,k=1)
    print("Predictions with k=1 - ",predictions_k1)
    print("Predictions with k=3 - ",predictions_k3)
    print("Inbuilt Predictions - ",pred)
    print("Actual - ",y_test[100:105])
    # Accuracy plot for k = 1 to 11
    accuracies = []
    for k in range(1, 12):
        pred = predict(X_train, y_train, X_test, 0, len(X_test), k=k)
        accuracy = np.sum(pred == y_test) / len(y_test)
        accuracies.append(accuracy)
    plt.figure()
    plt.plot(range(1, 12), accuracies, marker='o')
    plt.title('KNN Accuracy for Different k Values')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
