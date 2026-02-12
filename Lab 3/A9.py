import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from A6 import split_data
from A7 import KNN

def predict(X_train,y_train, X_test,start, end,k=3):
    neigh = KNN(X_train, y_train, k)
    predictions = neigh.predict(X_test[start:end,:])
    return predictions

def main():
    data = pd.read_csv("LAB03/train.csv")
    data = data[data.iloc[:, -1].isin(["WALKING", "STANDING"])]
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    predictions = predict(X_train, y_train, X_test, 100, 105)
    print("Predictions - ",predictions)
    print("Actual - ",y_test[100:105])

if __name__ == "__main__":
    main()