import numpy as np
import pandas as pd
from A9 import predict
from A6 import split_data

def knn(X_train, y_train, X_test, start, end, k=3):
    predictions = []
    for i in range(start, end):
        distances=[]
        key = X_test[i]
        for j in range(len(X_train)):
            dist = np.linalg.norm(X_train[j]-key)
            distances.append((dist,y_train[j]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        classes = {}
        for neighbor in neighbors:
            if neighbor[1] in classes:
                classes[neighbor[1]]+=1
            else:
                classes[neighbor[1]]=1
        predictions.append(max(classes, key=classes.get))
    return predictions

def main():
    data = pd.read_csv("LAB03/train.csv")
    data = data[data.iloc[:, -1].isin(["WALKING", "STANDING"])]
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    predictions = knn(X_train, y_train, X_test, 100, 105, k=3)
    predictions_inbuilt = predict(X_train, y_train, X_test, 100, 105)
    print("Predictions - ",predictions)
    print("Inbuilt Predictions - ",predictions_inbuilt)
    print("Actual - ",y_test[100:105])
    if np.array_equal(predictions, predictions_inbuilt):
        print("Predictions match with inbuilt function")
    else:
        print("Predictions do not match with inbuilt function")

if __name__ == "__main__":
    main()