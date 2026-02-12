import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from A6 import split_data

def KNN(X_train, y_train, k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh

def main():
    data=pd.read_csv("LAB03/train.csv")
    data=data[data.iloc[:,-1].isin(["WALKING","STANDING"])]
    X=data.iloc[:,:-1].to_numpy(dtype=float)
    y=data.iloc[:,-1].to_numpy()
    X_train,X_test,y_train,y_test = split_data(X,y,test_size=0.3)
    neigh = KNN(X_train, y_train, k=3)

if __name__ == "__main__":
    main()