import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data=pd.read_csv("LAB03/train.csv")
    data=data[data.iloc[:,-1].isin(["WALKING","STANDING"])]
    X=data.iloc[:,:-1].to_numpy(dtype=float)
    y=data.iloc[:,-1].to_numpy()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    print("Training data - ",X_train)
    print("Training labels - ",y_train)
    print("Testing data - ",X_test)
    print("Testing labels - ",y_test)

if __name__ == "__main__":
    main()