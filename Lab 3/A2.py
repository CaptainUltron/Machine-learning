import numpy as np
import pandas as pd
import math

def Mean(data):
    res=[]
    for i in range(len(data.columns)-1):
        res.append(sum(data.iloc[:,i])/len(data))
    return res

def var(data):
    m=Mean(data)
    var=0
    for i in data:
        var+=((i-m)**2)
    var=var/len(data)
    return var

def stddev(data):
    v=var(data)
    return math.sqrt(v)

def main():
    data=pd.read_csv("LAB03/train.csv")
    class1=data[data.iloc[:,-1]=="WALKING"]
    class1mean=np.mean(class1.iloc[:,0:-1].to_numpy(dtype=float),axis=0)
    print("Mean of class 1 - ",class1mean)
    class1std=np.std(class1.iloc[:,0:-1].to_numpy(dtype=float),axis=0)
    print("Standard Deviation of class 1 - ",class1std)
    class2=data[data.iloc[:,-1]=="STANDING"]
    class2mean=np.mean(class2.iloc[:,0:-1].to_numpy(dtype=float),axis=0)
    print("Mean of class 2 - ",class2mean)
    class2std=np.std(class2.iloc[:,0:-1].to_numpy(dtype=float),axis=0)
    print("Standard Deviation of class 2 - ",class2std)
    distance=np.linalg.norm(class1mean-class2mean)
    print("Distance between the two classes - ",distance)

if __name__ == "__main__":
    main()