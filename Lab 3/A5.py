import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

def minkowski_distance(vector1, vector2, p=2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    distance = np.sum(np.abs(vector1 - vector2) ** p) ** (1/p)
    return distance

def main():
    data = pd.read_csv("LAB03/train.csv")
    vector_a = data.iloc[0, :-1].to_numpy(dtype=float)
    vector_b = data.iloc[1, :-1].to_numpy(dtype=float)
    p=[1,2,3,4]
    for p in p:
        dist = minkowski_distance(vector_a, vector_b, p=p)
        inbuiltDist = minkowski(vector_a, vector_b, p=p)
        print(f"p = {p}:")
        print(f"  Own Function:  {dist}")
        print(f"  Inbuilt Function:   {inbuiltDist}")

if __name__ == "__main__":
    main()