import numpy as np
import pandas as pd

def dotProduct(A,B):
    dp=0
    for i in range(len(B)):
        dp+=(A[i]*B[i])
    return dp

def EuclideanNorm(A):
    n=0
    for i in A:
        n+=i**2
    n=n**(1/2)
    return n

def main():
    data=pd.read_csv("LAB03/train.csv")
    A=data.iloc[:,0]
    B=data.iloc[:,1]
    a=round(dotProduct(A,B),6)
    ae=round(EuclideanNorm(A),6)
    be=round(EuclideanNorm(B),6)
    dpi=round(np.dot(A,B),6)
    Euai=round(np.linalg.norm(A),6)
    Eubi=round(np.linalg.norm(B),6)

    if a==dpi:
        print("Dot Product Same - ", a,dpi)

    if ae==Euai:
        print("Euclidean Norm of A same - ",ae,Euai)

    if be==Eubi:
        print("Euclidean Norm of B same - ",be,Eubi)


if __name__ == "__main__":
    main()
