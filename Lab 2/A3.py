import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
m1 = np.mean(df["Price"])
v1 = np.var(df["Price"])
print(m1, v1)

print("=="*60)

def mean(data):
    sum = 0
    for i in data["Price"]:
        sum += i
        mean = sum/len(data)
    return mean

def variance(data):
    m = mean(data)
    var_sum = 0
    for i in data["Price"]:
        var_sum += (i - m)**2
        variance = var_sum/len(data)
    return variance


m2 = mean(df)
v2 = variance(df)
print(m2, v2)

print("=="*60)

"""for i in df["Day"]:
    print(i)"""
