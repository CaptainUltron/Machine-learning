import pandas as pd

df = pd.read_excel("Lab Session Data.xlsx")

def classify_cust(x):
    ans = []
    for i in x:
        if i > 200:
            ans.append("RICH")
        else:
            ans.append("POOR")
    return ans

df["Customer_Class"] = classify_cust(df["Payment (Rs)"])
print(df[["Customer", "Payment (Rs)", "Customer_Class"]])
