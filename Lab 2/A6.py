import pandas as pd
import numpy as np


df = pd.read_excel("Lab Session Data.xlsx")

A = df.loc[0, ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].values
B = df.loc[1, ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].values

dot_product = np.dot(A, B)

norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)

cosine_similarity = dot_product / (norm_A * norm_B)

print("Vector A:", A)
print("Vector B:", B)
print("Dot Product <A, B>:", dot_product)
print("||A||:", norm_A)
print("||B||:", norm_B)
print("Cosine Similarity:", cosine_similarity)
