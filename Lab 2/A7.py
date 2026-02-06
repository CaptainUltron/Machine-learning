import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')

X = data.select_dtypes(include='number').iloc[:20]
X_bin = (X > X.mean()).astype(int)

def jc(a, b):
    M11 = np.sum((a == 1) & (b == 1))
    M10 = np.sum((a == 1) & (b == 0))
    M01 = np.sum((a == 0) & (b == 1))
    denom = M11 + M10 + M01
    return M11 / denom if denom != 0 else 0

def smc(a, b):
    return np.mean(a == b)

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

n = 20
JC = np.zeros((n, n))
SMC = np.zeros((n, n))
COS = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        JC[i, j] = jc(X_bin.iloc[i], X_bin.iloc[j])
        SMC[i, j] = smc(X_bin.iloc[i], X_bin.iloc[j])
        COS[i, j] = cos(X.iloc[i], X.iloc[j])

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(JC, annot=True)
plt.title('JC')

plt.subplot(1, 3, 2)
sns.heatmap(SMC, annot=True)
plt.title('SMC')

plt.subplot(1, 3, 3)
sns.heatmap(COS, annot=True)
plt.title('COS')

plt.show()

