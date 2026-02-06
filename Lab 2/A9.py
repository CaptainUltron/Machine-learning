import pandas as pd
import numpy as np

def Z_score_normalize(column):
    mean = np.mean(column)
    std = np.std(column)
    normalized_column = (column - mean) / std
    return normalized_column

data = pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')

for i in data.index:
    v = str(data.loc[i, 'Volume'])
    if 'M' in v:
        data.loc[i, 'Volume'] = float(v[:-1]) * 1_000_000
    elif 'K' in v:
        data.loc[i, 'Volume'] = float(v[:-1]) * 1_000
    else:
        data.loc[i, 'Volume'] = float(v)

    chg = str(data.loc[i,'Chg%'])
    data.loc[i,'Chg%']=float(chg[:-1])

# normalization
data['Price']=Z_score_normalize(data['Price'])
data['Open']=Z_score_normalize(data['Open'])
data['High']=Z_score_normalize(data['High'])
data['Low']=Z_score_normalize(data['Low'])
data['Volume']=Z_score_normalize(data['Volume'])
data['Chg%']=Z_score_normalize(data['Chg%'])
print(data)

