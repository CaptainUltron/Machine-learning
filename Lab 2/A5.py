import pandas as pd

df = pd.read_excel("Lab Session Data.xlsx")

data = df.loc[0:1, ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]

data_binary = (data > 0).astype(int)

def cal_f(data):
    f00 = 0
    f01 = 0
    f10 = 0
    f11 = 0
    for i in range(data.shape[1]):
        val1 = data.iloc[0,i]
        val2 = data.iloc[1,i]
        if val1==0 and val2==0:
            f00+=1
        elif val1==0 and val2==1:
            f01+=1
        elif val1==1 and val2==0:
            f10+=1
        else:
            f11+=1
    return f00,f01,f10,f11

def cal_jc_smc(f00,f01,f10,f11):
    jc = f11/(f01+f10+f11)
    smc = (f00+f11)/(f00+f01+f10+f11)
    return jc,smc



f00, f01, f10, f11 = cal_f(data_binary)

jc, smc = cal_jc_smc(f00, f01, f10, f11)

print("Binary Data:")
print(data_binary)
print("\nf00 =", f00, "f01 =", f01, "f10 =", f10, "f11 =", f11)
print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)
