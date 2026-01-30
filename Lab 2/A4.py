import pandas as pd

def main():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    print(df.info())
    
    print("Missing values: ", df.isnull().sum())
    print("Mean: ", df.mean(numeric_only=True))
    print("Variance: ", df.var(numeric_only=True))

main()

