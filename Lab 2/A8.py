import pandas as pd

def main():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    for column in data.columns:
        if data[column].dtype != "object":
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)

    print("Missing values handled.")

if __name__ == "__main__":
    main()
