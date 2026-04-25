import pandas as pd

df = pd.read_excel("ev_tech_cleaned_dataset.xls", engine="xlrd")
print("\nDATA: \n",df.head(5))

print("\nInfo: \n")
print(df.info())
print("\nData Description: \n", df.describe())

print("\n Is Null Values Present\n",df.isnull().sum())

df['Parent_ID'] = df['Parent_ID'].fillna("ROOT")

print("\n Is Null Values Present\n",df.isnull().sum())

df.to_csv("new_ev_tech_cleaned_dataset.csv", index=False)