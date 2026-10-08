import pandas as pd

df = pd.read_csv('pasteurizer.csv')
df['STD_DT'] = pd.to_datetime(df['STD_DT'])

print("크기:", df.shape)
print("\n기간:", df['STD_DT'].min(), "~", df['STD_DT'].max())
print("\n결측치:")
print(df.isna().sum())
print("\n첫 5행:")
print(df.head())
print("\n마지막 5행:")
print(df.tail())