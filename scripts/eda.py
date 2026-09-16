import pandas as pd

df = pd.read_csv('twcs/twcs.csv', nrows=500000)
print("Total rows:", len(df))
print(df.head())

# Find top brands
brands = df[df['inbound'] == False]
top_brands = brands['author_id'].value_counts().head(20)
print("\nTop Brands:")
print(top_brands)
