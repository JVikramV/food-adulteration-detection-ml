import pandas as pd

df = pd.read_csv("data/milk quality.csv")

print("\nColumns count =", len(df.columns))
print(df.columns)
