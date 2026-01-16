import pandas as pd

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1",
    header=None
)

print(df.head(10))
print("\nUnique sentiment values:")
print(df[0].value_counts())
