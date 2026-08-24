import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(URL)

positive_row = df[df["status"] == 1].iloc[0]
negative_row = df[df["status"] == 0].iloc[0]

feature_columns = [c for c in df.columns if c not in ("name", "status")]

import json
print("--- one real POSITIVE (parkinsons) row, as JSON ---")
print(json.dumps({c: float(positive_row[c]) for c in feature_columns}, indent=2))
print()
print("--- one real NEGATIVE (healthy) row, as JSON ---")
print(json.dumps({c: float(negative_row[c]) for c in feature_columns}, indent=2))
