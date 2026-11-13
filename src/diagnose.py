import sys, os
sys.path.append(os.path.dirname(__file__))
import pandas as pd
from preprocess import clean

df = pd.read_csv("data/reviews.csv").head(6)

for i, row in df.iterrows():
    print(f"--- Оригинал ({row['label']}):")
    print(f"    {row['text']}")
    print(f"--- После clean():")
    print(f"    {clean(row['text'])}")
    print()