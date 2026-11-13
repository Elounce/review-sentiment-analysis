import sys, os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, C=1.0)
from sklearn.model_selection import cross_val_score
from preprocess import clean

df = pd.read_csv("data/reviews.csv")
df["text"] = df["text"].apply(clean)
df = df[df["label"].isin(["positive", "negative"])]

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_vec = vectorizer.fit_transform(X)



# 10 раз делим на train/test и считаем F1
scores = cross_val_score(model, X_vec, y, cv=10, scoring="f1_macro")
print("F1 на 10 разных разбиениях:")
for i, s in enumerate(scores, 1):
    print(f"  Разбиение {i}: {s:.3f}")
print(f"\nСредний F1: {scores.mean():.3f}")
print(f"Разброс: {scores.std():.3f}")