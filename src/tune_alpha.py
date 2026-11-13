import sys, os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from preprocess import clean

# 1. Загрузка и очистка (с лемматизацией)
print("Загрузка и очистка данных...")
df = pd.read_csv("data/reviews.csv")
df["text"] = df["text"].apply(clean)

# 2. Векторизация
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
X_vec = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Перебор alpha
print("\nПодбор alpha:")
print("-" * 40)
alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

for a in alphas:
    model = ComplementNB(alpha=a)
    scores = cross_val_score(model, X_vec, y, cv=10, scoring="f1_macro")
    print(f"alpha = {a:<4} | F1 = {scores.mean():.3f} ± {scores.std():.3f}")