import sys, os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from preprocess import clean

df = pd.read_csv("data/reviews.csv")
df["text"] = df["text"].apply(clean)

# Только positive и negative
df_bin = df[df["label"].isin(["positive", "negative"])]
print(f"Отзывов: {len(df_bin)}")
print(df_bin["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df_bin["text"], df_bin["label"],
    test_size=0.2, stratify=df_bin["label"], random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    stop_words=["и", "в", "во", "на", "с", "со", "к", "ко", "о", "об",
                "по", "за", "из", "от", "до", "у", "при", "про", "для",
                "без", "над", "под", "между", "через", "а", "но", "да",
                "или", "либо", "же", "бы", "ли",
                "я", "ты", "он", "она", "оно", "мы", "вы", "они",
                "мне", "меня", "тебя", "его", "её", "их", "нас", "вас"]
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = ComplementNB(alpha=0.3)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
print(classification_report(y_test, pred))
print(f"F1-macro: {f1_score(y_test, pred, average='macro'):.3f}")
print("\n=== ОШИБКИ ===")
for text, true, pred_i in zip(X_test, y_test, pred):
    if true != pred_i:
        print(f"[{true} → {pred_i}]  {text}")