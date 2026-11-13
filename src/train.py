import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from preprocess import clean

# 1. Загрузка данных
print("Загрузка данных...")
df = pd.read_csv("data/reviews.csv")

# Добавляем данные из фидбека (если файл существует)
if os.path.exists("data/feedback.csv"):
    print("Найдены данные обратной связи! Добавляем к обучению...")
    df_feedback = pd.read_csv("data/feedback.csv")
    # Берем только текст и правильный класс
    df_feedback = df_feedback[["text", "label"]]
    # Объединяем с основным датасетом
    df = pd.concat([df, df_feedback], ignore_index=True)
    # Удаляем дубликаты, если пользователь исправлял один и тот же отзыв
    df = df.drop_duplicates(subset=["text"])

print(f"Итоговый размер: {df.shape}")
print(df["label"].value_counts())

# 2. Векторизация
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1
)
X_vec = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Кросс-валидация (10 разбиений)
print("\nКросс-валидация (10 фолдов)...")
model = ComplementNB(alpha=2.0)
scores = cross_val_score(model, X_vec, y, cv=10, scoring="f1_macro")

print(f"\nF1-macro на каждом фолде:")
for i, s in enumerate(scores, 1):
    print(f"  Фолд {i}: {s:.3f}")
print(f"\n✅ Средний F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")

# 4. Финальное обучение на ВСЕХ данных
print("\nОбучение финальной модели на всех данных...")
model.fit(X_vec, y)

# 5. Сохранение
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/nb_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Модель сохранена в models/")