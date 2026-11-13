import os
import sys

# Добавляем путь к папке src
sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd
from preprocess import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB


def load_all_data():
    """Загружает и объединяет все доступные датасеты и пользовательский фидбек."""
    dfs = []

    # 1. Основной датасет
    if os.path.exists("data/reviews.csv"):
        d1 = pd.read_csv("data/reviews.csv")[["text", "label"]]
        dfs.append(d1)
        print(f"Загружен data/reviews.csv: {len(d1)} записей")

    # 2. Дополнительный датасет
    if os.path.exists("data/Dataset.csv"):
        d2 = pd.read_csv("data/Dataset.csv")[["text", "label"]]
        dfs.append(d2)
        print(f"Загружен data/Dataset.csv: {len(d2)} записей")

    # 3. Базовые паттерны тональности (отрицания, брак, рекомендации)
    if os.path.exists("data/extra_sentiment.csv"):
        d_extra = pd.read_csv("data/extra_sentiment.csv")[["text", "label"]]
        dfs.append(d_extra)
        print(f"Загружен data/extra_sentiment.csv: {len(d_extra)} записей")

    # 4. Пользовательские исправления (Feedback)
    if os.path.exists("data/feedback.csv"):
        d3 = pd.read_csv("data/feedback.csv")[["text", "label"]]
        dfs.append(d3)
        print(f"Загружен data/feedback.csv: {len(d3)} записей")

    if not dfs:
        raise FileNotFoundError("Не найден ни один файл датасета в папке data/!")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["text", "label"])
    full_df["text"] = full_df["text"].astype(str)
    full_df = full_df[full_df["text"].str.strip().str.len() > 2]
    full_df = full_df.drop_duplicates(subset=["text"])

    return full_df


def train_model():
    print("=" * 50)
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ НАИВНОГО БАЙЕСА (ComplementNB)")
    print("=" * 50)

    # 1. Загрузка данных
    df = load_all_data()
    print(f"\n📊 Итоговый объединенный датасет: {len(df)} уникальных отзывов")
    print("Распределение классов:")
    print(df["label"].value_counts())

    # 2. Предобработка и очистка
    print("\n🧹 Выполняется предобработка (очистка, лемматизация, нормализация сленга и отрицаний)...")
    df["clean_text"] = df["text"].apply(clean)

    # Убираем строки, ставшие пустыми после очистки
    df = df[df["clean_text"].str.strip().str.len() > 0]

    # 3. TF-IDF Векторизация
    print("\n🔤 Векторизация TF-IDF (униграммы + биграммы, sublinear_tf)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_vec = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # 4. Кросс-валидация
    print("\n📈 10-фолдовая кросс-валидация (F1-macro)...")
    model = ComplementNB(alpha=0.5)
    scores = cross_val_score(model, X_vec, y, cv=10, scoring="f1_macro")

    print(f"F1-macro по фолдам: {[round(s, 3) for s in scores]}")
    print(f"✅ Средний F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

    # 5. Финальное обучение на всех данных
    print("\n🎯 Финальное обучение классификатора...")
    model.fit(X_vec, y)

    # 6. Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/nb_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("💾 Модели сохранены в models/nb_model.pkl и models/vectorizer.pkl")

    # 7. Контрольные тесты
    print("\n🧪 Контрольные тесты работы классификатора:")
    test_phrases = [
        "Товар супер, всё доставили быстро и качественно!",
        "Полный кринж, а не товар, деньги на ветер.",
        "В целом пойдет, работает обычно, без восторгов.",
        "Не понравилось качество, не рекомендую покупать.",
        "Брак пришел, не включается вообще.",
        "Обычный китайский провод, заряжает нормально."
    ]
    for phrase in test_phrases:
        c_phrase = clean(phrase)
        vec_phrase = vectorizer.transform([c_phrase])
        pred = model.predict(vec_phrase)[0]
        proba = max(model.predict_proba(vec_phrase)[0])
        print(f"  [{pred.upper():<8}] ({proba:.0%}) -> '{phrase}'")

    print("\n🎉 Обучение успешно завершено!")
    return model, vectorizer


if __name__ == "__main__":
    train_model()
