import csv
import os
import sys

# Добавляем пути к модулям src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from preprocess import clean
from train import train_model

# ==========================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="Анализ тональности отзывов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEEDBACK_FILE = os.path.join(project_root, "data", "feedback.csv")
MODEL_PATH = os.path.join(project_root, "models", "nb_model.pkl")
VEC_PATH = os.path.join(project_root, "models", "vectorizer.pkl")

RU_CLASSES = {
    "positive": "🟢 Позитив",
    "neutral": "🟡 Нейтрал",
    "negative": "🔴 Негатив"
}
CLASS_COLORS = {
    "positive": "#2e7d32",
    "neutral": "#f9a825",
    "negative": "#c62828"
}


# ==========================================
# СОХРАНЕНИЕ ОБРАТНОЙ СВЯЗИ
# ==========================================
def save_feedback(text, correct_label, original_label):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "label", "model_prediction"])
        writer.writerow([text, correct_label, original_label])


# ==========================================
# ЗАГРУЗКА И КЭШИРОВАНИЕ МОДЕЛИ
# ==========================================
@st.cache_resource
def load_sentiment_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        return None, None
    clf = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    return clf, vec


def extract_keywords(clean_text, pred_class, clf, vec, top_n=3):
    """Извлекает ключевые токены, сильнее всего повлиявшие на решение модели."""
    try:
        x_vec = vec.transform([clean_text])
        if x_vec.nnz == 0:
            return []
        class_idx = list(clf.classes_).index(pred_class)
        coo = x_vec.tocoo()
        words = vec.get_feature_names_out()
        weights = []
        for col, val in zip(coo.col, coo.data):
            score = clf.feature_log_prob_[class_idx, col] * val
            weights.append((words[col], score))
        weights.sort(key=lambda t: t[1], reverse=True)
        return [w.replace("не_", "не ") for w, _ in weights[:top_n]]
    except Exception:
        return []


model, vectorizer = load_sentiment_model()

# ==========================================
# БОКОВАЯ ПАНЕЛЬ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ Панель управления")

    st.markdown("### О модели")
    st.markdown(
        "- **Алгоритм:** Complement Naive Bayes\n"
        "- **Признаки:** TF-IDF (1-2 граммы, sublinear TF)\n"
        "- **Предобработка:** Лемматизация (`pymorphy3`), словарь сленга, инверсия отрицаний (`не_слово`)\n"
    )

    feedback_count = 0
    if os.path.exists(FEEDBACK_FILE):
        try:
            fb_df = pd.read_csv(FEEDBACK_FILE)
            feedback_count = len(fb_df)
        except Exception:
            feedback_count = 0

    st.metric("Исправлений в Feedback", feedback_count)

    st.markdown("---")
    st.subheader("🔄 Активное дообучение")
    st.caption("Если вы исправили ошибки модели ниже, нажмите кнопку для моментального переобучения:")

    if st.button("🚀 Переобучить модель сейчас", use_container_width=True, type="secondary"):
        with st.spinner("Переобучение классификатора..."):
            train_model()
            st.cache_resource.clear()
            st.success("✅ Модель успешно переобучена и обновлена в памяти!")
            st.rerun()

    st.markdown("---")
    st.caption("Курсовой / дипломный проект по классификации тональности отзывов.")


# ==========================================
# ГЛАВНЫЙ ЭКРАН
# ==========================================
st.title("📊 Анализ тональности отзывов")
st.markdown(
    "Автоматическая классификация эмоциональной окраски отзывов на базе "
    "**наивного байесовского классификатора (ComplementNB)**."
)

if model is None or vectorizer is None:
    st.error("⚠️ Модель не найдена! Нажмите 'Переобучить модель' в боковом меню слева или запустите `python src/train.py`.")
    st.stop()

# ==========================================
# ВВОД ДАННЫХ
# ==========================================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📁 1. Загрузите файл")
    uploaded = st.file_uploader("Поддерживаются форматы CSV или TXT", type=["csv", "txt"])

with col_b:
    st.subheader("✍️ 2. Или введите текст вручную")
    manual_text = st.text_area(
        "Каждый отзыв — с новой строки:",
        placeholder="Товар супер, доставили вовремя!\nПолный кринж, ничего не работает, вернул деньги.\nОбычная вещь, ожидал большего, но сойдет.",
        height=160
    )

reviews = []
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        try:
            df_up = pd.read_csv(uploaded)
            reviews += df_up.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Ошибка чтения CSV: {e}")
    else:
        content = uploaded.read().decode("utf-8", errors="ignore")
        reviews += [r.strip() for r in content.split("\n") if r.strip()]

if manual_text:
    reviews += [r.strip() for r in manual_text.split("\n") if r.strip()]

reviews = [r for r in reviews if len(r) > 2]

# ==========================================
# КЛАССИФИКАЦИЯ
# ==========================================
if reviews:
    st.info(f"Подготовлено отзывов для анализа: **{len(reviews)}**")

    if st.button("🔍 Выполнить анализ", type="primary", use_container_width=True):
        with st.spinner("Очистка текста и классификация..."):
            cleaned_reviews = [clean(r) for r in reviews]
            X = vectorizer.transform(cleaned_reviews)
            preds = model.predict(X)
            proba = model.predict_proba(X)

            keywords_list = [
                extract_keywords(c, p, model, vectorizer)
                for c, p in zip(cleaned_reviews, preds)
            ]

            st.session_state["result_df"] = pd.DataFrame({
                "Отзыв": reviews,
                "Очищенный текст": cleaned_reviews,
                "Класс": preds,
                "Уверенность": [f"{max(p):.0%}" for p in proba],
                "Ключевые слова": keywords_list
            })

# ==========================================
# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
# ==========================================
if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]

    # 1. СТАТИСТИКА
    st.markdown("---")
    st.subheader("📈 Итоговая статистика")
    counts = result_df["Класс"].value_counts()

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Позитив", counts.get("positive", 0))
    c2.metric("🟡 Нейтрал", counts.get("neutral", 0))
    c3.metric("🔴 Негатив", counts.get("negative", 0))

    # 2. ДИАГРАММЫ
    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig1, ax1 = plt.subplots(figsize=(4, 3.5))
        pie_labels = [RU_CLASSES[k] for k in ["positive", "neutral", "negative"] if k in counts.index]
        pie_sizes = [counts[k] for k in ["positive", "neutral", "negative"] if k in counts.index]
        colors = [CLASS_COLORS[k] for k in ["positive", "neutral", "negative"] if k in counts.index]

        if pie_sizes:
            ax1.pie(pie_sizes, labels=pie_labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

    with col_bar:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        bar_x = ["Позитив", "Нейтрал", "Негатив"]
        bar_y = [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
        ax2.bar(bar_x, bar_y, color=["#2e7d32", "#f9a825", "#c62828"])
        ax2.set_ylabel("Количество")
        st.pyplot(fig2)

    # 3. КАРТОЧКИ С ОТЗЫВАМИ И КЛЮЧЕВЫМИ СЛОВАМИ
    st.markdown("---")
    st.subheader("📋 Распределение по категориям и ключевые слова")
    col_pos, col_neu, col_neg = st.columns(3)

    def render_column(col, label_key, title, color_hex):
        with col:
            subset = result_df[result_df["Класс"] == label_key]
            st.markdown(
                f"<h3 style='color:{color_hex}; border-bottom: 2px solid {color_hex}; padding-bottom:4px;'>"
                f"{title} ({len(subset)})</h3>",
                unsafe_allow_html=True
            )

            if subset.empty:
                st.caption("— отзывов этой категории нет —")
            else:
                with st.container(height=520):
                    for idx, row in subset.iterrows():
                        kw_badges = ""
                        if row["Ключевые слова"]:
                            badges = " ".join([
                                f"<span style='background:#e0e0e0; color:#333; padding:2px 6px; "
                                f"border-radius:10px; font-size:11px; margin-right:4px;'>{w}</span>"
                                for w in row["Ключевые слова"]
                            ])
                            kw_badges = f"<div style='margin-top:4px;'><b>Триггеры:</b> {badges}</div>"

                        st.markdown(
                            f"<div style='padding:10px; margin-bottom:8px; background:#ffffff; "
                            f"border: 1px solid #e0e0e0; border-left: 5px solid {color_hex}; "
                            f"border-radius:6px; font-size:14px; color:#212121; line-height:1.4;'>"
                            f"{row['Отзыв']}<br>"
                            f"<div style='margin-top:6px; font-size:12px; color:#616161;'>"
                            f"Уверенность модели: <b>{row['Уверенность']}</b></div>"
                            f"{kw_badges}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        with st.expander("❌ Модель ошиблась?"):
                            correct = st.selectbox(
                                "Укажите верный класс:",
                                ["positive", "neutral", "negative"],
                                index=["positive", "neutral", "negative"].index(label_key),
                                format_func=lambda x: RU_CLASSES[x],
                                key=f"sel_{idx}"
                            )
                            if st.button("Сохранить исправление", key=f"fb_btn_{idx}"):
                                save_feedback(row["Отзыв"], correct, row["Класс"])
                                st.success("Спасибо! Исправление сохранено в feedback.csv.")

    render_column(col_pos, "positive", "🟢 Позитив", CLASS_COLORS["positive"])
    render_column(col_neu, "neutral", "🟡 Нейтрал", CLASS_COLORS["neutral"])
    render_column(col_neg, "negative", "🔴 Негатив", CLASS_COLORS["negative"])

    # 4. СКАЧИВАНИЕ РЕЗУЛЬТАТОВ
    st.markdown("---")
    export_df = result_df[["Отзыв", "Класс", "Уверенность"]].copy()
    export_df["Класс_RU"] = export_df["Класс"].map(lambda x: RU_CLASSES.get(x, x))
    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Скачать результаты разметки (CSV)",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
        use_container_width=True
    )
