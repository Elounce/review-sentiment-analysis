import os
import re
import pandas as pd
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_slang_dict = None


def get_slang_dict():
    """Загружает и кэширует словарь интернет-сленга из data/slang_dict.csv"""
    global _slang_dict
    if _slang_dict is not None:
        return _slang_dict

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "slang_dict.csv"),
        "data/slang_dict.csv",
        "../data/slang_dict.csv"
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                d = {}
                for _, row in df.iterrows():
                    d[str(row["slang"]).strip().lower()] = (
                        str(row["нормализация"]).strip(),
                        str(row.get("тип", "polarity")).strip()
                    )
                _slang_dict = d
                return _slang_dict
            except Exception:
                pass
    _slang_dict = {}
    return _slang_dict


def normalize_slang(text):
    """Заменяет сленговые выражения на их литературные/эмоциональные аналоги."""
    slang = get_slang_dict()
    if not slang:
        return text

    # Шаг 1: Замена устойчивых сленговых фраз (от длинных к коротким)
    phrases = {k: v for k, v in slang.items() if v[1] == "phrase"}
    for phrase, (norm, _) in sorted(phrases.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(phrase, norm)

    # Шаг 2: Замена эмоциональных сленговых слов
    polarities = {k: v for k, v in slang.items() if v[1] == "polarity"}
    for word, (norm, _) in polarities.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", norm, text)

    # Шаг 3: Замена усилителей ("жесть" -> "очень")
    intensifiers = {k: v for k, v in slang.items() if v[1] == "intensifier"}
    for word, (norm, _) in intensifiers.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", "очень", text)

    return text


def clean(text):
    """
    Полный цикл очистки русскоязычного текста:
    1. Приведение к нижнему регистру
    2. Нормализация молодежного и интернет-сленга
    3. Удаление ссылок, HTML, email, спецсимволов и цифр
    4. Лемматизация (приведение слов к начальной форме через pymorphy3)
    5. Обработка отрицаний (склейка 'не', 'нет', 'ни' со следующим словом)
    """
    text = str(text).lower()

    # Нормализуем сленг
    text = normalize_slang(text)

    # Очистка от технического мусора
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Лемматизация
    words = text.split()
    lemmas = [_morph.parse(w)[0].normal_form for w in words]

    # Обработка отрицаний (склеиваем "не", "нет", "ни" со следующим словом)
    negations = {"не", "нет", "ни"}
    result = []
    i = 0
    while i < len(lemmas):
        if lemmas[i] in negations and i + 1 < len(lemmas):
            result.append(f"не_{lemmas[i+1]}")
            i += 2
        else:
            result.append(lemmas[i])
            i += 1

    return " ".join(result)
