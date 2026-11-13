import re
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()

def clean(text):
    text = str(text).lower()
    
    # Очистка от мусора
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Лемматизация
    words = text.split()
    lemmas = [_morph.parse(w)[0].normal_form for w in words]

    # Обработка отрицаний (склеиваем "не" со следующим словом)
    result = []
    i = 0
    while i < len(lemmas):
        if lemmas[i] == "не" and i + 1 < len(lemmas):
            # Если после "не" есть слово, склеиваем их
            result.append(f"не_{lemmas[i+1]}")
            i += 2  # Пропускаем два слова
        else:
            result.append(lemmas[i])
            i += 1

    return " ".join(result)