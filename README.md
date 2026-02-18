# hh_it_level_classifier (PoC)

Proof of Concept: выделяем резюме IT-разработчиков из hh.csv, автоматически размечаем уровень (junior/middle/senior) и обучаем классификатор.  
Артефакты: график баланса классов и classification_report.

## 1) Setup (macOS)
```bash
cd hh_it_level_classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# HH Level Classifier 
код лежит в https://github.com/Borshick02/hh_Classifier.git
## Что делает проект

1) **Фильтрует** резюме IT-разработчиков из `hh.csv`  
2) **Размечает уровень (label)** по правилам (ключевые слова в названии + fallback по опыту)  
3) **Строит график баланса классов**  
4) **Обучает модель классификации** (baseline: LogisticRegression + TF-IDF + OneHot + числовые признаки)  
5) **Сохраняет артефакты**:
   - `reports/class_balance.png`
   - `reports/classification_report.txt`
   - `resources/model.joblib`
   - `resources/meta.json`
Устнавка
 ```bash
cd hh_it_level_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Вывол результатов
```bash
python app ../hh.csv --mode prepare
python app ../hh.csv --mode train
python app ../hh.csv --mode evaluate
```
Как устроена разметка (labels)

Разметка уровня делается в src/hh_it_level_classifier/labels.py:

Сначала проверяем ключевые слова в названии должности (token-based; phrase: team lead/tech lead -> senior)

Если ключевых слов нет — fallback по опыту:

< 2 лет → junior

2–6 лет → middle

>= 6 лет → senior

Это PoC-разметка: она не идеальна, но показывает, что идея работает.
Признаки (features)

Используются:

Числовые: age, salary_rub, exp_years

Категориальные: city, employment, schedule

Текст: skills_text → TF-IDF (uni/bi-grams)

Модель: LogisticRegression в Pipeline:

numeric: imputer + scaler

categorical: imputer + onehot

text: selector + TF-IDF

classifier: LogisticRegression

Middle хуже размечается: в названиях резюме часто нет слова “middle”, поэтому метка часто ставится по опыту → появляется шум.

Признаки пересекаются: по зарплате/навыкам/опыту мидлы похожи и на джунов, и на сеньоров → модель чаще “уверенно” относит к соседним классам.
