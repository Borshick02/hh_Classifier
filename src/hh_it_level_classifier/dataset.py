from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from hh_it_level_classifier.labels import infer_level
from hh_it_level_classifier.utils import normalize_text, try_read_csv_encoding


IT_KEYWORDS = [
    "разработчик",
    "developer",
    "программист",
    "software",
    "backend",
    "frontend",
    "fullstack",
    "full stack",
    "python",
    "java",
    "javascript",
    "js ",
    "golang",
    "go ",
    "c#",
    "c++",
    "ios",
    "android",
    "qa",
    "тестировщик",
    "devops",
    "data engineer",
    "ml",
    "machine learning",
]


@dataclass(frozen=True, slots=True)
class PreparedRow:
    age: float | None
    salary_rub: float | None
    exp_years: float | None
    city: str | None
    position_text: str
    employment: str | None
    schedule: str | None
    skills_text: str
    label: str
    label_reason: str


def _pick_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_gender_age(s: str | None) -> tuple[str | None, float | None]:
    if not s:
        return None, None
    t = normalize_text(s)

    gender: str | None = None
    if "мужчина" in t:
        gender = "male"
    elif "женщина" in t:
        gender = "female"

    m = re.search(r"(\d+)\s*год", t)
    age = float(m.group(1)) if m else None
    return gender, age


def _parse_salary_rub(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s == "" or s.lower() == "nan":
        return None

    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    return float("".join(digits))


def _parse_experience_years(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s == "" or s.lower() == "nan":
        return None

    t = normalize_text(s)

    years = 0
    months = 0

    my = re.search(r"(\d+)\s*лет", t)
    if my:
        years = int(my.group(1))

    mm = re.search(r"(\d+)\s*месяц", t)
    if mm:
        months = int(mm.group(1))

    if years == 0 and months == 0:
        return None
    return float(years) + float(months) / 12.0


def _is_it_resume(position_text: str) -> bool:
    t = normalize_text(position_text)
    return any(k in t for k in IT_KEYWORDS)


@dataclass(frozen=True, slots=True)
class _DetectedCols:
    col_gender_age: str | None
    col_salary: str | None
    col_city: str | None
    col_employment: str | None
    col_schedule: str | None
    col_exp: str | None
    col_pos_want: str | None
    col_pos_last: str | None
    col_skills: str | None


def _detect_columns(columns: list[str], target_column_hint: str) -> _DetectedCols:
    return _DetectedCols(
        col_gender_age=_pick_first_existing(columns, ["Пол, возраст", "Пол, Возраст", "Пол, возраст "]),
        col_salary=_pick_first_existing(columns, [target_column_hint, "ЗП", "Зарплата", "salary"]),
        col_city=_pick_first_existing(columns, ["Город", "Город ", "city"]),
        col_employment=_pick_first_existing(columns, ["Занятость", "employment"]),
        col_schedule=_pick_first_existing(columns, ["График", "schedule"]),
        col_exp=_pick_first_existing(columns, ["Опыт (двойное нажатие для полной версии)", "Опыт работы", "Опыт", "experience"]),
        col_pos_want=_pick_first_existing(columns, ["Ищет работу на должность:", "Ищет работу на должность", "Должность", "position"]),
        col_pos_last=_pick_first_existing(columns, ["Последеняя/нынешняя должность", "Последняя/нынешняя должность", "Последняя должность"]),
        col_skills=_pick_first_existing(columns, ["Ключевые навыки", "Навыки", "skills"]),
    )


def _iter_csv_rows(
    csv_path: Path,
    encoding: str,
    cols: _DetectedCols,
    chunksize: int,
    limit_rows: int | None,
) -> Iterator[dict]:
    usecols = [
        c for c in [
            cols.col_gender_age,
            cols.col_salary,
            cols.col_city,
            cols.col_employment,
            cols.col_schedule,
            cols.col_exp,
            cols.col_pos_want,
            cols.col_pos_last,
            cols.col_skills,
        ]
        if c
    ]
    if not usecols:
        raise ValueError("Could not detect required columns in hh.csv")

    seen = 0
    for chunk in pd.read_csv(csv_path, encoding=encoding, chunksize=chunksize, usecols=usecols):
        for _, row in chunk.iterrows():
            if limit_rows is not None and seen >= limit_rows:
                return
            seen += 1
            yield row.to_dict()


def _row_to_prepared(row: dict, cols: _DetectedCols) -> PreparedRow | None:
    _, age = _parse_gender_age(row.get(cols.col_gender_age) if cols.col_gender_age else None)

    salary = _parse_salary_rub(row.get(cols.col_salary) if cols.col_salary else None)
    exp_years = _parse_experience_years(row.get(cols.col_exp) if cols.col_exp else None)

    pos1 = str(row.get(cols.col_pos_want)).strip() if cols.col_pos_want and row.get(cols.col_pos_want) is not None else ""
    pos2 = str(row.get(cols.col_pos_last)).strip() if cols.col_pos_last and row.get(cols.col_pos_last) is not None else ""
    position_text = (pos1 + " " + pos2).strip()
    if not position_text:
        return None
    if not _is_it_resume(position_text):
        return None

    city = str(row.get(cols.col_city)).strip() if cols.col_city and row.get(cols.col_city) is not None else None
    employment = str(row.get(cols.col_employment)).strip() if cols.col_employment and row.get(cols.col_employment) is not None else None
    schedule = str(row.get(cols.col_schedule)).strip() if cols.col_schedule and row.get(cols.col_schedule) is not None else None

    skills = str(row.get(cols.col_skills)).strip() if cols.col_skills and row.get(cols.col_skills) is not None else ""
    full_text = f"{position_text} {skills}".strip()

    label_res = infer_level(position_text=position_text, exp_years=exp_years)
    if label_res.label is None:
        return None

    return PreparedRow(
        age=age,
        salary_rub=salary,
        exp_years=exp_years,
        city=city,
        position_text=position_text,
        employment=employment,
        schedule=schedule,
        skills_text=full_text,
        label=label_res.label,
        label_reason=label_res.reason,
    )


def iter_prepared_rows(
    csv_path: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
) -> Iterator[PreparedRow]:
    last_error: Exception | None = None

    for enc in try_read_csv_encoding():
        try:
            head = pd.read_csv(csv_path, nrows=0, encoding=enc)
            cols = _detect_columns(head.columns.tolist(), target_column_hint)

            for raw in _iter_csv_rows(csv_path, enc, cols, chunksize, limit_rows):
                prepared = _row_to_prepared(raw, cols)
                if prepared is not None:
                    yield prepared
            return

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to read CSV with known encodings. Last error: {last_error}")


def build_dataframe(
    csv_path: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
) -> pd.DataFrame:
    rows = list(
        iter_prepared_rows(
            csv_path=csv_path,
            target_column_hint=target_column_hint,
            chunksize=chunksize,
            limit_rows=limit_rows,
        )
    )
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([asdict(r) for r in rows])


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    vc = df["label"].value_counts()
    return {str(k): int(v) for k, v in vc.items()}
