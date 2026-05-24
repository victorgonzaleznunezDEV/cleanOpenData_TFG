"""Deteccion y anonimizacion local de datos sensibles."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from .config import SENSITIVE_KEYWORDS


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?34[\s.-]?)?(?:[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3})(?!\w)")
NIF_RE = re.compile(r"\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b", re.IGNORECASE)
SENSITIVE_PATTERNS = {
    "emails": (EMAIL_RE, "<anon_email>"),
    "telefonos": (PHONE_RE, "<anon_telefono>"),
    "dni_nif": (NIF_RE, "<anon_dni_nif>"),
}


def _column_should_be_scanned(column: str, series: pd.Series) -> bool:
    column_lower = str(column).lower()
    if any(keyword in column_lower for keyword in SENSITIVE_KEYWORDS):
        return True
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def count_sensitive_values(df: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for column in df.columns:
        series = df[column]
        if not _column_should_be_scanned(column, series):
            continue
        for value in series.dropna():
            text = str(value)
            for key, (pattern, _) in SENSITIVE_PATTERNS.items():
                counts[key] += len(pattern.findall(text))
    counts["total"] = sum(counts.values())
    return dict(counts)


def anonymize_sensitive_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    anonymized = df.copy()
    total_counts: Counter[str] = Counter()
    column_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for column in anonymized.columns:
        series = anonymized[column]
        if not _column_should_be_scanned(column, series):
            continue

        def replace_value(value: Any) -> Any:
            if pd.isna(value):
                return value
            new_text = str(value)
            for key, (pattern, replacement) in SENSITIVE_PATTERNS.items():
                new_text, replacements = pattern.subn(replacement, new_text)
                if replacements:
                    total_counts[key] += replacements
                    column_counts[str(column)][key] += replacements
            return new_text

        anonymized[column] = series.map(replace_value)

    total_counts["total"] = sum(total_counts.values())
    incidents: list[dict[str, Any]] = []
    for column, counts in column_counts.items():
        total = sum(counts.values())
        detail = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
        incidents.append(
            {
                "area": "Anonimizacion",
                "severidad": "media",
                "columna": column,
                "registros_afectados": total,
                "detalle": f"Datos sensibles detectados y anonimizados ({detail}).",
            }
        )
    return anonymized, dict(total_counts), incidents
