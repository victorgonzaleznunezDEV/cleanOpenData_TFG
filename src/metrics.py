"""
Autor: Víctor González Núñez
Fecha: 07/07/2026
Descripción: TFG
"""

"""Calculo de metricas e incidencias de calidad del dato."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import DATE_KEYWORDS, EMAIL_KEYWORDS, NIF_KEYWORDS, PHONE_KEYWORDS, STANDARD_MISSING_VALUE
from .sensitive_data import EMAIL_RE, NIF_RE, PHONE_RE


@dataclass(frozen=True)
class QualityMetrics:
    rows: int
    columns: int
    total_cells: int
    missing_values: int
    duplicate_rows: int
    completeness_pct: float
    uniqueness_pct: float
    consistency_pct: float
    format_errors: int
    errors_detected: int
    records_processed: int
    sensitive_values: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_expected_format(column: str) -> str | None:
    column_lower = str(column).lower()
    if any(keyword in column_lower for keyword in EMAIL_KEYWORDS):
        return "email"
    if any(keyword in column_lower for keyword in PHONE_KEYWORDS):
        return "telefono"
    if any(keyword in column_lower for keyword in NIF_KEYWORDS):
        return "dni_nif"
    if any(keyword in column_lower for keyword in DATE_KEYWORDS):
        return "fecha"
    return None


def regex_for_expected_format(expected_format: str) -> str | None:
    regexes = {
        "email": r"^(?:<anon_email>|no_informado|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})$",
        "telefono": r"^(?:<anon_telefono>|no_informado|(?:\+?34[\s.-]?)?(?:[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}))$",
        "dni_nif": r"^(?:<anon_dni_nif>|no_informado|(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z]))$",
        "fecha": r"^(?:no_informado|\d{4}-\d{2}-\d{2})$",
    }
    return regexes.get(expected_format)


def value_matches_expected_format(value: Any, expected_format: str) -> bool:
    if pd.isna(value):
        return True

    text = str(value).strip()
    normalized = text.lower()
    if normalized == STANDARD_MISSING_VALUE or normalized.startswith("<anon_"):
        return True

    if expected_format == "email":
        return bool(EMAIL_RE.fullmatch(text))
    if expected_format == "telefono":
        return bool(PHONE_RE.fullmatch(text))
    if expected_format == "dni_nif":
        return bool(NIF_RE.fullmatch(text))
    if expected_format == "fecha":
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))
    return True


def detect_format_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for column in df.columns:
        expected_format = infer_expected_format(str(column))
        if expected_format is None:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        invalid_mask = ~series.map(lambda value: value_matches_expected_format(value, expected_format))
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            examples = series[invalid_mask].astype(str).head(3).tolist()
            issues.append(
                {
                    "area": "Validacion de formatos",
                    "severidad": "alta",
                    "columna": str(column),
                    "registros_afectados": invalid_count,
                    "detalle": (
                        f"Valores no compatibles con '{expected_format}'. "
                        f"Ejemplos: {', '.join(examples)}"
                    ),
                }
            )
    return issues


def calculate_quality_metrics(
    df: pd.DataFrame,
    *,
    format_issues: list[dict[str, Any]] | None = None,
    sensitive_values: int = 0,
) -> QualityMetrics:
    rows, columns = df.shape
    total_cells = int(rows * columns)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    format_errors = int(sum(issue.get("registros_afectados", 0) for issue in (format_issues or [])))
    completeness_pct = 100.0 if total_cells == 0 else round(100 * (1 - missing_values / total_cells), 2)
    uniqueness_pct = 100.0 if rows == 0 else round(100 * (1 - duplicate_rows / rows), 2)
    errors = missing_values + duplicate_rows + format_errors + int(sensitive_values)
    consistency_pct = round(max(0.0, 100 * (1 - format_errors / max(total_cells, 1))), 2)

    return QualityMetrics(
        rows=int(rows),
        columns=int(columns),
        total_cells=total_cells,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        completeness_pct=completeness_pct,
        uniqueness_pct=uniqueness_pct,
        consistency_pct=consistency_pct,
        format_errors=format_errors,
        errors_detected=int(errors),
        records_processed=int(rows),
        sensitive_values=int(sensitive_values),
    )
