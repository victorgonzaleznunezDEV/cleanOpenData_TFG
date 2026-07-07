"""
Autor: Víctor González Núñez
Fecha: 07/07/2026
Descripción: TFG
"""
"""Motor principal de limpieza y transformacion de datasets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .ai_anonymizer import anonymize_with_ai
from .config import NON_NUMERIC_KEYWORDS, NULL_TOKENS, STANDARD_MISSING_VALUE
from .metrics import detect_format_issues, infer_expected_format, value_matches_expected_format
from .sensitive_data import anonymize_sensitive_data


@dataclass
class ProcessingResult:
    dataframe: pd.DataFrame
    incidents: list[dict[str, Any]]
    sensitive_summary: dict[str, int]
    ai_summary: dict[str, Any]


def normalize_column_name(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "columna"


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    normalized = df.copy()
    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}
    new_columns: list[str] = []
    for original in normalized.columns:
        base_name = normalize_column_name(original)
        count = seen.get(base_name, 0) + 1
        seen[base_name] = count
        final_name = base_name if count == 1 else f"{base_name}_{count}"
        mapping[str(original)] = final_name
        new_columns.append(final_name)
    normalized.columns = new_columns
    return normalized, mapping


def standardize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        series = cleaned[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        def clean_value(value: Any) -> Any:
            if pd.isna(value):
                return pd.NA
            text = unicodedata.normalize("NFKC", str(value))
            text = re.sub(r"\s+", " ", text).strip()
            if text.lower() in NULL_TOKENS:
                return pd.NA
            return text

        cleaned[column] = series.map(clean_value)
    return cleaned


def standardize_basic_formats(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cleaned = df.copy()
    incidents: list[dict[str, Any]] = []
    for column in cleaned.columns:
        series = cleaned[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        expected_format = infer_expected_format(column)
        if expected_format == "fecha":
            parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
            if float(parsed.notna().mean()) >= 0.75:
                parsed_full = pd.to_datetime(series, errors="coerce", dayfirst=True)
                invalid_count = int(series.notna().sum() - parsed_full.notna().sum())
                cleaned[column] = parsed_full.dt.strftime("%Y-%m-%d")
                cleaned.loc[parsed_full.isna() & series.notna(), column] = STANDARD_MISSING_VALUE
                if invalid_count:
                    incidents.append(
                        _incident("Estandarizacion", "media", column, invalid_count, "Fecha no convertible a ISO.")
                    )
        elif not any(keyword in column.lower() for keyword in NON_NUMERIC_KEYWORDS):
            normalized_numbers = non_null.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            converted = pd.to_numeric(normalized_numbers, errors="coerce")
            if float(converted.notna().mean()) >= 0.9:
                full_values = series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                converted_full = pd.to_numeric(full_values, errors="coerce")
                invalid_count = int(series.notna().sum() - converted_full.notna().sum())
                cleaned[column] = converted_full
                if invalid_count:
                    incidents.append(
                        _incident("Estandarizacion", "media", column, invalid_count, "Numero no convertible.")
                    )
    return cleaned, incidents


def replace_invalid_format_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cleaned = df.copy()
    incidents: list[dict[str, Any]] = []
    for column in cleaned.columns:
        expected_format = infer_expected_format(column)
        if expected_format is None:
            continue
        series = cleaned[column]
        invalid_mask = series.notna() & ~series.map(lambda value: value_matches_expected_format(value, expected_format))
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            cleaned.loc[invalid_mask, column] = pd.NA
            incidents.append(
                _incident(
                    "Validacion de formatos",
                    "alta",
                    column,
                    invalid_count,
                    f"Valores no compatibles con {expected_format}.",
                )
            )
    return cleaned, incidents


def fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    filled = df.copy()
    incidents: list[dict[str, Any]] = []
    for column in filled.columns:
        missing_count = int(filled[column].isna().sum())
        if missing_count == 0:
            continue
        series = filled[column]
        if pd.api.types.is_numeric_dtype(series):
            replacement = series.median(skipna=True)
            if pd.isna(replacement):
                replacement = 0
        else:
            replacement = STANDARD_MISSING_VALUE
        filled[column] = series.fillna(replacement)
        incidents.append(
            _incident(
                "Tratamiento de nulos",
                "media",
                column,
                missing_count,
                f"Valores nulos reemplazados por '{replacement}'.",
            )
        )
    return filled, incidents


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    duplicated_count = int(df.duplicated().sum())
    if duplicated_count == 0:
        return df, []
    return df.drop_duplicates().reset_index(drop=True), [
        _incident("Duplicados", "alta", "dataset", duplicated_count, "Registros duplicados eliminados.")
    ]


def _incident(area: str, severity: str, column: str, rows: int, detail: str) -> dict[str, Any]:
    return {
        "area": area,
        "severidad": severity,
        "columna": column,
        "registros_afectados": rows,
        "detalle": detail,
    }


def process_dataset(
    df: pd.DataFrame,
    *,
    use_ai_anonymization: bool = False,
    gemini_api_key: str | None = None,
) -> ProcessingResult:
    incidents: list[dict[str, Any]] = []

    processed, column_mapping = normalize_columns(df)
    changed_columns = {old: new for old, new in column_mapping.items() if old != new}
    if changed_columns:
        incidents.append(
            _incident("Normalizacion", "baja", "dataset", len(changed_columns), "Columnas normalizadas a snake_case.")
        )

    processed = standardize_text_values(processed)
    incidents.extend(detect_format_issues(processed))

    processed, new_incidents = standardize_basic_formats(processed)
    incidents.extend(new_incidents)

    processed, sensitive_summary, new_incidents = anonymize_sensitive_data(processed)
    sensitive_summary.setdefault("emails", 0)
    sensitive_summary.setdefault("telefonos", 0)
    sensitive_summary.setdefault("dni_nif", 0)
    sensitive_summary.setdefault("ia", 0)
    sensitive_summary.setdefault("total", 0)
    incidents.extend(new_incidents)

    if use_ai_anonymization:
        ai_outcome = anonymize_with_ai(
            processed,
            requested=True,
            api_key=gemini_api_key,
        )
        processed = ai_outcome.dataframe
        sensitive_summary["ia"] = int(ai_outcome.summary.get("values_anonymized", 0))
        sensitive_summary["total"] = int(sensitive_summary.get("total", 0)) + sensitive_summary["ia"]
        incidents.extend(ai_outcome.incidents)
        ai_summary = ai_outcome.summary
    else:
        ai_summary = anonymize_with_ai(
            processed,
            requested=False,
        ).summary

    processed, new_incidents = replace_invalid_format_values(processed)
    incidents.extend(new_incidents)

    processed, new_incidents = remove_duplicate_rows(processed)
    incidents.extend(new_incidents)

    processed, new_incidents = fill_missing_values(processed)
    incidents.extend(new_incidents)

    return ProcessingResult(processed, incidents, sensitive_summary, ai_summary)
