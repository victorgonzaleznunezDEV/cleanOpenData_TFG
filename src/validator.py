"""
Autor: Víctor González Núñez
Fecha: 07/07/2026
Descripción: TFG
"""

"""Validacion de datasets con expectativas de calidad compatibles con Great Expectations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .metrics import infer_expected_format, value_matches_expected_format


@dataclass
class GXValidationSummary:
    available: bool
    success: bool
    evaluated_expectations: int
    successful_expectations: int
    unsuccessful_expectations: int
    success_percent: float
    rows: list[dict[str, Any]]
    message: str = ""

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _row(expectation: str, column: str, success: bool, unexpected_count: int, total: int) -> dict[str, Any]:
    unexpected_percent = round(100 * unexpected_count / total, 2) if total else 0.0
    return {
        "expectativa": expectation,
        "columna": column,
        "exito": success,
        "valores_inesperados": unexpected_count,
        "porcentaje_inesperado": unexpected_percent,
    }


def validate_dataframe(df: pd.DataFrame) -> GXValidationSummary:
    rows: list[dict[str, Any]] = []
    row_count = int(len(df))
    rows.append(_row("ExpectTableRowCountToBeBetween", "dataset", row_count >= 1, 0 if row_count >= 1 else 1, 1))

    for column in df.columns:
        column_name = str(column)
        total = int(len(df[column]))
        missing_count = int(df[column].isna().sum())
        missing_ratio = missing_count / total if total else 0.0
        rows.append(
            _row(
                "ExpectColumnValuesToNotBeNull",
                column_name,
                missing_ratio <= 0.05,
                missing_count,
                total,
            )
        )

        expected_format = infer_expected_format(column_name)
        if expected_format:
            invalid_mask = df[column].notna() & ~df[column].map(
                lambda value: value_matches_expected_format(value, expected_format)
            )
            invalid_count = int(invalid_mask.sum())
            rows.append(
                _row(
                    "ExpectColumnValuesToMatchRegex",
                    column_name,
                    invalid_count / total <= 0.05 if total else True,
                    invalid_count,
                    total,
                )
            )

    evaluated = len(rows)
    successful = sum(1 for row in rows if row["exito"])
    unsuccessful = evaluated - successful
    success_percent = round(100 * successful / evaluated, 2) if evaluated else 0.0
    return GXValidationSummary(True, unsuccessful == 0, evaluated, successful, unsuccessful, success_percent, rows)
