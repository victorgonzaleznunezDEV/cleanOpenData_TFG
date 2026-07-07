"""
Autor: Víctor González Núñez
Fecha: 07/07/2026
Descripción: TFG
""""""Exportacion de datasets e informes de calidad."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

from .metrics import QualityMetrics
from .validator import GXValidationSummary


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(df: pd.DataFrame, *, sheet_name: str = "datos_limpios") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buffer.seek(0)
    return buffer.getvalue()


def _incident(area: str, severity: str, column: str, rows: int, detail: str) -> dict[str, Any]:
    return {
        "area": area,
        "severidad": severity,
        "columna": column,
        "registros_afectados": rows,
        "detalle": detail,
    }


def build_report_frames(
    *,
    before_metrics: QualityMetrics,
    after_metrics: QualityMetrics,
    incidents: list[dict[str, Any]],
    validation_before: GXValidationSummary,
    validation_after: GXValidationSummary,
    sensitive_summary: dict[str, int],
    ai_summary: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    incidents_df = pd.DataFrame(incidents)
    if incidents_df.empty:
        incidents_df = pd.DataFrame(
            [_incident("Sin incidencias", "info", "dataset", 0, "No se detectaron incidencias.")]
        )

    return {
        "metricas": pd.DataFrame(
            [
                {"fase": "antes", **before_metrics.as_dict()},
                {"fase": "despues", **after_metrics.as_dict()},
            ]
        ),
        "incidencias": incidents_df,
        "gx_antes": validation_before.as_dataframe()
        if validation_before.rows
        else pd.DataFrame([{"mensaje": validation_before.message}]),
        "gx_despues": validation_after.as_dataframe()
        if validation_after.rows
        else pd.DataFrame([{"mensaje": validation_after.message}]),
        "anonimizacion": pd.DataFrame(
            [{"tipo": key, "cantidad": value} for key, value in sorted(sensitive_summary.items())]
        ),
        "ia": pd.DataFrame([ai_summary or {"requested": False, "used": False}]),
    }


def report_to_xlsx_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buffer.seek(0)
    return buffer.getvalue()
