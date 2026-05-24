"""Anonimizacion asistida por IA para columnas sensibles no evidentes."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


DEFAULT_AI_PROVIDER = "gemini"
DEFAULT_AI_ANONYMIZATION_MODEL = "gemini-2.5-flash"
AI_CONFIDENCE_THRESHOLD = 0.55
AI_RETRY_DELAYS_SECONDS = (1.0, 3.0)


class AIColumnDecision(BaseModel):
    column: str = Field(description="Nombre exacto de la columna del dataset.")
    pii_type: Literal[
        "person_name",
        "address",
        "email",
        "phone",
        "national_id",
        "free_text_pii",
        "other_identifier",
        "not_sensitive",
    ]
    strategy: Literal["redact_column", "keep"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class AIAnonymizationPlan(BaseModel):
    columns: list[AIColumnDecision]


@dataclass
class AIAnonymizationOutcome:
    dataframe: pd.DataFrame
    summary: dict[str, Any]
    incidents: list[dict[str, Any]]


PLACEHOLDERS = {
    "person_name": "<anon_nombre>",
    "address": "<anon_direccion>",
    "email": "<anon_email>",
    "phone": "<anon_telefono>",
    "national_id": "<anon_documento>",
    "free_text_pii": "<anon_texto_sensible>",
    "other_identifier": "<anon_identificador>",
}


def _resolve_model(model: str | None = None) -> str:
    return model or DEFAULT_AI_ANONYMIZATION_MODEL


def _candidate_models(model: str | None = None) -> list[str]:
    return [_resolve_model(model)]


def _sample_dataframe(df: pd.DataFrame, max_rows: int = 12) -> dict[str, Any]:
    sample = df.head(max_rows).where(pd.notna(df.head(max_rows)), None)
    payload = {
        "columns": list(map(str, df.columns)),
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "sample_rows": sample.astype(object).to_dict(orient="records"),
    }
    return json.loads(json.dumps(payload, default=str, ensure_ascii=False))


def _empty_summary(
    *,
    requested: bool,
    used: bool,
    message: str,
    model: str | None = None,
) -> dict[str, Any]:
    return {
        "requested": requested,
        "used": used,
        "provider": DEFAULT_AI_PROVIDER,
        "model": _resolve_model(model),
        "columns_detected": 0,
        "values_anonymized": 0,
        "message": message,
    }


def _build_plan(df: pd.DataFrame, *, api_key: str, model: str) -> AIAnonymizationPlan:
    from google import genai

    client = genai.Client(api_key=api_key)
    sample_payload = _sample_dataframe(df)
    prompt = (
        "Eres un asistente de gobierno del dato. Analiza columnas de un dataset "
        "tabular y decide si deben anonimizarse antes de publicar datos abiertos. "
        "Usa exactamente los nombres de columnas recibidos. Marca como sensibles "
        "nombres de personas, direcciones, emails, telefonos, documentos oficiales, "
        "identificadores personales y texto libre con datos personales. Manten "
        "categorias generales, importes, fechas y observaciones no personales.\n\n"
        "Genera un plan de anonimizacion para este dataset. "
        f"Dataset resumido en JSON: {json.dumps(sample_payload, ensure_ascii=False)}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": AIAnonymizationPlan.model_json_schema(),
        },
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, AIAnonymizationPlan):
        return parsed
    if isinstance(parsed, dict):
        return AIAnonymizationPlan.model_validate(parsed)
    if getattr(response, "text", None):
        return AIAnonymizationPlan.model_validate_json(response.text)

    raise ValueError("Gemini no devolvio un plan estructurado.")


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).upper()
    retryable_markers = (
        "503",
        "502",
        "504",
        "429",
        "500",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "INSUFFICIENT_SYSTEM_RESOURCE",
    )
    return any(marker in text for marker in retryable_markers)


def _friendly_error_message(exc: Exception, *, model: str) -> str:
    text = str(exc)
    if "503" in text or "UNAVAILABLE" in text.upper():
        return (
            f"Gemini no esta disponible temporalmente para el modelo {model}. "
            "Puede deberse a alta demanda. Se mantiene la anonimizacion local; "
            "prueba de nuevo en unos minutos."
        )
    if "429" in text or "RESOURCE_EXHAUSTED" in text.upper():
        return (
            f"Gemini ha rechazado la peticion por limite de cuota o saturacion en el modelo {model}. "
            "Se mantiene la anonimizacion local."
        )
    return f"No se pudo ejecutar la anonimizacion IA con Gemini usando el modelo {model}: {exc}"


def _build_gemini_plan_with_retries(
    df: pd.DataFrame,
    *,
    api_key: str,
    model: str,
) -> tuple[AIAnonymizationPlan, str, int]:
    last_error: Exception | None = None
    attempt = 0

    for candidate_model in _candidate_models(model):
        for retry_index, delay in enumerate((0.0, *AI_RETRY_DELAYS_SECONDS), start=1):
            attempt += 1
            if delay:
                time.sleep(delay)
            try:
                plan = _build_plan(df, api_key=api_key, model=candidate_model)
                return plan, candidate_model, attempt
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    raise
                if retry_index == len(AI_RETRY_DELAYS_SECONDS) + 1:
                    break

    if last_error is None:
        raise RuntimeError("No se pudo seleccionar un modelo de Gemini.")
    raise last_error


def anonymize_with_ai(
    df: pd.DataFrame,
    *,
    requested: bool,
    api_key: str | None = None,
    model: str | None = None,
) -> AIAnonymizationOutcome:
    """Aplica un plan de anonimizacion generado por un proveedor IA sobre columnas completas."""
    resolved_model = _resolve_model(model)

    if not requested:
        return AIAnonymizationOutcome(
            dataframe=df,
            summary=_empty_summary(
                requested=False,
                used=False,
                message="Anonimizacion IA no solicitada.",
                model=resolved_model,
            ),
            incidents=[],
        )

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not resolved_api_key:
        return AIAnonymizationOutcome(
            dataframe=df,
            summary=_empty_summary(
                requested=True,
                used=False,
                message="Falta GEMINI_API_KEY. Se mantiene la anonimizacion local por reglas.",
                model=resolved_model,
            ),
            incidents=[
                {
                    "area": "Anonimizacion IA",
                    "severidad": "media",
                    "columna": "dataset",
                    "registros_afectados": 0,
                    "detalle": "No se ejecuto IA porque no existe GEMINI_API_KEY.",
                }
            ],
        )

    try:
        plan, used_model, attempts = _build_gemini_plan_with_retries(
            df,
            api_key=resolved_api_key,
            model=resolved_model,
        )
    except Exception as exc:
        friendly_message = _friendly_error_message(exc, model=resolved_model)
        return AIAnonymizationOutcome(
            dataframe=df,
            summary=_empty_summary(
                requested=True,
                used=False,
                message=friendly_message,
                model=resolved_model,
            ),
            incidents=[
                {
                    "area": "Anonimizacion IA",
                    "severidad": "alta",
                    "columna": "dataset",
                    "registros_afectados": 0,
                    "detalle": friendly_message,
                }
            ],
        )

    anonymized = df.copy()
    incidents: list[dict[str, Any]] = []
    valid_columns = set(map(str, anonymized.columns))
    columns_detected = 0
    values_anonymized = 0

    for decision in plan.columns:
        if decision.strategy != "redact_column":
            continue
        if decision.pii_type == "not_sensitive":
            continue
        if decision.confidence < AI_CONFIDENCE_THRESHOLD:
            continue
        if decision.column not in valid_columns:
            continue

        placeholder = PLACEHOLDERS.get(decision.pii_type, "<anon_ia>")
        mask = anonymized[decision.column].notna()
        changed_mask = mask & (anonymized[decision.column].astype(str) != placeholder)
        changed_count = int(changed_mask.sum())

        if changed_count == 0:
            continue

        anonymized[decision.column] = anonymized[decision.column].astype(object)
        anonymized.loc[mask, decision.column] = placeholder
        columns_detected += 1
        values_anonymized += changed_count
        incidents.append(
            {
                "area": "Anonimizacion IA",
                "severidad": "alta",
                "columna": decision.column,
                "registros_afectados": changed_count,
                "detalle": (
                    f"Columna anonimizada por IA como {decision.pii_type}. "
                    f"Confianza: {decision.confidence:.2f}. Motivo: {decision.reason}"
                ),
            }
        )

    return AIAnonymizationOutcome(
        dataframe=anonymized,
        summary={
            "requested": True,
            "used": True,
            "provider": DEFAULT_AI_PROVIDER,
            "model": used_model,
            "attempts": attempts,
            "columns_detected": columns_detected,
            "values_anonymized": values_anonymized,
            "message": "Plan de anonimizacion Gemini aplicado.",
        },
        incidents=incidents,
    )
