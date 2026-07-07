"""
Autor: Víctor González Núñez
Fecha: 07/07/2026
Descripción: TFG
"""

from __future__ import annotations

import hashlib
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_anonymizer import DEFAULT_AI_PROVIDER
from src.config import APP_NAME, APP_SUBTITLE
from src.data_loader import read_local_dataset, read_uploaded_dataset
from src.exporter import build_report_frames, dataframe_to_csv_bytes, dataframe_to_xlsx_bytes, report_to_xlsx_bytes
from src.metrics import QualityMetrics, calculate_quality_metrics, detect_format_issues
from src.processing import ProcessingResult, process_dataset
from src.sensitive_data import count_sensitive_values
from src.validator import GXValidationSummary, validate_dataframe


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATASET = BASE_DIR / "data" / "ejemplo_datos_abiertos.csv"


def load_css() -> None:
    css_path = BASE_DIR / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def safe_text(value: object) -> str:
    return escape(str(value), quote=True)


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def read_streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def get_configured_gemini_api_key() -> str | None:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or read_streamlit_secret("GEMINI_API_KEY")
        or read_streamlit_secret("GOOGLE_API_KEY")
    )


def render_front_header() -> None:
    st.html(
        f"""
        <section class="odq-app-window">
            <div class="odq-window-bar">
                <div class="odq-window-dots" aria-hidden="true">
                    <span></span><span></span><span></span>
                </div>
                <div class="odq-window-title">Aplicación de preparacion de datos abiertos</div>
                <div class="odq-window-badge">Local</div>
            </div>
            <header class="odq-hero">
                <div class="odq-hero-copy">
                    <h1>{safe_text(APP_NAME)}</h1>
                    <p class="odq-lead">{safe_text(APP_SUBTITLE)}</p>
                    <div class="odq-tfg-meta">
                        <strong>TFG Metodología para la gobernanza, preparación y publicacion de datos abiertos con ayuda de IA</strong>
                        <span>Autor: Víctor González Núñez</span>
                        <span>Junio 2026, UNIR</span>
                    </div>
                </div>
            </header>
            <div class="odq-capabilities">
                <article>
                    <span>01</span>
                    <strong>Carga</strong>
                    <p>Sube un CSV o Excel, o usa el dataset de ejemplo para probar el flujo.</p>
                </article>
                <article>
                    <span>02</span>
                    <strong>Calidad</strong>
                    <p>Revisa registros, nulos, duplicados, formatos y posibles datos sensibles.</p>
                </article>
                <article>
                    <span>03</span>
                    <strong>Anonimizacion</strong>
                    <p>Aplica reglas locales y, si lo activas, Gemini para columnas sensibles.</p>
                </article>
                <article>
                    <span>04</span>
                    <strong>Exportacion</strong>
                    <p>Descarga el dataset limpio y un informe con metricas e incidencias.</p>
                </article>
            </div>
        </section>
        """,
    )


def render_panel_header(kicker: str, title: str, detail: str = "") -> None:
    detail_html = f"<p>{safe_text(detail)}</p>" if detail else ""
    st.html(
        f"""
        <section class="odq-panel-heading">
            <div class="odq-step-chip">{safe_text(kicker)}</div>
            <div>
                <h2>{safe_text(title)}</h2>
                {detail_html}
            </div>
        </section>
        """
    )


def render_empty_state() -> None:
    st.html(
        """
        <section class="odq-empty">
            <div class="odq-empty-icon">CSV</div>
            <h2>Esperando documento</h2>
            <p>Sube un CSV/XLSX o activa el dataset de ejemplo para iniciar la vista previa.</p>
        </section>
        """
    )


def render_loaded_summary(
    filename: str,
    metrics: QualityMetrics,
    sensitive_counts: dict[str, int],
    use_ai_anonymization: bool,
) -> None:
    cards = [
        ("Archivo", filename),
        ("Registros", fmt_int(metrics.rows)),
        ("Columnas", fmt_int(metrics.columns)),
        ("Nulos", fmt_int(metrics.missing_values)),
        ("Duplicados", fmt_int(metrics.duplicate_rows)),
        ("Sensibles", fmt_int(sensitive_counts.get("total", 0))),
        ("Anonimizar IA", "Si" if use_ai_anonymization else "No"),
    ]
    cards_html = "".join(
        "<article class='odq-summary-card'>"
        f"<span>{safe_text(label)}</span>"
        f"<strong>{safe_text(value)}</strong>"
        "</article>"
        for label, value in cards
    )
    st.html(f"<section class='odq-summary'>{cards_html}</section>")


def render_metric_cards(metrics: QualityMetrics) -> None:
    columns = st.columns(5)
    columns[0].metric("Registros", fmt_int(metrics.rows))
    columns[1].metric("Completitud", f"{metrics.completeness_pct:.1f}%")
    columns[2].metric("Unicidad", f"{metrics.uniqueness_pct:.1f}%")
    columns[3].metric("Consistencia", f"{metrics.consistency_pct:.1f}%")
    columns[4].metric("Errores", fmt_int(metrics.errors_detected))


def render_ai_status(ai_summary: dict[str, object]) -> None:
    render_panel_header(
        "IA",
        "Anonimizacion inteligente",
        "Comprueba si Gemini se ha ejecutado, cuantas columnas ha detectado y si la aplicacion ha mantenido solo las reglas locales.",
    )
    provider = str(ai_summary.get("provider", DEFAULT_AI_PROVIDER)).capitalize()
    cols = st.columns(4)
    cols[0].metric("Proveedor", provider)
    cols[1].metric("Solicitada", "Si" if ai_summary.get("requested") else "No")
    cols[2].metric("Ejecutada", "Si" if ai_summary.get("used") else "No")
    cols[3].metric("Columnas IA", fmt_int(int(ai_summary.get("columns_detected", 0))))

    message = str(ai_summary.get("message", ""))
    if ai_summary.get("requested") and not ai_summary.get("used"):
        st.warning(message)
    elif ai_summary.get("used"):
        st.success(message)
    else:
        st.info(message)


def render_validation(summary: GXValidationSummary, title: str) -> None:
    render_panel_header(
        "Validacion",
        title,
        "Revisa las expectativas de calidad evaluadas sobre el dataset y detecta si quedan valores nulos, formatos incorrectos o reglas incumplidas.",
    )
    if not summary.available:
        st.warning(summary.message)
        return
    if summary.message and summary.evaluated_expectations == 0:
        st.warning(summary.message)
        return

    cols = st.columns(4)
    cols[0].metric("Expectativas", fmt_int(summary.evaluated_expectations))
    cols[1].metric("Correctas", fmt_int(summary.successful_expectations))
    cols[2].metric("Fallidas", fmt_int(summary.unsuccessful_expectations))
    cols[3].metric("Exito GX", f"{summary.success_percent:.1f}%")

    validation_df = summary.as_dataframe()
    if not validation_df.empty:
        st.dataframe(validation_df, width="stretch", hide_index=True)


def render_downloads(
    processed_df: pd.DataFrame,
    filename: str,
    report_frames: dict[str, pd.DataFrame],
) -> None:
    stem = Path(filename).stem or "dataset"
    csv_name = f"{stem}_limpio.csv"
    xlsx_name = f"{stem}_limpio.xlsx"
    report_name = f"{stem}_informe_calidad.xlsx"

    render_panel_header(
        "04",
        "Descarga del resultado",
        "Elige el formato de salida que necesites: CSV para datos abiertos, XLSX para revision en hoja de calculo o informe para conservar la trazabilidad.",
    )
    col_csv, col_xlsx, col_report = st.columns(3)
    col_csv.download_button(
        "Descargar CSV limpio",
        data=dataframe_to_csv_bytes(processed_df),
        file_name=csv_name,
        mime="text/csv",
        width="stretch",
    )
    col_xlsx.download_button(
        "Descargar XLSX limpio",
        data=dataframe_to_xlsx_bytes(processed_df),
        file_name=xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    col_report.download_button(
        "Descargar informe",
        data=report_to_xlsx_bytes(report_frames),
        file_name=report_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )


def analyze_input(df: pd.DataFrame) -> tuple[QualityMetrics, dict[str, int]]:
    format_issues = detect_format_issues(df)
    sensitive_counts = count_sensitive_values(df)
    metrics = calculate_quality_metrics(
        df,
        format_issues=format_issues,
        sensitive_values=sensitive_counts.get("total", 0),
    )
    return metrics, sensitive_counts


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="DQ", layout="wide")
    load_css()
    render_front_header()

    render_panel_header(
        "01",
        "Carga del documento",
        "Sube el dataset que quieres preparar en formato CSV, XLSX o XLS. Si solo quieres probar la aplicacion, activa el dataset de ejemplo.",
    )
    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Subir documento",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
    with sample_col:
        use_sample = st.toggle("Usar ejemplo", value=False)

    render_panel_header(
        "IA",
        "Anonimizacion con IA",
        "Activa esta opcion solo si quieres que Gemini revise una muestra del dataset para detectar columnas sensibles adicionales, como nombres, direcciones o identificadores.",
    )
    use_ai_anonymization = st.checkbox("Anonimizar este fichero con IA", value=False)
    configured_gemini_api_key = get_configured_gemini_api_key()
    runtime_gemini_api_key = configured_gemini_api_key

    if use_ai_anonymization:
        if configured_gemini_api_key:
            st.success("Clave de Gemini configurada para esta ejecucion.")
        else:
            runtime_gemini_api_key = st.text_input(
                "API key de Gemini",
                type="password",
                placeholder="Pega aqui tu clave de Gemini...",
                help="La clave se usa solo durante esta sesion y no se guarda en el proyecto.",
            )
            if not runtime_gemini_api_key:
                st.warning("Introduce una API key de Gemini para ejecutar IA. Sin clave se aplicaran solo reglas locales.")

    df: pd.DataFrame | None = None
    filename = ""
    source_key = ""

    try:
        if uploaded_file is not None:
            df, filename = read_uploaded_dataset(uploaded_file)
            uploaded_bytes = uploaded_file.getvalue()
            source_key = (
                f"upload:{filename}:{hashlib.sha256(uploaded_bytes).hexdigest()}:"
                f"ai={use_ai_anonymization}:gemini_key={bool(runtime_gemini_api_key)}"
            )
        elif use_sample:
            df, filename = read_local_dataset(SAMPLE_DATASET)
            source_key = (
                f"sample:{filename}:{SAMPLE_DATASET.stat().st_mtime_ns}:"
                f"ai={use_ai_anonymization}:gemini_key={bool(runtime_gemini_api_key)}"
            )
    except Exception as exc:
        st.error(f"No se pudo leer el fichero: {exc}")
        return

    if df is None:
        render_empty_state()
        return

    st.session_state.setdefault("last_filename", None)
    if st.session_state["last_filename"] != source_key:
        st.session_state["last_filename"] = source_key
        st.session_state.pop("processing_output", None)

    before_metrics, initial_sensitive_counts = analyze_input(df)
    render_loaded_summary(filename, before_metrics, initial_sensitive_counts, use_ai_anonymization)

    render_panel_header(
        "02",
        "Resultados de la carga",
        "Comprueba que el fichero se ha leido correctamente. Esta vista muestra las primeras filas y las metricas iniciales antes de limpiar los datos.",
    )
    st.dataframe(df.head(50), width="stretch", hide_index=True)
    render_metric_cards(before_metrics)

    render_panel_header(
        "03",
        "Procesamiento",
        "Pulsa el boton para normalizar columnas y textos, corregir formatos basicos, anonimizar datos sensibles, eliminar duplicados y completar valores nulos.",
    )
    process_clicked = st.button("Procesar dataset", type="primary", width="stretch")
    if process_clicked:
        with st.spinner("Procesando dataset y ejecutando validaciones..."):
            validation_before = validate_dataframe(df)
            processing_result = process_dataset(
                df,
                use_ai_anonymization=use_ai_anonymization,
                gemini_api_key=runtime_gemini_api_key,
            )
            processed_df = processing_result.dataframe
            after_format_issues = detect_format_issues(processed_df)
            after_sensitive_counts = count_sensitive_values(processed_df)
            after_metrics = calculate_quality_metrics(
                processed_df,
                format_issues=after_format_issues,
                sensitive_values=after_sensitive_counts.get("total", 0),
            )
            validation_after = validate_dataframe(processed_df)
            report_frames = build_report_frames(
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                incidents=processing_result.incidents,
                validation_before=validation_before,
                validation_after=validation_after,
                sensitive_summary=processing_result.sensitive_summary,
                ai_summary=processing_result.ai_summary,
            )
            st.session_state["processing_output"] = {
                "result": processing_result,
                "after_metrics": after_metrics,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "report_frames": report_frames,
                "use_ai_anonymization": use_ai_anonymization,
            }

    output = st.session_state.get("processing_output")
    if not output:
        st.stop()

    processing_result: ProcessingResult = output["result"]
    after_metrics: QualityMetrics = output["after_metrics"]
    validation_before: GXValidationSummary = output["validation_before"]
    validation_after: GXValidationSummary = output["validation_after"]
    report_frames: dict[str, pd.DataFrame] = output["report_frames"]

    render_panel_header(
        "03.1",
        "Resultados del procesamiento",
        "Revisa las metricas finales para comparar la calidad del dataset despues de aplicar limpieza, validacion y anonimizacion.",
    )
    render_metric_cards(after_metrics)
    render_ai_status(processing_result.ai_summary)

    render_panel_header(
        "Anonimizacion",
        "Datos sensibles anonimizados",
        "Consulta cuantos valores se han sustituido mediante reglas locales y cuantos proceden del plan generado por Gemini.",
    )
    anon_cols = st.columns(5)
    anon_cols[0].metric("Emails", fmt_int(processing_result.sensitive_summary.get("emails", 0)))
    anon_cols[1].metric("Telefonos", fmt_int(processing_result.sensitive_summary.get("telefonos", 0)))
    anon_cols[2].metric("DNI/NIF", fmt_int(processing_result.sensitive_summary.get("dni_nif", 0)))
    anon_cols[3].metric("IA", fmt_int(processing_result.sensitive_summary.get("ia", 0)))
    anon_cols[4].metric("Total", fmt_int(processing_result.sensitive_summary.get("total", 0)))

    render_panel_header(
        "Vista final",
        "Dataset procesado",
        "Comprueba una muestra del resultado final antes de descargarlo. Los datos sensibles detectados aparecen sustituidos por marcadores anonimos.",
    )
    st.dataframe(processing_result.dataframe.head(100), width="stretch", hide_index=True)

    render_panel_header(
        "Control",
        "Incidencias detectadas",
        "Revisa las acciones aplicadas durante el procesamiento: normalizaciones, formatos corregidos, duplicados eliminados y anonimizaciones realizadas.",
    )
    incidents_df = pd.DataFrame(processing_result.incidents)
    if incidents_df.empty:
        st.success("No se detectaron incidencias durante el procesamiento.")
    else:
        st.dataframe(incidents_df, width="stretch", hide_index=True)

    render_validation(validation_before, "Great Expectations antes")
    render_validation(validation_after, "Great Expectations despues")
    render_downloads(processing_result.dataframe, filename, report_frames)


if __name__ == "__main__":
    main()
