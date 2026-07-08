"""Configuracion comunn de la aplicacion."""

from __future__ import annotations


APP_NAME = "Open Data Clean App"
APP_SUBTITLE = "Aplicacion para la preparacion, validación y anonimización de datos abiertos"
STANDARD_MISSING_VALUE = "no_informado"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

NULL_TOKENS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "sin dato",
    "sin datos",
    "no informado",
    "no aplica",
}

DATE_KEYWORDS = ("fecha", "date", "fch")
EMAIL_KEYWORDS = ("email", "e_mail", "mail", "correo")
PHONE_KEYWORDS = ("telefono", "teléfono", "phone", "movil", "móvil", "celular")
NIF_KEYWORDS = ("dni", "nif", "nie", "documento")
SENSITIVE_KEYWORDS = EMAIL_KEYWORDS + PHONE_KEYWORDS + NIF_KEYWORDS
NON_NUMERIC_KEYWORDS = ("id", "codigo", "código", "cod", "postal", "zip") + SENSITIVE_KEYWORDS
