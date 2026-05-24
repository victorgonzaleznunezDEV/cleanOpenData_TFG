"""Funciones de carga de datasets CSV y Excel."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .config import SUPPORTED_EXTENSIONS


def _read_csv(buffer: BytesIO | str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(buffer, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        return pd.read_csv(buffer, sep=None, engine="python", encoding="latin-1")


def read_uploaded_dataset(uploaded_file: BinaryIO) -> tuple[pd.DataFrame, str]:
    filename = getattr(uploaded_file, "name", "dataset")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Formato no soportado. Formatos permitidos: {allowed}")

    data = BytesIO(uploaded_file.getvalue())
    if extension == ".csv":
        return _read_csv(data), filename
    return pd.read_excel(data), filename


def read_local_dataset(path: str | Path) -> tuple[pd.DataFrame, str]:
    dataset_path = Path(path)
    extension = dataset_path.suffix.lower()
    if extension == ".csv":
        return _read_csv(dataset_path), dataset_path.name
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path), dataset_path.name
    raise ValueError(f"Formato no soportado: {extension}")
