# Open Data Clean App

Aplicación web en Python y Streamlit para preparar conjuntos de datos abiertos antes de su publicacion. Permite cargar ficheros CSV o Excel, limpiar datos, validar calidad, anonimizar informacion sensible y descargar el dataset procesado junto con un informe de incidencias.

## Estructura del proyecto

.
|-- app.py
|-- assets/
|   `-- styles.css
|-- data/
|   |-- ejemplo_datos_abiertos.csv
|-- requirements.txt
|-- src/
|   |-- ai_anonymizer.py
|   |-- config.py
|   |-- data_loader.py
|   |-- exporter.py
|   |-- metrics.py
|   |-- processing.py
|   |-- sensitive_data.py
|   `-- validator.py



## Instalación

Desde la carpeta del proyecto:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt



## Ejecucion

Con el entorno virtual activado:
streamlit run app.py

Streamlit mostrara una URL local, normalmente:
http://localhost:8501
