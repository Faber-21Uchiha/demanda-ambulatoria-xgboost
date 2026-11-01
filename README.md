# Predicción de la demanda ambulatoria con XGBoost

Repositorio de código, datos sintéticos y resultados del proyecto de grado **«Modelo predictivo de la demanda de servicios ambulatorios mediante XGBoost»**, desarrollado en la Especialización en Analítica de Datos de la Corporación Unificada Nacional de Educación Superior (CUN).

El estudio se realiza sobre una entidad anonimizada, **Red Integrada de Servicios de Salud Metropolitana E.S.E.**, con cuatro zonas de operación (Norte, Sur, Centro y Occidente). Todos los registros publicados en este repositorio son **sintéticos**: no contienen información de personas reales ni datos institucionales.

## Contenido

| Componente | Objetivo específico | Scripts |
|---|---|---|
| Calidad, depuración y análisis exploratorio | OE1 | `01` a `04` |
| Estructura temporal, estabilidad y conducta prescriptiva | OE2 | `05` |
| Modelo predictivo XGBoost | OE3 | `06` |
| Figuras del informe | OE1–OE3 | `07`, `08` |

## Estructura del repositorio

```
.
├── 01_datos/
│   ├── demanda_ambulatoria_sintetica_2024_2025_100k.csv   # 100.000 registros brutos (34 MB)
│   ├── dataset_depurado_OE1.csv                           # 87.830 registros depurados (30 MB)
│   ├── diccionario_y_reglas_OE1.xlsx                      # diccionario y reglas de validación
│   └── README_dataset_sintetico.txt                       # ficha del generador
├── 02_codigo/
│   ├── 01_generar_datos_sinteticos.py
│   ├── 02_evaluar_calidad.py
│   ├── 03_limpiar_datos.py
│   ├── 04_analisis_exploratorio.py
│   ├── 05_analisis_series_y_asociacion.py
│   ├── 06_modelo_xgboost.py
│   ├── 07_figuras_oe2.py
│   └── 08_figuras_fase3.py
├── 03_resultados/
│   ├── figuras/                                           # figuras del documento (PNG, 200 ppp)
│   ├── tablas/                                            # tablas intermedias (CSV)
│   ├── metricas_calidad.json
│   ├── resultados_series_y_asociacion.json
│   ├── modelos_comparados.json
│   ├── resultados_modelo_xgboost.json
│   └── modelo_xgboost.json                                # artefacto del modelo entrenado
└── 04_documentos/
    └── README.md
```

## Requisitos

- Python 3.11 o superior
- Dependencias: ver `requirements.txt`

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

Los scripts se ejecutan **en orden** desde la raíz del repositorio. Cada uno escribe sus salidas en `03_resultados/` y no modifica las entradas.

```bash
python 02_codigo/01_generar_datos_sinteticos.py        # genera los 100.000 registros (~1 min)
python 02_codigo/02_evaluar_calidad.py                 # 15 indicadores de calidad (~1 min)
python 02_codigo/03_limpiar_datos.py                   # depuración trazable (~1 min)
python 02_codigo/04_analisis_exploratorio.py           # EDA y figuras del OE1 (~1 min)
python 02_codigo/05_analisis_series_y_asociacion.py    # ACF, STL, VIF, estabilidad y OE2 (~2 min)
python 02_codigo/06_modelo_xgboost.py                  # XGBoost, validación temporal y métricas (~3 min)
python 02_codigo/07_figuras_oe2.py                     # figuras del OE2
python 02_codigo/08_figuras_fase3.py                   # arquitectura, importancia y residuos
```

El paso `01` puede omitirse si se parte de los archivos CSV ya incluidos en `01_datos/`.

## Reproducibilidad

| Elemento | Valor |
|---|---|
| Semilla de generación de datos | `20261005` |
| Semilla de análisis y modelado | `2026` |
| Partición temporal | 73 semanas de entrenamiento (2024-04-07 a 2025-08-24) y 18 de prueba (2025-08-31 a 2025-12-28) |
| Validación cruzada | `TimeSeriesSplit`, 4 pliegues, ventana expansiva |
| Versiones de referencia | pandas 2.3.3, NumPy 2.3.4, scikit-learn 1.9.0, statsmodels 0.15.0, XGBoost 3.4.1, Matplotlib 3.10.7 |

Integridad de los datos (SHA-256):

```
1e8f4b08aca876f10890721fca0946e6c531b3f1739bd57cccae9c2eff7b2863  demanda_ambulatoria_sintetica_2024_2025_100k.csv
a94d19d09be4a68e7381d83d664ef22086691defbc8391262414a7748da28ef9  dataset_depurado_OE1.csv
```

## Resultados principales

Serie semanal: 104 semanas completas, media de 840,8 registros, desviación estándar de 52,5.

| Modelo | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| Media histórica | 52,4 | 59,7 | 6,06 | −0,130 |
| Naive simple (t−1) | 41,4 | 54,5 | 4,87 | 0,061 |
| Naive estacional (t−52) | 33,8 | 44,0 | 3,82 | 0,387 |
| Regresión lineal múltiple | 40,7 | 56,3 | 4,59 | −0,004 |
| **XGBoost** | **40,1** | **49,5** | **4,63** | **0,223** |

Métricas sobre las 18 semanas del conjunto de prueba. El detalle de la validación temporal, la importancia de variables y el diagnóstico de residuos se encuentra en `03_resultados/resultados_modelo_xgboost.json`.

## Privacidad y datos

Los datos institucionales no se publican: contienen información personal y clínica protegida por la Ley 1581 de 2012 y la Resolución 2275 de 2023. La capa publicada es una réplica sintética que reproduce la estructura, los problemas de calidad y las distribuciones marginales del universo previsto, pero no sus relaciones conjuntas; por ello los resultados se interpretan como validación metodológica y no como medición del comportamiento real de la entidad.

## Cómo citar

```
Aleman Mendoza, F. (2026). Modelo de machine learning para la predicción de la demanda de
servicios ambulatorios basado en perfiles diagnósticos [Repositorio de código]. GitHub.
https://github.com/Faber-21Uchiha/demanda-ambulatoria-xgboost
```

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos sintéticos se distribuyen con fines académicos y de replicación.
