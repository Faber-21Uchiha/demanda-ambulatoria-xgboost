# Predicción de la demanda ambulatoria con XGBoost

Repositorio de código, datos sintéticos y resultados del proyecto de grado **«Modelo predictivo de la demanda de servicios ambulatorios mediante XGBoost»**, desarrollado en la Especialización en Analítica de Datos de la Corporación Unificada Nacional de Educación Superior (CUN).

El estudio se realiza sobre una entidad anonimizada, **Red Integrada de Servicios de Salud Metropolitana E.S.E.**, con cuatro zonas de operación (Norte, Sur, Centro y Occidente). Todos los registros publicados en este repositorio son **sintéticos**: no contienen información de personas reales ni datos institucionales.

## Contenido

| Componente | Objetivo específico | Scripts |
|---|---|---|
| Calidad, depuración y análisis exploratorio | OE1 | `01` a `04` |
| Estructura temporal, estabilidad y conducta prescriptiva | OE2 | `05` |
| Modelo predictivo XGBoost (agregado y panel zona-semana) | OE3 | `06`, `09`, `10` |
| Protocolo de operación del modelo | OE4 | `12` |
| Simulación de impacto sobre capacidad y PQRS | OE5 | `11` |
| Figuras del informe | OE1–OE5 | `07`, `08`, `13` |

## Estructura del repositorio

```
.
├── 01_datos/
│   ├── demanda_ambulatoria_sintetica_2024_2025_100k.csv   # 100.000 registros brutos (34 MB)
│   ├── dataset_depurado_OE1.csv                           # 87.830 registros depurados (30 MB)
│   ├── diccionario_y_reglas_OE1.xlsx                      # diccionario y reglas de validación
│   ├── capacidad_agenda_sintetica.csv                     # capacidad semanal por zona (OE4–OE5)
│   ├── pqrs_sinteticas_2024_2025.csv                      # PQRS semanales por zona (OE5)
│   ├── panel_zona_semana_capacidad.csv                    # panel 104 semanas × 4 zonas (OE3)
│   └── README_dataset_sintetico.txt                       # ficha del generador
├── 02_codigo/
│   ├── 01_generar_datos_sinteticos.py
│   ├── 02_evaluar_calidad.py
│   ├── 03_limpiar_datos.py
│   ├── 04_analisis_exploratorio.py
│   ├── 05_analisis_series_y_asociacion.py
│   ├── 06_modelo_xgboost.py
│   ├── 07_figuras_oe2.py
│   ├── 08_figuras_fase3.py
│   ├── 09_panel_zonas_capacidad_pqrs.py
│   ├── 10_modelo_panel_zonas.py
│   ├── 11_simulacion_impacto_oe5.py
│   ├── 12_protocolo_oe4.py
│   ├── 13_figuras_oe4_oe5.py
│   └── panel_features.py                                  # ingeniería de características del panel
├── 03_resultados/
│   ├── figuras/                                           # figuras del documento (PNG, 200 ppp)
│   ├── tablas/                                            # tablas intermedias (CSV)
│   ├── modelos/
│   │   └── modelo_panel_xgboost.ubj                       # artefacto del modelo de panel (447,9 KB)
│   ├── metricas_calidad.json
│   ├── resultados_series_y_asociacion.json
│   ├── modelos_comparados.json
│   ├── resultados_modelo_xgboost.json
│   ├── modelo_xgboost.json                                # artefacto del modelo agregado
│   ├── resultados_modelo_panel.json
│   ├── resultados_sarima.json
│   ├── resultados_protocolo_oe4.json
│   ├── resultados_simulacion_oe5.json
│   ├── supuestos_capa_oe5.json
│   ├── predicciones_panel.csv
│   └── tabla_psi_deriva.csv
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
python 02_codigo/06_modelo_xgboost.py                  # XGBoost agregado, validación temporal y métricas (~3 min)
python 02_codigo/07_figuras_oe2.py                     # figuras del OE2
python 02_codigo/08_figuras_fase3.py                   # arquitectura, importancia y residuos
python 02_codigo/09_panel_zonas_capacidad_pqrs.py      # capa sintética de capacidad y PQRS, panel zona-semana
python 02_codigo/10_modelo_panel_zonas.py              # XGBoost de panel, referentes y SARIMA (~3 min)
python 02_codigo/11_simulacion_impacto_oe5.py          # escenarios de capacidad y sensibilidad (OE5)
python 02_codigo/12_protocolo_oe4.py                   # protocolo de operación y monitoreo (OE4)
python 02_codigo/13_figuras_oe4_oe5.py                 # figuras del OE4 y del OE5
```

El paso `01` puede omitirse si se parte de los archivos CSV ya incluidos en `01_datos/`. Los pasos `09` a `13` consumen la capa sintética de capacidad y PQRS y el panel zona-semana generados por `09`.

## Reproducibilidad

| Elemento | Valor |
|---|---|
| Semilla de generación de datos | `20261005` |
| Semilla de análisis y modelado | `2026` |
| Partición temporal (agregado) | 73 semanas de entrenamiento (2024-04-07 a 2025-08-24) y 18 de prueba (2025-08-31 a 2025-12-28) |
| Partición temporal (panel) | 104 semanas × 4 zonas = 416 observaciones; 18 semanas de prueba por zona (2025-08-25 a 2025-12-22); 364 utilizables tras el calentamiento |
| Validación cruzada | `TimeSeriesSplit`, 4 pliegues, ventana expansiva |
| Ventana de simulación (OE5) | 18 semanas de prueba del panel |
| Versiones de referencia | pandas 2.3.3, NumPy 2.3.4, scikit-learn 1.9.0, statsmodels 0.15.0, XGBoost 3.4.1, Matplotlib 3.10.7 |

Integridad de los datos (SHA-256):

```
1e8f4b08aca876f10890721fca0946e6c531b3f1739bd57cccae9c2eff7b2863  demanda_ambulatoria_sintetica_2024_2025_100k.csv
a94d19d09be4a68e7381d83d664ef22086691defbc8391262414a7748da28ef9  dataset_depurado_OE1.csv
9f0c23d19faf4d5713f837824eb839aa99b70beb57d84e2dc5c0eedaaf9a2c85  capacidad_agenda_sintetica.csv
2e4ea56eb150febeeb0d8f67013e8076fc2b76a2823aaf4940f5f1c4d83a3f26  pqrs_sinteticas_2024_2025.csv
78acccacf08a6a4ff19f9066a4439dc19deb871c1775ab57be8a05ffae1c043f  panel_zona_semana_capacidad.csv
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

Modelo de panel zona-semana (resultado principal del OE3), 18 semanas de prueba por zona:

| Modelo | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| **XGBoost de panel** | **13,7** | **16,6** | **6,93** | **0,965** |
| Naive estacional (t−52) | 16,2 | 20,7 | 7,91 | 0,945 |
| Media histórica por zona | 16,4 | 20,0 | 7,83 | 0,949 |
| SARIMA(1,1,1)(0,1,0,52), serie agregada | 32,6 | 43,7 | 3,69 | 0,395 |

El modelo de panel mejora el MAE del naive estacional en 15,4 % y su MAPE en 12,4 %. Gana en Centro, Norte y Sur y pierde solo en Occidente, la zona de menor volumen. Artefacto: `03_resultados/modelos/modelo_panel_xgboost.ubj` (400 árboles, 2.632 nodos, 447,9 KB).

Protocolo de operación (OE4): latencia de inferencia de 1,04 a 7,79 ms por lote, 35 variables monitoreadas con PSI, de las cuales 12 en alerta y 2 en nivel de reentrenamiento.

Simulación de impacto (OE5), ventana de 18 semanas del panel:

| Escenario | No atendidas | PQRS esperadas | Reducción (%) |
|---|---|---|---|
| Base | 941 | 266,1 | — |
| Reasignación (límite 10 %) | 936 | 265,3 | 0,3 |
| +3 % dirigido por el pronóstico | 638 | 184,2 | 30,8 |
| +3 % proporcional | 634 | 182,6 | 31,4 |
| Oráculo: reasignación perfecta | — | 239,6 | 9,9 |
| Oráculo: dirigido perfecto | — | 156,3 | 41,3 |

El nivel de capacidad domina el impacto; la reasignación sola no aporta porque el error de composición del pronóstico (4,35 % en norma L1) es similar al de la capacidad vigente (4,20 %). La relación PQRS↔no atendidas se estimó con OLS-HC1: b = 0,2727 [0,2569; 0,2885], R² = 0,795, coherente con el parámetro 0,28 del generador. Los supuestos están en `03_resultados/supuestos_capa_oe5.json`.

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
