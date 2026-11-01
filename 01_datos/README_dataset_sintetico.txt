DATASET SINTÉTICO — ACA 1 PROYECTO DE GRADO (VALIDACIÓN METODOLÓGICA OE1)
=====================================================================

Archivo: demanda_ambulatoria_sintetica_2024_2025_100k.csv
Registros: 100,000
Variables: 39
Periodo simulado: enero 2024 - diciembre 2025
Semilla de generación: 20261005 (reproducible con 01_generar_datos_sinteticos.py)

IMPORTANTE — DECLARACIÓN ACADÉMICA:
Este conjunto de datos es completamente SINTÉTICO y fue generado con fines
exclusivamente académicos y de validación metodológica. NO corresponde a
registros reales de la Subred Integrada de Servicios de Salud Sur Occidente
E.S.E. ni de ninguna otra institución. No contiene nombres, documentos,
teléfonos ni ningún identificador de personas reales. Los resultados obtenidos
sobre este conjunto no se presentan como evidencia del comportamiento real de
la población atendida.

PROPÓSITO:
Validar el desarrollo del Objetivo Específico 1 del proyecto: evaluación de
calidad, completitud y representatividad de registros de demanda ambulatoria,
definición de criterios de inclusión/exclusión y preparación del conjunto de
datos para las fases posteriores (caracterización prescriptiva y modelado
predictivo con XGBoost).

PROBLEMAS DE CALIDAD INYECTADOS DE FORMA CONTROLADA:
- 800 registros duplicados exactos;
- valores faltantes en edad, sexo, localidad, EPS, servicio, CIE-10, tipo de
  orden, estrato y tiempo de espera;
- 300 edades fuera de rango;
- 500 fechas fuera del periodo 2024-2025 (con campos derivados inconsistentes);
- 700 códigos CIE-10 no válidos;
- 600 categorías de servicio inconsistentes;
- 250 tiempos de espera atípicos;
- 250 registros sin identificador de profesional.

Los porcentajes y cantidades exactas de cada problema se calculan con el
script 02_evaluar_calidad.py y se reportan en el documento de grado.
