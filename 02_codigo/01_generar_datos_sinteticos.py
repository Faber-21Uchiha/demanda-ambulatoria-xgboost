# -*- coding: utf-8 -*-
"""
01_generar_datos_sinteticos.py
Generador del conjunto de datos sintetico para la validacion metodologica del
Objetivo Especifico 1 del proyecto de grado (ACA 1 - Proyecto 2).

ADVERTENCIA ACADEMICA:
Este conjunto de datos es COMPLETAMENTE SINTETICO. No corresponde a registros
reales de la Subred Integrada de Servicios de Salud Sur Occidente E.S.E. ni de
ninguna institucion. Se construyo con fines exclusivamente academicos, para
validar de manera reproducible el instrumento de evaluacion de calidad de datos
disenado en el OE1, mientras se obtiene la autorizacion de acceso a las fuentes
institucionales. No contiene nombres, documentos ni identificadores reales.

Semilla de generacion: 20261005 (garantiza reproducibilidad exacta).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20261005
N_TOTAL = 100_000
N_UNICOS = 99_200          # registros unicos antes de inyectar duplicados
N_DUP = N_TOTAL - N_UNICOS  # 800 duplicados exactos inyectados de forma controlada

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. Catalogos sinteticos
# ----------------------------------------------------------------------------
LOCALIDADES = ["Kennedy", "Bosa", "Fontibón", "Puente Aranda"]
P_LOCALIDAD = [0.38, 0.31, 0.18, 0.13]

SEXOS = ["F", "M", "Otro"]
P_SEXO = [0.54, 0.455, 0.005]

REGIMENES = ["Contributivo", "Subsidiado"]
P_REGIMEN = [0.58, 0.42]

ESTRATOS = ["1", "2", "3", "4"]
P_ESTRATO = [0.18, 0.45, 0.31, 0.06]

SERVICIOS = [
    "Medicina general", "Enfermería", "Odontología", "Pediatría",
    "Medicina interna", "Medicina familiar", "Ginecología",
    "Psicología", "Nutrición", "Laboratorio", "Imágenes diagnósticas",
]
P_SERVICIO = [0.34, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.04, 0.02]

ESPECIALIDAD_MAP = {
    "Medicina general": "Medicina general",
    "Enfermería": "Enfermería",
    "Odontología": "Odontología",
    "Pediatría": "Pediatría",
    "Medicina interna": "Medicina interna",
    "Medicina familiar": "Medicina familiar",
    "Ginecología": "Ginecología",
    "Psicología": "Psicología",
    "Nutrición": "Nutrición",
    "Laboratorio": "Apoyo diagnóstico",
    "Imágenes diagnósticas": "Apoyo diagnóstico",
}

NIVELES = ["Primer nivel", "Segundo nivel"]
P_NIVEL = [0.72, 0.28]

MODALIDADES = ["Presencial", "Teleconsulta"]
P_MODALIDAD = [0.93, 0.07]

JORNADAS = ["Mañana", "Tarde"]
P_JORNADA = [0.61, 0.39]

TIPOS_ORDEN = ["Medicamento", "Laboratorio", "Imagen diagnóstica", "Interconsulta", "Procedimiento", "Control"]
P_TIPO_ORDEN = [0.34, 0.25, 0.10, 0.09, 0.12, 0.10]

# Catalogo CIE-10 de referencia (20 codigos validos con descripcion)
CATALOGO_CIE10 = pd.DataFrame([
    ("I10",   "Hipertensión esencial (primaria)"),
    ("E11.9", "Diabetes mellitus tipo 2 sin complicaciones"),
    ("J06.9", "Infección aguda de las vías respiratorias superiores, no especificada"),
    ("J45.9", "Asma, no especificada"),
    ("M54.5", "Dolor lumbar"),
    ("K21.9", "Enfermedad por reflujo gastroesofágico sin esofagitis"),
    ("N39.0", "Infección de vías urinarias, sitio no especificado"),
    ("R51.9", "Cefalea, no especificada"),
    ("Z00.0", "Examen médico general"),
    ("Z23",   "Necesidad de inmunización contra enfermedades infecciosas"),
    ("E66.9", "Obesidad, no especificada"),
    ("F41.9", "Trastorno de ansiedad, no especificado"),
    ("K02.9", "Caries dental, no especificada"),
    ("J18.9", "Neumonía, no especificada"),
    ("G43.9", "Migraña, no especificada"),
    ("M25.5", "Dolor articular"),
    ("R10.9", "Dolor abdominal, no especificado"),
    ("L20.9", "Dermatitis atópica, no especificada"),
    ("A09",   "Gastroenteritis y colitis de origen no especificado"),
    ("Z71.1", "Persona con temor a una queja para la que no se hace diagnóstico"),
], columns=["codigo_cie10", "diagnostico_categoria"])

P_CIE = np.array([0.09, 0.07, 0.10, 0.04, 0.07, 0.06, 0.06, 0.06, 0.09, 0.05,
                  0.05, 0.04, 0.06, 0.03, 0.03, 0.05, 0.05, 0.03, 0.04, 0.03])
P_CIE = P_CIE / P_CIE.sum()

SEDES_POR_LOCALIDAD = {
    "Kennedy": ["Sede Kennedy Central", "Centro de Atención Kennedy"],
    "Bosa": ["Sede Bosa Centro", "Centro de Atención Bosa"],
    "Fontibón": ["Sede Fontibón", "Centro de Atención Fontibón"],
    "Puente Aranda": ["Sede Puente Aranda", "Centro de Atención Puente Aranda"],
}

DIAS_ES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
           4: "Viernes", 5: "Sábado", 6: "Domingo"}

PROFESIONALES = [f"PROF{i:04d}" for i in range(1, 181)]
EPS_LIST = ["EPS001", "EPS002", "EPS003", "EPS004", "EPS005"]
P_EPS = [0.28, 0.24, 0.20, 0.16, 0.12]

# ----------------------------------------------------------------------------
# 2. Fechas 2024-2025 con estacionalidad mensual y efecto fin de semana
# ----------------------------------------------------------------------------
fechas = pd.date_range("2024-01-01", "2025-12-31", freq="D")
peso_mes = np.array([0.90, 0.92, 1.00, 1.02, 1.05, 1.00,
                     0.98, 1.02, 1.05, 1.08, 1.03, 0.92])
peso_diario = np.array([
    peso_mes[d.month - 1] * (0.58 if d.weekday() >= 5 else 1.0) for d in fechas
])
peso_diario /= peso_diario.sum()

idx_fecha = rng.choice(len(fechas), size=N_UNICOS, p=peso_diario)
fecha = fechas[idx_fecha]

# ----------------------------------------------------------------------------
# 3. Generacion de los 99.200 registros unicos
# ----------------------------------------------------------------------------
localidad = rng.choice(LOCALIDADES, N_UNICOS, p=P_LOCALIDAD)
sexo = rng.choice(SEXOS, N_UNICOS, p=P_SEXO)
regimen = rng.choice(REGIMENES, N_UNICOS, p=P_REGIMEN)
estrato = rng.choice(ESTRATOS, N_UNICOS, p=P_ESTRATO)

# Edad: mezcla de subpoblaciones pediatrica, adulta y adulta mayor
comp = rng.choice([0, 1, 2], N_UNICOS, p=[0.14, 0.71, 0.15])
edad = np.empty(N_UNICOS)
edad[comp == 0] = rng.integers(0, 18, (comp == 0).sum())
edad[comp == 1] = np.clip(rng.normal(43, 16, (comp == 1).sum()), 18, 79)
edad[comp == 2] = np.clip(rng.normal(70, 7, (comp == 2).sum()), 60, 95)

servicio = rng.choice(SERVICIOS, N_UNICOS, p=P_SERVICIO)
especialidad = np.array([ESPECIALIDAD_MAP[s] for s in servicio])
nivel = rng.choice(NIVELES, N_UNICOS, p=P_NIVEL)
modalidad = rng.choice(MODALIDADES, N_UNICOS, p=P_MODALIDAD)
jornada = rng.choice(JORNADAS, N_UNICOS, p=P_JORNADA)
tipo_orden = rng.choice(TIPOS_ORDEN, N_UNICOS, p=P_TIPO_ORDEN)

cie_idx = rng.choice(len(CATALOGO_CIE10), N_UNICOS, p=P_CIE)
codigo_cie = CATALOGO_CIE10.iloc[cie_idx]["codigo_cie10"].to_numpy()
diag_cat = CATALOGO_CIE10.iloc[cie_idx]["diagnostico_categoria"].to_numpy()

sede = np.array([rng.choice(SEDES_POR_LOCALIDAD[l]) for l in localidad])
eps = rng.choice(EPS_LIST, N_UNICOS, p=P_EPS)
id_prof = rng.choice(PROFESIONALES, N_UNICOS)

cantidad_ordenes = rng.poisson(1.35, N_UNICOS) + 1
atencion_programada = rng.choice(["Sí", "No"], N_UNICOS, p=[0.82, 0.18])
cancelada = rng.choice(["No", "Sí"], N_UNICOS, p=[0.93, 0.07])
inasistencia = rng.choice(["No", "Sí"], N_UNICOS, p=[0.91, 0.09])
tiempo_espera = np.clip(rng.gamma(3.0, 12.0, N_UNICOS), 2, 240).round()
cantidad_servicios = rng.choice([1, 2, 3], N_UNICOS, p=[0.83, 0.14, 0.03])
requiere_control = rng.choice(["Sí", "No"], N_UNICOS, p=[0.46, 0.54])
remision = rng.choice(["Sí", "No"], N_UNICOS, p=[0.12, 0.88])

df = pd.DataFrame({
    "id_registro": [f"REG{i:06d}" for i in range(1, N_UNICOS + 1)],
    "id_usuario": [f"USR{int(v):06d}" for v in rng.integers(1, 45001, N_UNICOS)],
    "id_profesional": id_prof,
    "id_sede": [f"SED{int(v):03d}" for v in rng.integers(1, 9, N_UNICOS)],
    "fecha_atencion": fecha,
    "anio": fecha.year,
    "mes": fecha.month,
    "semana_anio": fecha.isocalendar().week.astype(int).to_numpy(),
    "dia_semana": [DIAS_ES[d.weekday()] for d in fecha],
    "trimestre": ["T" + str(((m - 1) // 3) + 1) for m in fecha.month],
    "jornada": jornada,
    "edad": edad.astype(float),
    "grupo_edad": pd.cut(edad, bins=[-1, 5, 11, 17, 28, 44, 59, 74, 200],
                         labels=["0-5", "6-11", "12-17", "18-28", "29-44",
                                 "45-59", "60-74", "75+"]).astype(str),
    "sexo": sexo,
    "localidad_residencia": localidad,
    "estrato_socioeconomico": estrato,
    "regimen": regimen,
    "eps": eps,
    "tipo_atencion": "Consulta externa",
    "servicio": servicio,
    "especialidad": especialidad,
    "nivel_atencion": nivel,
    "modalidad_atencion": modalidad,
    "sede": sede,
    "codigo_cie10": codigo_cie,
    "diagnostico_categoria": diag_cat,
    "tipo_diagnostico": rng.choice(["Principal", "Relacionado"], N_UNICOS, p=[0.86, 0.14]),
    "prioridad": rng.choice(["Rutinaria", "Preferente"], N_UNICOS, p=[0.88, 0.12]),
    "tipo_orden": tipo_orden,
    "categoria_orden": tipo_orden,
    "cantidad_ordenes": cantidad_ordenes,
    "requiere_control": requiere_control,
    "remision": remision,
    "cantidad_servicios": cantidad_servicios,
    "atencion_programada": atencion_programada,
    "atencion_realizada": np.where(cancelada == "Sí", "No", "Sí"),
    "cancelada": cancelada,
    "inasistencia": inasistencia,
    "tiempo_espera_min": tiempo_espera.astype(float),
})

# ----------------------------------------------------------------------------
# 4. Inyeccion controlada de problemas de calidad
# ----------------------------------------------------------------------------
# 4.1 Duplicados exactos (800 copias de registros existentes)
dup_idx = rng.choice(df.index, size=N_DUP, replace=False)
df = pd.concat([df, df.loc[dup_idx].copy()], ignore_index=True)

# 4.2 Valores faltantes en variables criticas y sociodemograficas
tasas_nulos = {
    "edad": 0.025, "sexo": 0.012, "localidad_residencia": 0.018,
    "eps": 0.015, "servicio": 0.010, "codigo_cie10": 0.012,
    "tipo_orden": 0.014, "tiempo_espera_min": 0.025,
    "estrato_socioeconomico": 0.035,
}
for col, tasa in tasas_nulos.items():
    idx_n = rng.choice(df.index, size=int(N_TOTAL * tasa), replace=False)
    df.loc[idx_n, col] = np.nan

# 4.3 Edades fuera de rango (negativas o superiores a 120 anos)
idx_edad = rng.choice(df.index, size=300, replace=False)
df.loc[idx_edad, "edad"] = rng.choice([-5, -1, 121, 150], size=300)

# 4.4 Fechas fuera del periodo de estudio (2022-2023).
#     Se actualizan anio y mes, pero NO los campos derivados semana_anio,
#     dia_semana ni trimestre: se genera una inconsistencia interna realista.
idx_fecha_mala = rng.choice(df.index, size=500, replace=False)
fechas_malas = pd.to_datetime(rng.choice(pd.date_range("2022-01-01", "2023-12-31", freq="D"), size=500))
df.loc[idx_fecha_mala, "fecha_atencion"] = fechas_malas
df.loc[idx_fecha_mala, "anio"] = fechas_malas.year
df.loc[idx_fecha_mala, "mes"] = fechas_malas.month

# 4.5 Codigos CIE-10 no validos (fuera del catalogo de referencia)
idx_cie = rng.choice(df.index, size=700, replace=False)
df.loc[idx_cie, "codigo_cie10"] = rng.choice(
    ["XXX.99", "ABC123", "I9999", "Z99.99", "NO_VALIDO"], size=700)

# 4.6 Categorias de servicio inconsistentes (mayusculas, espacios, guiones)
idx_serv = rng.choice(df.index, size=600, replace=False)
df.loc[idx_serv, "servicio"] = rng.choice(
    ["Med. General", "medicina general ", "MEDICINA GENERAL", "Medicina-General"], size=600)

# 4.7 Tiempos de espera atipicos (0 o superiores al rango plausible de 240 min)
idx_wait = rng.choice(df.index, size=250, replace=False)
df.loc[idx_wait, "tiempo_espera_min"] = rng.choice([0, 500, 720, 999], size=250)

# 4.8 Identificador de profesional faltante (afecta integridad)
idx_prof = rng.choice(df.index, size=250, replace=False)
df.loc[idx_prof, "id_profesional"] = np.nan

# Ordenar por fecha y exportar
df = df.sort_values(["fecha_atencion", "id_registro"], na_position="last").reset_index(drop=True)
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"]).dt.strftime("%Y-%m-%d")

# ----------------------------------------------------------------------------
# 5. Exportacion
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
carpeta_datos = BASE / "01_datos"
carpeta_datos.mkdir(parents=True, exist_ok=True)
ruta_csv = carpeta_datos / "demanda_ambulatoria_sintetica_2024_2025_100k.csv"
df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

# 5.1 Diccionario de datos y reglas de calidad (instrumento del OE1)
DESCRIPCIONES = {
    "id_registro": ("Texto", "Identificador tecnico sintetico del registro.", "Identificacion"),
    "id_usuario": ("Texto", "Identificador sintetico no identificable del usuario atendido.", "Identificacion"),
    "id_profesional": ("Texto", "Identificador sintetico del profesional que genera la orden.", "Integridad"),
    "id_sede": ("Texto", "Identificador sintetico de la sede de atencion.", "Identificacion"),
    "fecha_atencion": ("Fecha", "Fecha de la atencion ambulatoria (AAAA-MM-DD).", "Temporalidad"),
    "anio": ("Entero", "Ano derivado de la fecha de atencion.", "Temporalidad"),
    "mes": ("Entero", "Mes derivado de la fecha de atencion.", "Temporalidad"),
    "semana_anio": ("Entero", "Semana calendario ISO derivada de la fecha.", "Temporalidad"),
    "dia_semana": ("Texto", "Dia de la semana de la atencion.", "Temporalidad"),
    "trimestre": ("Texto", "Trimestre calendario derivado de la fecha.", "Temporalidad"),
    "jornada": ("Texto", "Jornada de atencion (Manana o Tarde).", "Temporalidad"),
    "edad": ("Numerica", "Edad del usuario en anos cumplidos.", "Sociodemografica"),
    "grupo_edad": ("Texto", "Grupo etario derivado de la edad.", "Sociodemografica"),
    "sexo": ("Texto", "Sexo registrado del usuario (F, M, Otro).", "Sociodemografica"),
    "localidad_residencia": ("Texto", "Localidad de residencia del usuario.", "Representatividad"),
    "estrato_socioeconomico": ("Texto", "Estrato socioeconomico de la residencia (1 a 4).", "Sociodemografica"),
    "regimen": ("Texto", "Regimen de afiliacion al SGSSS.", "Sociodemografica"),
    "eps": ("Texto", "Identificador sintetico de la EPS contratante.", "Contratacion"),
    "tipo_atencion": ("Texto", "Tipo general de la atencion.", "Atencion"),
    "servicio": ("Texto", "Servicio ambulatorio prestado.", "Atencion"),
    "especialidad": ("Texto", "Especialidad o area del servicio.", "Atencion"),
    "nivel_atencion": ("Texto", "Nivel de atencion (primer o segundo nivel).", "Atencion"),
    "modalidad_atencion": ("Texto", "Modalidad de la atencion (Presencial o Teleconsulta).", "Atencion"),
    "sede": ("Texto", "Nombre sintetico de la sede.", "Atencion"),
    "codigo_cie10": ("Texto", "Codigo diagnostico CIE-10 del registro.", "Clinica"),
    "diagnostico_categoria": ("Texto", "Descripcion del diagnostico asociada al codigo.", "Clinica"),
    "tipo_diagnostico": ("Texto", "Clasificacion del diagnostico (Principal o Relacionado).", "Clinica"),
    "prioridad": ("Texto", "Prioridad de la atencion (Rutinaria o Preferente).", "Atencion"),
    "tipo_orden": ("Texto", "Tipo de orden medica generada.", "Ordenes"),
    "categoria_orden": ("Texto", "Categoria funcional de la orden.", "Ordenes"),
    "cantidad_ordenes": ("Entero", "Numero de ordenes generadas en el registro.", "Ordenes"),
    "requiere_control": ("Texto", "Indica si la atencion requiere control posterior.", "Ordenes"),
    "remision": ("Texto", "Indica si se genero remision.", "Ordenes"),
    "cantidad_servicios": ("Entero", "Numero de servicios asociados al registro.", "Demanda"),
    "atencion_programada": ("Texto", "Indica si la atencion fue programada.", "Demanda"),
    "atencion_realizada": ("Texto", "Indica si la atencion se realizo efectivamente.", "Demanda"),
    "cancelada": ("Texto", "Indica si la atencion fue cancelada.", "Demanda"),
    "inasistencia": ("Texto", "Indica si hubo inasistencia del usuario.", "Demanda"),
    "tiempo_espera_min": ("Numerica", "Tiempo de espera en minutos.", "Calidad/oportunidad"),
}

DOMINIOS = {
    "sexo": "F | M | Otro",
    "localidad_residencia": "Kennedy | Bosa | Fontibón | Puente Aranda",
    "regimen": "Contributivo | Subsidiado",
    "estrato_socioeconomico": "1 | 2 | 3 | 4",
    "servicio": "Medicina general | Enfermería | Odontología | Pediatría | Medicina interna | Medicina familiar | Ginecología | Psicología | Nutrición | Laboratorio | Imágenes diagnósticas",
    "nivel_atencion": "Primer nivel | Segundo nivel",
    "modalidad_atencion": "Presencial | Teleconsulta",
    "jornada": "Mañana | Tarde",
    "tipo_orden": "Medicamento | Laboratorio | Imagen diagnóstica | Interconsulta | Procedimiento | Control",
    "tipo_diagnostico": "Principal | Relacionado",
    "prioridad": "Rutinaria | Preferente",
    "grupo_edad": "0-5 | 6-11 | 12-17 | 18-28 | 29-44 | 45-59 | 60-74 | 75+",
    "eps": "EPS001 ... EPS005",
    "id_profesional": "PROF0001 ... PROF0180",
    "edad": "0 a 120 años",
    "tiempo_espera_min": "1 a 240 minutos (rango plausible)",
    "fecha_atencion": "2024-01-01 a 2025-12-31",
}

diccionario = pd.DataFrame([
    {
        "variable": v,
        "tipo_dato": d[0],
        "descripcion": d[1],
        "uso_en_el_proyecto": d[2],
        "dominio_esperado": DOMINIOS.get(v, "—"),
    }
    for v, d in DESCRIPCIONES.items()
])

reglas = pd.DataFrame([
    ["Completitud", "Valores faltantes", "Porcentaje de nulos por variable y por registro", "Detectar variables e registros con informacion incompleta"],
    ["Unicidad", "Duplicados", "Registros identicos en todas las columnas (comparacion de contenido completo)", "Eliminar duplicados conservando la primera aparicion"],
    ["Validez", "CIE-10", "Pertenencia del codigo al catalogo de referencia", "Excluir registros con codigo no valido"],
    ["Validez", "Edad", "Rango 0 a 120 anos", "Excluir registros con edad imposible o sin edad"],
    ["Consistencia", "Fecha", "Fecha dentro del periodo enero 2024 - diciembre 2025", "Excluir registros fuera del periodo"],
    ["Consistencia", "Campos derivados", "Coherencia entre fecha y anio/mes/semana", "Recalcular campos derivados"],
    ["Consistencia", "Categorias", "Pertenencia a los dominios definidos", "Homologar categorias inconsistentes"],
    ["Integridad", "Profesional", "Identificador del profesional presente", "Excluir registros sin profesional"],
    ["Integridad", "Campos criticos", "Presencia de fecha, servicio, diagnostico, localidad, EPS, tipo de orden y sexo", "Excluir registros sin campos criticos"],
    ["Representatividad", "Localidad", "Distribucion de registros por localidad", "Verificar cobertura territorial"],
    ["Representatividad", "Servicio", "Distribucion de registros por servicio", "Verificar cobertura asistencial"],
    ["Representatividad", "Diagnostico", "Distribucion de codigos CIE-10", "Verificar diversidad diagnostica"],
    ["Representatividad", "Profesional", "Numero de profesionales con registros", "Verificar variedad prescriptiva"],
    ["Temporalidad", "Demanda", "Registros por mes y por semana", "Verificar cobertura y estacionalidad"],
    ["Atipicos", "Tiempo de espera", "Valores fuera del rango plausible (0 o > 240 min)", "Anular valores atipicos (tratamiento, no exclusion)"],
], columns=["dimension", "elemento", "indicador", "tratamiento"])

carpeta_doc = BASE / "04_documentacion"
carpeta_doc.mkdir(parents=True, exist_ok=True)
ruta_xlsx = carpeta_doc / "diccionario_y_reglas_OE1.xlsx"
with pd.ExcelWriter(ruta_xlsx, engine="openpyxl") as writer:
    diccionario.to_excel(writer, sheet_name="Diccionario", index=False)
    CATALOGO_CIE10.to_excel(writer, sheet_name="Catalogo_CIE10", index=False)
    reglas.to_excel(writer, sheet_name="Reglas_calidad", index=False)

readme = f"""DATASET SINTÉTICO — ACA 1 PROYECTO DE GRADO (VALIDACIÓN METODOLÓGICA OE1)
=====================================================================

Archivo: demanda_ambulatoria_sintetica_2024_2025_100k.csv
Registros: {len(df):,}
Variables: {len(df.columns)}
Periodo simulado: enero 2024 - diciembre 2025
Semilla de generación: {SEED} (reproducible con 01_generar_datos_sinteticos.py)

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
- {N_DUP} registros duplicados exactos;
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
"""
ruta_readme = carpeta_datos / "README_dataset_sintetico.txt"
ruta_readme.write_text(readme, encoding="utf-8")

# ----------------------------------------------------------------------------
# 6. Verificacion
# ----------------------------------------------------------------------------
print("VERIFICACION DEL GENERADOR")
print(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
print(f"Duplicados exactos: {df.duplicated().sum():,}")
print(f"Celdas nulas: {int(df.isna().sum().sum()):,}")
f = pd.to_datetime(df["fecha_atencion"], errors="coerce")
print(f"Fechas fuera de 2024-2025: {(~f.between('2024-01-01','2025-12-31')).sum():,}")
print(f"CIE-10 fuera de catalogo: {(~df['codigo_cie10'].isin(CATALOGO_CIE10['codigo_cie10'])).sum():,}")
print(f"Archivos generados:\n - {ruta_csv}\n - {ruta_xlsx}\n - {ruta_readme}")
