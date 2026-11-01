# -*- coding: utf-8 -*-
"""
02_evaluar_calidad.py
Evaluacion de calidad, completitud y representatividad del conjunto de datos
(Ojetivo Especifico 1 - ACA 1). Lee el dataset sintetico bruto, aplica la ficha
de evaluacion de calidad (instrumento del OE1) y produce:
  - tablas/tabla_completitud_variables.csv
  - tablas/tabla_dimensiones_calidad.csv
  - tablas/tabla_distribucion_mensual.csv
  - graficas/fig_completitud.png
  - graficas/fig_registros_mes_brutos.png
  - metricas.json (insumo para el documento de grado)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
R2 = BASE / "03_resultados"
GRAF = R2 / "graficas"
TABLAS = R2 / "tablas"
GRAF.mkdir(parents=True, exist_ok=True)
TABLAS.mkdir(parents=True, exist_ok=True)

# Paleta institucional CUN
VERDE = "#84BD00"
VERDE_OSCURO = "#007A33"
AZUL = "#0C2340"
TEAL = "#00859B"
GRIS = "#898D8D"
plt.rcParams.update({
    "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 9, "axes.edgecolor": "#666666",
})

df = pd.read_csv(BASE / "01_datos" / "demanda_ambulatoria_sintetica_2024_2025_100k.csv",
                 encoding="utf-8-sig")
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"], errors="coerce")

catalogo = pd.read_excel(BASE / "04_documentacion" / "diccionario_y_reglas_OE1.xlsx",
                         sheet_name="Catalogo_CIE10")
codigos_validos = set(catalogo["codigo_cie10"])

n_total = len(df)
m = {}  # metricas para el documento

# ---------------------------------------------------------------------------
# 1. Completitud
# ---------------------------------------------------------------------------
completitud = pd.DataFrame({
    "variable": df.columns,
    "tipo_dato": [str(df[c].dtype) for c in df.columns],
    "nulos": [int(df[c].isna().sum()) for c in df.columns],
})
completitud["pct_nulos"] = (completitud["nulos"] / n_total * 100).round(2)
completitud["pct_completitud"] = (100 - completitud["pct_nulos"]).round(2)
completitud["valores_unicos"] = [int(df[c].nunique()) for c in df.columns]
completitud = completitud.sort_values("pct_completitud")
completitud.to_csv(TABLAS / "tabla_completitud_variables.csv", index=False, encoding="utf-8-sig")

registros_completos = int(df.notna().all(axis=1).sum())
celdas_totales = int(df.size)
celdas_nulas = int(df.isna().sum().sum())
m["n_registros"] = n_total
m["n_variables"] = len(df.columns)
m["celdas_totales"] = celdas_totales
m["celdas_nulas"] = celdas_nulas
m["pct_celdas_completas"] = round(100 * (celdas_totales - celdas_nulas) / celdas_totales, 2)
m["registros_completos"] = registros_completos
m["pct_registros_completos"] = round(100 * registros_completos / n_total, 2)
vars_con_nulos = completitud[completitud["nulos"] > 0]
m["n_variables_con_nulos"] = len(vars_con_nulos)
m["variables_con_nulos"] = {r.variable: float(r.pct_nulos) for r in vars_con_nulos.itertuples()}

# Figura: completitud por variable
d2 = completitud.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
colores = [VERDE_OSCURO if v >= 98 else (VERDE if v >= 95 else "#E8A33D" if v >= 90 else "#C0392B")
           for v in d2["pct_completitud"]]
ax.barh(d2["variable"], d2["pct_completitud"], color=colores, height=0.62)
for i, v in enumerate(d2["pct_completitud"]):
    ax.text(v - 1.2, i, f"{v:.2f}%", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
ax.set_xlim(0, 100)
ax.set_xlabel("Porcentaje de completitud (%)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(GRAF / "fig_completitud.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Unicidad (duplicados)
# ---------------------------------------------------------------------------
dups_exactos = int(df.duplicated().sum())
ids_repetidos = int(df["id_registro"].duplicated(keep=False).sum())
m["duplicados_exactos"] = dups_exactos
m["pct_duplicados_exactos"] = round(100 * dups_exactos / n_total, 2)
m["registros_con_id_repetido"] = ids_repetidos

# ---------------------------------------------------------------------------
# 3. Validez
# ---------------------------------------------------------------------------
cie_nulos = int(df["codigo_cie10"].isna().sum())
cie_invalidos = int((~df["codigo_cie10"].isna() & ~df["codigo_cie10"].isin(codigos_validos)).sum())
m["cie_nulos"] = cie_nulos
m["cie_invalidos"] = cie_invalidos
m["cie_validos"] = int(df["codigo_cie10"].isin(codigos_validos).sum())
m["pct_cie_validos"] = round(100 * m["cie_validos"] / n_total, 2)

edad_nula = int(df["edad"].isna().sum())
edad_invalida = int(((df["edad"] < 0) | (df["edad"] > 120)).sum())
m["edad_nula"] = edad_nula
m["edad_invalida"] = edad_invalida
m["pct_edad_valida"] = round(100 * df["edad"].between(0, 120).sum() / n_total, 2)

espera_atipica = int(((df["tiempo_espera_min"] == 0) | (df["tiempo_espera_min"] > 240)).sum())
m["espera_atipica"] = espera_atipica

# ---------------------------------------------------------------------------
# 4. Consistencia
# ---------------------------------------------------------------------------
fuera_periodo = int((~df["fecha_atencion"].between("2024-01-01", "2025-12-31")).sum())
m["fuera_periodo"] = fuera_periodo
m["pct_en_periodo"] = round(100 * (n_total - fuera_periodo) / n_total, 2)

# inconsistencia fecha vs campos derivados (anio/mes)
mask_der = df["fecha_atencion"].notna()
inconsistente_anio = int((df.loc[mask_der, "fecha_atencion"].dt.year != df.loc[mask_der, "anio"]).sum())
inconsistente_mes = int((df.loc[mask_der, "fecha_atencion"].dt.month != df.loc[mask_der, "mes"]).sum())
m["inconsistente_anio"] = inconsistente_anio
m["inconsistente_mes"] = inconsistente_mes

servicios_canonicos = ["Medicina general", "Enfermería", "Odontología", "Pediatría",
                       "Medicina interna", "Medicina familiar", "Ginecología",
                       "Psicología", "Nutrición", "Laboratorio", "Imágenes diagnósticas"]
serv_inconsistentes = int((~df["servicio"].isin(servicios_canonicos)).sum())
serv_nulos = int(df["servicio"].isna().sum())
m["servicio_inconsistente"] = serv_inconsistentes - serv_nulos
m["servicio_nulos"] = serv_nulos

# ---------------------------------------------------------------------------
# 5. Integridad (campos criticos)
# ---------------------------------------------------------------------------
campos_criticos = ["fecha_atencion", "servicio", "codigo_cie10", "localidad_residencia",
                   "eps", "tipo_orden", "id_profesional", "sexo", "edad"]
criticos_faltantes = {c: int(df[c].isna().sum()) for c in campos_criticos}
m["campos_criticos_faltantes"] = criticos_faltantes
m["profesional_nulo"] = criticos_faltantes["id_profesional"]
registros_sin_criticos = int((df[campos_criticos].isna().any(axis=1)).sum())
m["registros_sin_algun_campo_critico"] = registros_sin_criticos

# ---------------------------------------------------------------------------
# 6. Temporalidad
# ---------------------------------------------------------------------------
m["fecha_min"] = str(df["fecha_atencion"].min().date())
m["fecha_max"] = str(df["fecha_atencion"].max().date())
en_periodo = df[df["fecha_atencion"].between("2024-01-01", "2025-12-31")]
meses_cubiertos = en_periodo["fecha_atencion"].dt.to_period("M").nunique()
m["meses_cubiertos"] = int(meses_cubiertos)
m["registros_2024"] = int((en_periodo["fecha_atencion"].dt.year == 2024).sum())
m["registros_2025"] = int((en_periodo["fecha_atencion"].dt.year == 2025).sum())

dist_mensual = (en_periodo.groupby(en_periodo["fecha_atencion"].dt.to_period("M")).size()
               .rename("registros").reset_index())
dist_mensual.columns = ["mes", "registros"]
dist_mensual["mes"] = dist_mensual["mes"].astype(str)
dist_mensual.to_csv(TABLAS / "tabla_distribucion_mensual.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.5, 3.4))
colores_a = [TEAL] * 12 + [AZUL] * 12
ax.bar(dist_mensual["mes"], dist_mensual["registros"], color=colores_a, width=0.72)
ax.set_xticks(range(len(dist_mensual)))
ax.set_xticklabels([s[2:] for s in dist_mensual["mes"]], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Registros")
ax.set_xlabel("Mes (2024 en teal, 2025 en azul)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(GRAF / "fig_registros_mes_brutos.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Representatividad
# ---------------------------------------------------------------------------
rep = {}
rep["localidades"] = df["localidad_residencia"].value_counts(dropna=False).to_dict()
rep["servicios_top"] = df["servicio"].value_counts(dropna=False).head(11).to_dict()
rep["cie_top"] = df["codigo_cie10"].value_counts(dropna=False).head(10).to_dict()
rep["n_profesionales"] = int(df["id_profesional"].nunique())
rep["n_usuarios"] = int(df["id_usuario"].nunique())
rep["n_eps"] = int(df["eps"].nunique())
rep["n_cie_unicos"] = int(df["codigo_cie10"].nunique())
m["representatividad"] = rep
m["n_servicios_canonicos_presentes"] = int(df["servicio"].isin(servicios_canonicos).sum())

# ---------------------------------------------------------------------------
# 8. Tabla resumen por dimensiones (ficha de calidad aplicada)
# ---------------------------------------------------------------------------
dimensiones = pd.DataFrame([
    ["Completitud", "Porcentaje de celdas completas", f"{m['pct_celdas_completas']:.2f}%", ">= 90% por variable"],
    ["Completitud", "Registros sin ningun valor nulo", f"{m['registros_completos']:,} ({m['pct_registros_completos']:.2f}%)", "Informativo"],
    ["Unicidad", "Duplicados exactos", f"{m['duplicados_exactos']:,} ({m['pct_duplicados_exactos']:.2f}%)", "0 duplicados"],
    ["Unicidad", "Registros con identificador repetido", f"{m['registros_con_id_repetido']:,}", "Identificador unico"],
    ["Validez", "Codigos CIE-10 validos", f"{m['cie_validos']:,} ({m['pct_cie_validos']:.2f}%)", "100% validos"],
    ["Validez", "Edades dentro de rango (0-120)", f"{m['pct_edad_valida']:.2f}%", "100% en rango"],
    ["Validez", "Tiempos de espera atipicos", f"{m['espera_atipica']:,}", "Dentro de rango plausible"],
    ["Consistencia", "Fechas dentro del periodo 2024-2025", f"{m['pct_en_periodo']:.2f}%", "100% en periodo"],
    ["Consistencia", "Anio incoherente con la fecha", f"{m['inconsistente_anio']:,}", "Coherencia total"],
    ["Consistencia", "Mes incoherente con la fecha", f"{m['inconsistente_mes']:,}", "Coherencia total"],
    ["Consistencia", "Categorias de servicio fuera de dominio", f"{m['servicio_inconsistente']:,}", "100% en dominio"],
    ["Integridad", "Registros sin algun campo critico", f"{m['registros_sin_algun_campo_critico']:,}", "Campos criticos completos"],
    ["Temporalidad", "Meses con registros (de 24 esperados)", f"{m['meses_cubiertos']}", "24 meses"],
    ["Temporalidad", "Rango de fechas observado", f"{m['fecha_min']} a {m['fecha_max']}", "2024-01-01 a 2025-12-31"],
    ["Representatividad", "Localidades representadas", f"{len([k for k in rep['localidades'] if isinstance(k, str)])} de 4", "4 localidades"],
    ["Representatividad", "Codigos CIE-10 distintos", f"{rep['n_cie_unicos']}", "Diversidad diagnostica"],
    ["Representatividad", "Profesionales con registros", f"{rep['n_profesionales']}", "Variedad prescriptiva"],
], columns=["dimension", "indicador", "resultado", "criterio_referencia"])
dimensiones.to_csv(TABLAS / "tabla_dimensiones_calidad.csv", index=False, encoding="utf-8-sig")

(R2 / "metricas.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print("EVALUACION DE CALIDAD COMPLETADA")
print(f"Completitud global: {m['pct_celdas_completas']}% | registros completos: {m['pct_registros_completos']}%")
print(f"Duplicados exactos: {m['duplicados_exactos']:,} | IDs repetidos: {m['registros_con_id_repetido']:,}")
print(f"CIE validos: {m['pct_cie_validos']}% (invalidos: {m['cie_invalidos']:,}, nulos: {m['cie_nulos']:,})")
print(f"Edad: validos {m['pct_edad_valida']}% (invalidos: {m['edad_invalida']}, nulos: {m['edad_nula']})")
print(f"Fechas en periodo: {m['pct_en_periodo']}% | inconsistencias anio/mes: {m['inconsistente_anio']}/{m['inconsistente_mes']}")
print(f"Servicio inconsistente: {m['servicio_inconsistente']:,} | profesional nulo: {m['profesional_nulo']}")
print(f"Meses cubiertos: {m['meses_cubiertos']} | prof.: {rep['n_profesionales']} | usuarios: {rep['n_usuarios']}")
print("Archivos: tabla_completitud_variables.csv, tabla_dimensiones_calidad.csv, fig_completitud.png, fig_registros_mes_brutos.png")
