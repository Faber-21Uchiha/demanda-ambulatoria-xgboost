# -*- coding: utf-8 -*-
"""
03_limpiar_datos.py
Aplicacion de los criterios de inclusion y exclusion definidos en el OE1 sobre
el conjunto de datos bruto (sintetico) y generacion del conjunto de datos
depurado que alimentara el OE2 (caracterizacion prescriptiva).

Criterios (aplicados en orden secuencial y trazable):
  C1  Fecha de atencion dentro del periodo 01/2024 - 12/2025
  C2  Sin duplicados exactos (se conserva la primera aparicion)
  C3  Edad presente y dentro del rango 0-120 anos
  C4  Codigo CIE-10 presente y perteneciente al catalogo de referencia
  C5  Campos criticos presentes: servicio, localidad, EPS, tipo de orden,
      profesional, sexo
  Tratamientos (no excluyen, corrigen):
  T1  Homologacion de categorias de servicio al dominio canonico
  T2  Anulacion de tiempos de espera atipicos (0 o > 240 min)
  T3  Recalculo de campos derivados para restaurar la consistencia interna
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
R2 = BASE / "03_resultados"
TABLAS = R2 / "tablas"
TABLAS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "01_datos" / "demanda_ambulatoria_sintetica_2024_2025_100k.csv",
                 encoding="utf-8-sig")
df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"], errors="coerce")

catalogo = pd.read_excel(BASE / "04_documentacion" / "diccionario_y_reglas_OE1.xlsx",
                         sheet_name="Catalogo_CIE10")
codigos_validos = set(catalogo["codigo_cie10"])
mapa_diag = dict(zip(catalogo["codigo_cie10"], catalogo["diagnostico_categoria"]))

n0 = len(df)
trazabilidad = []
vivos = df.copy()

def aplicar_criterio(codigo, descripcion, mask, estado_actual):
    """Aplica un criterio de exclusion de forma trazable sobre el estado vigente."""
    excluidos = estado_actual[mask]
    resultado = estado_actual[~mask]
    trazabilidad.append({
        "paso": codigo,
        "criterio": descripcion,
        "registros_excluidos": int(len(excluidos)),
        "registros_restantes": int(len(resultado)),
        "pct_excluidos": round(100 * len(excluidos) / n0, 2),
    })
    return resultado

# ---------------------------------------------------------------------------
# C1: fecha valida dentro del periodo de estudio
# ---------------------------------------------------------------------------
mask_c1 = (~vivos["fecha_atencion"].between("2024-01-01", "2025-12-31"))
vivos = aplicar_criterio("C1", "Fecha fuera del periodo enero 2024 - diciembre 2025 (o fecha invalida)",
                         mask_c1, vivos)

# ---------------------------------------------------------------------------
# C2: eliminacion de duplicados exactos
# ---------------------------------------------------------------------------
mask_c2 = vivos.duplicated(keep="first")
vivos = aplicar_criterio("C2", "Registro duplicado exacto de otro registro (se conserva la primera aparicion)",
                         mask_c2, vivos)

# ---------------------------------------------------------------------------
# C3: edad presente y dentro de rango 0-120
# ---------------------------------------------------------------------------
mask_c3 = vivos["edad"].isna() | (vivos["edad"] < 0) | (vivos["edad"] > 120)
vivos = aplicar_criterio("C3", "Edad ausente o fuera del rango 0-120 anos",
                        mask_c3, vivos)

# ---------------------------------------------------------------------------
# C4: CIE-10 presente y valido
# ---------------------------------------------------------------------------
mask_c4 = vivos["codigo_cie10"].isna() | (~vivos["codigo_cie10"].isin(codigos_validos))
vivos = aplicar_criterio("C4", "Diagnostico CIE-10 ausente o no perteneciente al catalogo de referencia",
                        mask_c4, vivos)

# ---------------------------------------------------------------------------
# C5: campos criticos presentes
# ---------------------------------------------------------------------------
criticos = ["servicio", "localidad_residencia", "eps", "tipo_orden", "id_profesional", "sexo"]
mask_c5 = vivos[criticos].isna().any(axis=1)
vivos = aplicar_criterio("C5", "Ausencia de al menos un campo critico (servicio, localidad, EPS, tipo de orden, profesional o sexo)",
                        mask_c5, vivos)

# ---------------------------------------------------------------------------
# T1: homologacion de categorias de servicio al dominio canonico
# ---------------------------------------------------------------------------
def homologar_servicio(s):
    if pd.isna(s):
        return s
    t = str(s).strip().lower().replace("-", " ").replace(".", " ")
    t = " ".join(t.split())
    aliases = {
        "med general": "Medicina general",
        "medicina general": "Medicina general",
        "enfermeria": "Enfermería",
        "odontologia": "Odontología",
        "pediatria": "Pediatría",
        "medicina interna": "Medicina interna",
        "medicina familiar": "Medicina familiar",
        "ginecologia": "Ginecología",
        "psicologia": "Psicología",
        "nutricion": "Nutrición",
        "laboratorio": "Laboratorio",
        "imagenes diagnosticas": "Imágenes diagnósticas",
    }
    return aliases.get(t, s)

n_homologados = int((~vivos["servicio"].isin(
    ["Medicina general", "Enfermería", "Odontología", "Pediatría", "Medicina interna",
     "Medicina familiar", "Ginecología", "Psicología", "Nutrición", "Laboratorio",
     "Imágenes diagnósticas"])).sum())
vivos["servicio"] = vivos["servicio"].map(homologar_servicio)

# ---------------------------------------------------------------------------
# T2: anulacion de tiempos de espera atipicos (tratamiento, no exclusion)
# ---------------------------------------------------------------------------
mask_t2 = (vivos["tiempo_espera_min"] == 0) | (vivos["tiempo_espera_min"] > 240)
n_espera_anulados = int(mask_t2.sum())
vivos.loc[mask_t2, "tiempo_espera_min"] = np.nan

# ---------------------------------------------------------------------------
# T3: recalculo de campos derivados (restaura consistencia interna)
# ---------------------------------------------------------------------------
f = vivos["fecha_atencion"]
vivos["anio"] = f.dt.year.astype(int)
vivos["mes"] = f.dt.month.astype(int)
vivos["semana_anio"] = f.dt.isocalendar().week.astype(int).to_numpy()
DIAS_ES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
vivos["dia_semana"] = f.dt.weekday.map(DIAS_ES)
vivos["trimestre"] = "T" + ((vivos["mes"] - 1) // 3 + 1).astype(str)
vivos["grupo_edad"] = pd.cut(vivos["edad"], bins=[-1, 5, 11, 17, 28, 44, 59, 74, 200],
                             labels=["0-5", "6-11", "12-17", "18-28", "29-44",
                                     "45-59", "60-74", "75+"]).astype(str)
vivos["diagnostico_categoria"] = vivos["codigo_cie10"].map(mapa_diag)
vivos["categoria_orden"] = vivos["tipo_orden"]

# ---------------------------------------------------------------------------
# C6: verificacion final de unicidad tras los tratamientos T1-T3
# (al homologar categorias y anular valores atipicos, registros que solo
#  diferaban en el campo corrupto pueden volverse identicos)
# ---------------------------------------------------------------------------
mask_c6 = vivos.duplicated(keep="first")
vivos = aplicar_criterio("C6", "Duplicados residuales identificados tras los tratamientos T1-T3",
                         mask_c6, vivos)

# ---------------------------------------------------------------------------
# Exportaciones
# ---------------------------------------------------------------------------
vivos = vivos.sort_values(["fecha_atencion", "id_registro"]).reset_index(drop=True)
vivos["fecha_atencion"] = vivos["fecha_atencion"].dt.strftime("%Y-%m-%d")
ruta_limpio = BASE / "01_datos" / "dataset_depurado_OE1.csv"
vivos.to_csv(ruta_limpio, index=False, encoding="utf-8-sig")

traz = pd.DataFrame(trazabilidad)
traz.to_csv(TABLAS / "tabla_trazabilidad_exclusiones.csv", index=False, encoding="utf-8-sig")

m = {}
m["n_inicial"] = n0
m["n_final"] = int(len(vivos))
m["n_excluidos_total"] = int(n0 - len(vivos))
m["pct_excluidos"] = round(100 * (n0 - len(vivos)) / n0, 2)
m["pct_conservados"] = round(100 * len(vivos) / n0, 2)
m["trazabilidad"] = trazabilidad
m["n_homologados_servicio"] = n_homologados
m["n_espera_anulados"] = n_espera_anulados
m["celdas_nulas_final"] = int(vivos.isna().sum().sum())
m["duplicados_final"] = int(vivos.duplicated().sum())

# Actualizar metricas.json con la capa de limpieza
ruta_metricas = R2 / "metricas.json"
metricas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
metricas["limpieza"] = m
ruta_metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print("LIMPIEZA COMPLETADA")
print(traz.to_string(index=False))
print(f"\nRegistros: {n0:,} -> {len(vivos):,}  (excluidos {n0 - len(vivos):,} = {m['pct_excluidos']}%)")
print(f"Servicios homologados (T1): {n_homologados:,} | tiempos de espera anulados (T2): {n_espera_anulados:,}")
print(f"Dataset final: {ruta_limpio}")
