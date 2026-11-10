# -*- coding: utf-8 -*-
"""
09_panel_zonas_capacidad_pqrs.py

Construye la capa de datos que faltaba para los objetivos especificos 4 y 5:

  1. Panel zona-semana (104/105 semanas x 4 zonas) con la demanda observada en
     el conjunto depurado del OE1.
  2. Serie sintetica de capacidad instalada (cupos ofertados) por zona y semana,
     construida con una regla de asignacion presupuestal "pegajosa": la
     capacidad de cada zona se fija con el nivel del ano anterior y solo se
     ajusta por estacionalidad mensual y ruido operativo.
  3. Registros sinteticos de PQRS asociados a la demanda no atendida (falta de
     oportunidad), con motivo, servicio, dias de oportunidad y estado.

DECLARACION ACADEMICA
---------------------
La capacidad y las PQRS son SINTETICAS. No provienen de la institucion. Se
generan de forma determinista a partir de la demanda observada en la capa
sintetica del OE1 y de parametros declarados en el archivo de supuestos. Su
unico proposito es permitir que el OE5 se evalue con cifras reproducibles y no
con supuestos narrativos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261109
BASE = Path(__file__).resolve().parents[1]
DATOS = BASE / "01_datos"
RES = BASE / "03_resultados"

MAPA_ZONA = {
    "Kennedy": "Norte",
    "Bosa": "Sur",
    "Fontibón": "Centro",
    "Puente Aranda": "Occidente",
}

# Factores de oferta por zona: capacidad nominal respecto de la demanda media
# de 2024. Menores a 1 implican deficit estructural de cupos.
FACTOR_CAPACIDAD = {"Norte": 0.985, "Sur": 0.972, "Centro": 0.958, "Occidente": 0.945}

# Probabilidad de que una cita no atendida por falta de cupo derive en PQRS.
P_PQRS_OPORTUNIDAD = 0.28
# PQRS administrativas de base por cada 1.000 atenciones.
TASA_PQRS_ADMIN = 2.1

MOTIVOS = ["Falta de oportunidad en la asignación", "Demora en la asignación de cita",
           "Cancelación de cita por el prestador", "Información insuficiente sobre el trámite"]
P_MOTIVO = [0.62, 0.21, 0.09, 0.08]
CANALES = ["Presencial", "Telefónico", "Virtual"]
P_CANAL = [0.31, 0.46, 0.23]
ESTADOS = ["Resuelta", "En trámite", "Cerrada sin respuesta"]
P_ESTADO = [0.74, 0.19, 0.07]

PESO_MES = np.array([0.90, 0.92, 1.00, 1.02, 1.05, 1.00,
                     0.98, 1.02, 1.05, 1.08, 1.03, 0.92])


def cargar_demanda():
    df = pd.read_csv(DATOS / "dataset_depurado_OE1.csv", encoding="utf-8-sig")
    df["fecha_atencion"] = pd.to_datetime(df["fecha_atencion"])
    df["zona"] = df["localidad_residencia"].map(MAPA_ZONA)
    df["semana"] = df["fecha_atencion"].dt.to_period("W-SUN").dt.start_time
    return df


def construir_panel(df):
    g = df.groupby(["zona", "semana"])
    panel = pd.DataFrame({
        "demanda": g.size(),
        "atendidas": g["atencion_realizada"].apply(lambda s: (s == "Sí").sum()),
        "canceladas": g["cancelada"].apply(lambda s: (s == "Sí").sum()),
        "inasistencias": g["inasistencia"].apply(lambda s: (s == "Sí").sum()),
        "medicina_general": g["servicio"].apply(lambda s: (s == "Medicina general").mean()),
        "enfermeria": g["servicio"].apply(lambda s: (s == "Enfermería").mean()),
        "odontologia": g["servicio"].apply(lambda s: (s == "Odontología").mean()),
        "pediatria": g["servicio"].apply(lambda s: (s == "Pediatría").mean()),
        "apoyo_diagnostico": g["servicio"].apply(
            lambda s: s.isin(["Laboratorio", "Imágenes diagnósticas"]).mean()),
        "edad_media": g["edad"].mean(),
        "prop_femenino": g["sexo"].apply(lambda s: (s == "F").mean()),
        "prop_mayor_60": g["grupo_edad"].apply(lambda s: s.isin(["60-74", "75+"]).mean()),
        "prop_contributivo": g["regimen"].apply(lambda s: (s == "Contributivo").mean()),
        "prop_prioritaria": g["prioridad"].apply(lambda s: (s == "Preferente").mean()),
        "prop_remision": g["remision"].apply(lambda s: (s == "Sí").mean()),
        "espera_media": g["tiempo_espera_min"].mean(),
    }).reset_index()

    panel["anio"] = panel["semana"].dt.year
    panel["mes"] = panel["semana"].dt.month
    panel["semana_anio"] = panel["semana"].dt.isocalendar().week.astype(int)
    # se excluye la semana incompleta del cierre (29-dic-2025 a 31-dic-2025)
    panel = panel[panel["semana"] + pd.Timedelta(days=6) <= pd.Timestamp("2025-12-31")]
    panel = panel.sort_values(["zona", "semana"]).reset_index(drop=True)
    return panel


def construir_capacidad(panel, rng):
    """Capacidad pegajosa: base del ano anterior + estacionalidad + ruido AR(1)."""
    base_2024 = panel[panel["anio"] == 2024].groupby("zona")["demanda"].mean()
    filas = []
    ruido_prev = {z: 0.0 for z in base_2024.index}
    for _, r in panel.iterrows():
        z = r["zona"]
        est = 1 + 0.6 * (PESO_MES[r["mes"] - 1] - 1)
        e = rng.normal(0, 0.012)
        ruido = 0.55 * ruido_prev[z] + e
        ruido_prev[z] = ruido
        cap = base_2024[z] * FACTOR_CAPACIDAD[z] * est * (1 + ruido)
        filas.append(max(1, int(round(cap))))
    panel["capacidad"] = filas
    panel["utilizacion"] = panel["demanda"] / panel["capacidad"]
    panel["no_atendidas"] = (panel["demanda"] - panel["capacidad"]).clip(lower=0)
    panel["tasa_no_atencion"] = panel["no_atendidas"] / panel["demanda"]
    return panel


def generar_pqrs(panel, df, rng):
    """PQRS ligadas a la demanda no atendida, mas un fondo administrativo."""
    filas = []
    servicios_por_zona = df.groupby("zona")["servicio"].value_counts(normalize=True)
    fondo = df.groupby("zona")["servicio"].value_counts()
    for _, r in panel.iterrows():
        z = r["zona"]
        lam_op = P_PQRS_OPORTUNIDAD * r["no_atendidas"]
        lam_admin = TASA_PQRS_ADMIN / 1000 * r["demanda"]
        n_op = int(rng.poisson(lam_op)) if lam_op > 0 else 0
        n_ad = int(rng.poisson(lam_admin)) if lam_admin > 0 else 0
        for _ in range(n_op):
            # deficit relativo del ano: a mayor deficit, mas dias de espera
            ratio = float(r["tasa_no_atencion"])
            dias = int(np.clip(rng.gamma(shape=2.4, scale=6 + 40 * ratio), 1, 120))
            filas.append((r["semana"], z, "Oportunidad", dias, True))
        for _ in range(n_ad):
            dias = int(np.clip(rng.gamma(shape=1.6, scale=4), 1, 60))
            filas.append((r["semana"], z, "Administrativa", dias, False))

    if not filas:
        return pd.DataFrame(columns=["id_pqrs", "semana_hecho", "fecha_radicacion", "zona",
                                     "servicio", "motivo", "dias_oportunidad", "canal",
                                     "estado", "vinculada_falta_oportunidad"])

    p = pd.DataFrame(filas, columns=["semana", "zona", "tipo", "dias_oportunidad",
                                     "vinculada_falta_oportunidad"])
    servicios = []
    motivos = []
    for _, r in p.iterrows():
        dist = servicios_por_zona.loc[r["zona"]]
        servicios.append(rng.choice(dist.index.to_numpy(), p=dist.to_numpy()))
        if r["tipo"] == "Oportunidad":
            motivos.append(rng.choice(MOTIVOS, p=P_MOTIVO))
        else:
            motivos.append(rng.choice(MOTIVOS[1:], p=np.array(P_MOTIVO[1:]) / sum(P_MOTIVO[1:])))
    p["servicio"] = servicios
    p["motivo"] = motivos
    p["semana_hecho"] = p["semana"]
    # radicacion entre 1 y 15 dias despues del cierre de la semana
    p["fecha_radicacion"] = p["semana"] + pd.to_timedelta(rng.integers(1, 16, len(p)), unit="D")
    p["canal"] = rng.choice(CANALES, len(p), p=P_CANAL)
    p["estado"] = rng.choice(ESTADOS, len(p), p=P_ESTADO)
    p["id_pqrs"] = [f"PQRS{i:06d}" for i in range(1, len(p) + 1)]
    p = p[["id_pqrs", "semana_hecho", "fecha_radicacion", "zona", "servicio", "motivo",
           "dias_oportunidad", "canal", "estado", "vinculada_falta_oportunidad"]]
    p = p.sort_values(["fecha_radicacion", "id_pqrs"]).reset_index(drop=True)
    return p


def main():
    rng = np.random.default_rng(SEED)
    df = cargar_demanda()
    panel = construir_panel(df)
    panel = construir_capacidad(panel, rng)
    pqrs = generar_pqrs(panel, df, rng)

    panel_sal = panel.copy()
    panel_sal["semana"] = panel_sal["semana"].dt.strftime("%Y-%m-%d")
    panel_sal.to_csv(DATOS / "panel_zona_semana_capacidad.csv", index=False, encoding="utf-8-sig")

    pqrs_sal = pqrs.copy()
    pqrs_sal["semana_hecho"] = pd.to_datetime(pqrs_sal["semana_hecho"]).dt.strftime("%Y-%m-%d")
    pqrs_sal["fecha_radicacion"] = pd.to_datetime(pqrs_sal["fecha_radicacion"]).dt.strftime("%Y-%m-%d")
    pqrs_sal.to_csv(DATOS / "pqrs_sinteticas_2024_2025.csv", index=False, encoding="utf-8-sig")

    agenda = panel[["zona", "semana", "capacidad", "demanda"]].copy()
    agenda["semana"] = agenda["semana"].dt.strftime("%Y-%m-%d")
    agenda.to_csv(DATOS / "capacidad_agenda_sintetica.csv", index=False, encoding="utf-8-sig")

    supuestos = {
        "semilla": SEED,
        "mapa_localidad_zona": MAPA_ZONA,
        "factor_capacidad_por_zona": FACTOR_CAPACIDAD,
        "regla_capacidad": ("capacidad nominal = demanda media de 2024 de la zona x factor "
                            "x estacionalidad mensual atenuada x (1 + ruido AR(1) sd 1,2 %)"),
        "p_pqrs_por_cita_no_atendida": P_PQRS_OPORTUNIDAD,
        "pqrs_administrativas_por_1000_atenciones": TASA_PQRS_ADMIN,
        "n_semanas": int(panel["semana"].nunique()),
        "n_zonas": int(panel["zona"].nunique()),
        "n_observaciones_panel": int(len(panel)),
        "demanda_total": int(panel["demanda"].sum()),
        "capacidad_total": int(panel["capacidad"].sum()),
        "no_atendidas_total": int(panel["no_atendidas"].sum()),
        "tasa_no_atencion_global": round(float(panel["no_atendidas"].sum() / panel["demanda"].sum()) * 100, 2),
        "pqrs_total": int(len(pqrs)),
        "pqrs_oportunidad": int(pqrs["vinculada_falta_oportunidad"].sum()),
        "semanas": [str(panel["semana"].min().date()), str(panel["semana"].max().date())],
    }
    (RES / "supuestos_capa_oe5.json").write_text(
        json.dumps(supuestos, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(supuestos, ensure_ascii=False, indent=1))
    print("\nPanel por zona:")
    print(panel.groupby("zona")[["demanda", "capacidad", "no_atendidas"]].sum().to_string())
    print("\nPQRS por motivo:")
    print(pqrs["motivo"].value_counts().to_string())
    print("\nPQRS por zona:")
    print(pqrs.groupby("zona").size().to_string())


if __name__ == "__main__":
    main()
