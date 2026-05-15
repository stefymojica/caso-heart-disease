"""
Problema 4 - Repeticion del Problema 2 usando imputacion EM
=============================================================

Se repite el flujo de problema2.py, cambiando solo el paso de imputacion:
- Antes: mediana por variable
- Ahora: algoritmo EM (aproximado con IterativeImputer)

Nota: IterativeImputer implementa una imputacion multivariada iterativa,
comunmente usada como aproximacion practica al enfoque EM para datos faltantes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Este script requiere 'statsmodels'. Instala con: pip install statsmodels"
    ) from exc

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, accuracy_score


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def _map_to_nearest_allowed(values: pd.Series, allowed: list[int]) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    allowed_arr = np.asarray(allowed, dtype=float)
    idx = np.abs(arr[:, None] - allowed_arr[None, :]).argmin(axis=1)
    mapped = allowed_arr[idx]
    return pd.Series(mapped.astype(int), index=values.index)


def main() -> None:
    print("PASO 1 (EM aprox): Carga e imputacion de datos")

    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    print(f"Dimensiones originales: {df.shape}")
    print(f"\nValores faltantes por variable:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Imputacion EM (aprox) con modelo iterativo multivariado.
    # Para evitar fuga de informacion, se imputa solo sobre predictores
    # y se conserva 'num' (variable de desenlace original) sin imputar.
    if df["num"].isna().any():
        raise ValueError("'num' contiene faltantes. Revisar datos antes del modelado.")

    predictors = [c for c in COLUMNS if c != "num"]
    imputer = IterativeImputer(
        random_state=123,
        max_iter=100,
        tol=1e-3,
        sample_posterior=False,
        initial_strategy="median",
    )
    imputed = imputer.fit_transform(df[predictors])
    df_em = pd.DataFrame(imputed, columns=predictors, index=df.index)
    df_em["num"] = df["num"].to_numpy()

    # Recuperar naturaleza categorica en variables categoricas
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope"]:
        df_em[col] = np.rint(df_em[col]).astype(int)

    df_em["ca"] = np.clip(np.rint(df_em["ca"]).astype(int), 0, 3)
    df_em["thal"] = _map_to_nearest_allowed(df_em["thal"], [3, 6, 7])

    # num es discreta (0,1,2,3,4) y proviene del dataset original
    df_em["num"] = np.clip(np.rint(df_em["num"]).astype(int), 0, 4)

    df_em["heart_disease"] = (df_em["num"] > 0).astype(int)

    print("\nImputacion EM aproximada completada (IterativeImputer).")
    print("Verificacion de faltantes tras imputacion:")
    print(df_em.isna().sum()[df_em.isna().sum() > 0])
    print("\nDistribucion de la variable respuesta:")
    print(df_em["heart_disease"].value_counts().rename({0: "Sin enfermedad", 1: "Con enfermedad"}))

    # ----------------------------------------------------------------
    # PASO 2: DISTRIBUCIONES BIVARIADAS
    # ----------------------------------------------------------------
    print("\nPASO 2: Distribuciones bivariadas (categoricas vs respuesta)")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, var in enumerate(CATEGORICAL):
        ct = pd.crosstab(df_em[var], df_em["heart_disease"], normalize="index") * 100
        ct.columns = ["Sin enfermedad (%)", "Con enfermedad (%)"]
        ct["Con enfermedad (%)"].plot(
            kind="bar", ax=axes[i], color="steelblue", edgecolor="black"
        )
        axes[i].set_title(f"{var} vs Enfermedad cardiaca")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("% Con enfermedad")
        axes[i].tick_params(axis="x", rotation=0)
        axes[i].set_ylim(0, 100)

        if ct["Con enfermedad (%)"].isin([0, 100]).any():
            print(f"  '{var}': categoria con 0% o 100% -> posible separacion perfecta")

    plt.suptitle("Proporcion de enfermedad cardiaca por covariable categorica (EM)", fontsize=13)
    plt.tight_layout()
    plt.savefig("p4_bivariadas_categoricas_em.png", dpi=150)
    print("Grafico guardado: p4_bivariadas_categoricas_em.png")

    # ----------------------------------------------------------------
    # PASO 3: MODELO BIVARIADO CON fbs (manual + GLM)
    # ----------------------------------------------------------------
    print("\nPASO 3: Modelo bivariado con 'fbs' - estimacion manual y GLM")

    ct_fbs = pd.crosstab(df_em["fbs"], df_em["heart_disease"])
    ct_fbs.columns = ["No enfermedad (Y=0)", "Enfermedad (Y=1)"]
    ct_fbs.index.name = "fbs"
    print("\nTabla de contingencia fbs vs heart_disease:")
    print(ct_fbs)

    n00 = ct_fbs.loc[0, "No enfermedad (Y=0)"]
    n01 = ct_fbs.loc[0, "Enfermedad (Y=1)"]
    n10 = ct_fbs.loc[1, "No enfermedad (Y=0)"]
    n11 = ct_fbs.loc[1, "Enfermedad (Y=1)"]

    pi_0 = n01 / (n00 + n01)
    pi_1 = n11 / (n10 + n11)
    odds_0 = pi_0 / (1 - pi_0)
    odds_1 = pi_1 / (1 - pi_1)

    beta0_manual = np.log(odds_0)
    beta1_manual = np.log(odds_1 / odds_0)

    print("\nEstimacion MANUAL:")
    print(f"  pi(fbs=0) = {pi_0:.4f}  ->  odds = {odds_0:.4f}")
    print(f"  pi(fbs=1) = {pi_1:.4f}  ->  odds = {odds_1:.4f}")
    print(f"  beta0 = log(odds_0) = {beta0_manual:.4f}")
    print(f"  beta1 = log(OR)     = {beta1_manual:.4f}  (OR = {np.exp(beta1_manual):.4f})")

    X_fbs = sm.add_constant(df_em["fbs"])
    glm_fbs = sm.GLM(df_em["heart_disease"], X_fbs, family=sm.families.Binomial()).fit()

    print("\nVerificacion con GLM (statsmodels):")
    print(f"  beta0 = {glm_fbs.params['const']:.4f}")
    print(f"  beta1 = {glm_fbs.params['fbs']:.4f}")
    coinciden = np.allclose(
        [beta0_manual, beta1_manual],
        [glm_fbs.params["const"], glm_fbs.params["fbs"]],
        atol=1e-3,
    )
    print(f"  Coinciden manual vs GLM: {coinciden}")

    # ----------------------------------------------------------------
    # PASO 4: MODELO MULTIVARIADO + WALD
    # ----------------------------------------------------------------
    print("\nPASO 4: Modelo multivariado - todas las variables")

    features = [c for c in COLUMNS if c != "num"]
    X = sm.add_constant(df_em[features])
    y = df_em["heart_disease"]
    glm_full = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    print(glm_full.summary())

    summary_df = pd.DataFrame(
        {
            "Coeficiente": glm_full.params,
            "Error Estandar": glm_full.bse,
            "z (Wald)": glm_full.tvalues,
            "p-valor": glm_full.pvalues,
            "OR": np.exp(glm_full.params),
        }
    ).round(4)

    print("\nResultados del modelo multivariado (test de Wald):\n")
    print(summary_df.to_string())

    alpha = 0.05
    sig = summary_df[summary_df["p-valor"] < alpha].index.tolist()
    no_sig = summary_df[summary_df["p-valor"] >= alpha].index.tolist()

    print(f"\nVariables SIGNIFICATIVAS (p < {alpha}):")
    print("  " + ", ".join(sig))
    print(f"\nVariables NO significativas (p >= {alpha}):")
    print("  " + ", ".join(no_sig))

    # ----------------------------------------------------------------
    # PASO 5: PROBABILIDADES PREDICHAS
    # ----------------------------------------------------------------
    print("\nPASO 5: Visualizacion de probabilidades predichas")

    df_em["prob_predicha"] = glm_full.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    orden = df_em["prob_predicha"].argsort().values
    axes[0].scatter(
        range(len(df_em)),
        df_em["prob_predicha"].iloc[orden],
        c=df_em["heart_disease"].iloc[orden],
        cmap="RdBu_r",
        alpha=0.6,
        s=20,
    )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Umbral 0.5")
    axes[0].set_xlabel("Pacientes (ordenados por probabilidad)")
    axes[0].set_ylabel("Probabilidad predicha")
    axes[0].set_title("Probabilidades predichas vs variable respuesta")
    axes[0].legend(["Umbral 0.5", "Sin enfermedad (0)", "Con enfermedad (1)"])

    df_em.boxplot(
        column="prob_predicha",
        by="heart_disease",
        ax=axes[1],
        boxprops=dict(color="steelblue"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_title("Distribucion de probabilidades predichas")
    axes[1].set_xlabel("Enfermedad cardiaca real (0 = No, 1 = Si)")
    axes[1].set_ylabel("Probabilidad predicha")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig("p4_probabilidades_predichas_em.png", dpi=150)
    print("Grafico guardado: p4_probabilidades_predichas_em.png")

    auc = roc_auc_score(y, df_em["prob_predicha"])
    y_pred = (df_em["prob_predicha"] >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    print(f"\nAUC-ROC  = {auc:.4f}")
    print(f"Accuracy = {acc:.4f}  (umbral 0.5)")

    print("\n--- Conclusion final ---")
    nivel = "bueno" if auc >= 0.75 else "moderado" if auc >= 0.65 else "limitado"
    print(f"Con imputacion EM aproximada (IterativeImputer), el modelo alcanza")
    print(f"AUC = {auc:.4f} y accuracy = {acc:.4f}, lo que indica un poder discriminativo {nivel}.")
    sig_vars = [v for v in sig if v != "const"]
    if sig_vars:
        print(f"Variables significativas al 5%: {', '.join(sig_vars)}.")
    else:
        print("Ninguna variable individual resulta significativa al 5% tras el ajuste multivariado.")
    print("\nNota metodologica: se utilizo IterativeImputer como aproximacion practica")
    print("al enfoque EM para imputacion multivariada de datos faltantes.")


if __name__ == "__main__":
    main()
