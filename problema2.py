"""
Problema 2 - Regresión Logística sobre Heart Disease (UCI)
=============================================================
Pasos:
  1. Carga e imputación de datos faltantes (?) con la mediana
  2. Distribuciones bivariadas (variable respuesta vs covariables categóricas)
  3. Modelo bivariado con `fbs`: estimación manual y verificación con GLM
  4. Modelo multivariado + test de Wald
  5. Visualización de probabilidades predichas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Covariables categóricas del dataset
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# ─────────────────────────────────────────────
# 1. CARGA E IMPUTACIÓN
# ─────────────────────────────────────────────
print("-"*50)
print("PASO 1: Carga e imputación de datos")
print("-"*50)

df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")

print(f"Dimensiones originales: {df.shape}")
print(f"\nValores faltantes por variable:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Imputar con la mediana de cada columna
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  '{col}' imputada con mediana = {median_val}")

df["heart_disease"] = (df["num"] > 0).astype(int)

print(f"Valores faltantes luego de imputar = {df.isnull().sum().sum()}")

print(f"\nDistribución de la variable respuesta:")
print(df["heart_disease"].value_counts().rename({0: "Sin enfermedad", 1: "Con enfermedad"}))

# ─────────────────────────────────────────────
# 2. DISTRIBUCIONES BIVARIADAS
# ─────────────────────────────────────────────
print("\n"+("-"*50))
print("PASO 2: Distribuciones bivariadas (categóricas vs respuesta)")
print("-"*50)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, var in enumerate(CATEGORICAL):
    ct = pd.crosstab(df[var], df["heart_disease"], normalize="index") * 100
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
        print(f" '{var}': categoría con 0% o 100% → posible separación perfecta")

plt.suptitle("Proporción de enfermedad cardiaca por covariable categórica", fontsize=13)
plt.tight_layout()
plt.savefig("bivariadas_categoricas.png", dpi=150)
plt.show()
print("Gráfico guardado: bivariadas_categoricas.png \n")

# ─────────────────────────────────────────────
# 3. MODELO BIVARIADO CON fbs
# ─────────────────────────────────────────────
print("-"*50)
print("PASO 3: Modelo bivariado con 'fbs' — estimación manual y GLM")
print("-"*50)

# --- Tabla de contingencia ---
ct_fbs = pd.crosstab(df["fbs"], df["heart_disease"])
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

print(f"\nEstimación MANUAL:")
print(f"  π(fbs=0) = {pi_0:.4f}  →  odds = {odds_0:.4f}")
print(f"  π(fbs=1) = {pi_1:.4f}  →  odds = {odds_1:.4f}")
print(f"  β0 = log(odds_0) = {beta0_manual:.4f}")
print(f"  β1 = log(OR)     = {beta1_manual:.4f}  (OR = {np.exp(beta1_manual):.4f})")

# --- Verificación con statsmodels GLM ---
X_fbs = sm.add_constant(df["fbs"])
glm_fbs = sm.GLM(
    df["heart_disease"], X_fbs,
    family=sm.families.Binomial()
).fit()

print(f"\nVerificación con GLM (statsmodels):")
print(f"  β0 = {glm_fbs.params['const']:.4f}")
print(f"  β1 = {glm_fbs.params['fbs']:.4f}")
coinciden = np.allclose([beta0_manual, beta1_manual],
                        [glm_fbs.params['const'], glm_fbs.params['fbs']], atol=1e-3)
print(f"\n  Los resultados manuales y GLM coinciden: {coinciden}\n")

# ─────────────────────────────────────────────
# 4. MODELO MULTIVARIADO
# ─────────────────────────────────────────────
print("-"*50)
print("PASO 4: Modelo multivariado — todas las variables")
print("-"*50)

features = [c for c in COLUMNS if c != "num"]
X = sm.add_constant(df[features])
y = df["heart_disease"]

glm_full = sm.GLM(y, X, family=sm.families.Binomial()).fit()

print(glm_full.summary())

summary_df = pd.DataFrame({
    "Coeficiente": glm_full.params,
    "Error Estándar": glm_full.bse,
    "z (Wald)": glm_full.tvalues,
    "p-valor": glm_full.pvalues,
    "OR": np.exp(glm_full.params),
}).round(4)

print("\nResultados del modelo multivariado (test de Wald):\n")
print(summary_df.to_string())

ALPHA = 0.05
sig = summary_df[summary_df["p-valor"] < ALPHA].index.tolist()
no_sig = summary_df[summary_df["p-valor"] >= ALPHA].index.tolist()

print(f"\nVariables SIGNIFICATIVAS (p < {ALPHA}):")
print("  " + ", ".join(sig))
print(f"\nVariables NO significativas (p >= {ALPHA}):")
print("  " + ", ".join(no_sig))

# ─────────────────────────────────────────────
# 5. PROBABILIDADES PREDICHAS
# ─────────────────────────────────────────────
print("\n"+("-"*50))
print("PASO 5: Visualización de probabilidades predichas")
print("-"*50)

df["prob_predicha"] = glm_full.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

orden = df["prob_predicha"].argsort().values
axes[0].scatter(
    range(len(df)), df["prob_predicha"].iloc[orden],
    c=df["heart_disease"].iloc[orden], cmap="RdBu_r",
    alpha=0.6, s=20
)
axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Umbral 0.5")
axes[0].set_xlabel("Pacientes (ordenados por probabilidad)")
axes[0].set_ylabel("Probabilidad predicha")
axes[0].set_title("Probabilidades predichas vs variable respuesta")
axes[0].legend(["Umbral 0.5", "Sin enfermedad (0)", "Con enfermedad (1)"])

df.boxplot(column="prob_predicha", by="heart_disease", ax=axes[1],
           boxprops=dict(color="steelblue"),
           medianprops=dict(color="red", linewidth=2))
axes[1].set_title("Distribución de probabilidades predichas")
axes[1].set_xlabel("Enfermedad cardiaca real (0 = No, 1 = Sí)")
axes[1].set_ylabel("Probabilidad predicha")
plt.suptitle("")

plt.tight_layout()
plt.savefig("probabilidades_predichas.png", dpi=150)
plt.show()
print("Gráfico guardado: probabilidades_predichas.png")

from sklearn.metrics import roc_auc_score, accuracy_score

auc = roc_auc_score(y, df["prob_predicha"])
y_pred = (df["prob_predicha"] >= 0.5).astype(int)
acc = accuracy_score(y, y_pred)

print(f"\nAUC-ROC  = {auc:.4f}")
print(f"Accuracy = {acc:.4f}  (umbral 0.5)")
print("\n¿Describe el modelo la enfermedad cardiaca?")
print(f"  Un AUC de {auc:.2f} indica {'buen' if auc >= 0.75 else 'moderado'} poder discriminativo.")
