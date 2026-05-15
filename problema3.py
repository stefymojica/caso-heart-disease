"""
Problema 3 - Comparacion de poder predictivo entre dos modelos
================================================================

Dataset: docs/AAD-taller03.xlsx
Variables esperadas:
  - Incumplimiento (0/1 observado)
  - ScoreLogisticoA (score/probabilidad modelo A)
  - ScoreLogisticoB (score/probabilidad modelo B)

Estrategia estadistica:
  1) Medir discriminacion con AUC-ROC para cada modelo.
  2) Ajustar orientacion del score si viene en sentido inverso.
  3) Estimar IC95% del AUC por bootstrap.
  4) Comparar modelos con bootstrap pareado sobre Delta AUC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


FILE_PATH = Path(__file__).resolve().parent / "docs" / "AAD-taller03.xlsx"
TARGET = "Incumplimiento"
MODEL_A = "ScoreLogisticoA"
MODEL_B = "ScoreLogisticoB"

N_BOOT = 2000
SEED = 123


def _validate_columns(df: pd.DataFrame) -> None:
    required = {TARGET, MODEL_A, MODEL_B}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")


def _orient_score(y: np.ndarray, score: np.ndarray, name: str) -> tuple[np.ndarray, float, str]:
    """
    Asegura que score alto implique mayor riesgo (AUC >= 0.5)
    comparando score vs -score.
    """
    auc_direct = roc_auc_score(y, score)
    auc_neg = roc_auc_score(y, -score)

    if auc_direct >= auc_neg:
        return score, auc_direct, f"{name}: orientacion original"
    return -score, auc_neg, f"{name}: orientacion invertida (se uso -score)"


def _bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    vals: list[float] = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        sb = score[idx]

        # Necesitamos ambas clases para calcular AUC
        if np.unique(yb).size < 2:
            continue
        vals.append(roc_auc_score(yb, sb))

    if not vals:
        return (np.nan, np.nan)

    low, high = np.percentile(vals, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_delta_auc(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float], float]:
    """
    Bootstrap pareado de Delta AUC = AUC_A - AUC_B.
    Retorna:
      - delta observado
      - IC95% bootstrap percentil
      - p-valor bilateral aproximado para H0: delta = 0
    """
    auc_a = roc_auc_score(y, score_a)
    auc_b = roc_auc_score(y, score_b)
    delta_obs = auc_a - auc_b

    deltas: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        ab = score_a[idx]
        bb = score_b[idx]
        if np.unique(yb).size < 2:
            continue
        deltas.append(roc_auc_score(yb, ab) - roc_auc_score(yb, bb))

    if not deltas:
        return float(delta_obs), (np.nan, np.nan), np.nan

    deltas_arr = np.asarray(deltas)
    ci_low, ci_high = np.percentile(deltas_arr, [2.5, 97.5])

    # p bilateral aproximado con correccion de continuidad
    # para evitar reportar p exactamente 0 por remuestreo finito.
    b = deltas_arr.size
    count_left = np.sum(deltas_arr <= 0)
    count_right = np.sum(deltas_arr >= 0)
    p_left = (count_left + 1) / (b + 1)
    p_right = (count_right + 1) / (b + 1)
    p_two = min(1.0, float(2 * min(p_left, p_right)))

    return float(delta_obs), (float(ci_low), float(ci_high)), p_two


def main() -> None:
    print("PROBLEMA 3: Comparacion de dos modelos de score")
    print("=" * 62)

    df = pd.read_excel(FILE_PATH)
    _validate_columns(df)

    if df[[TARGET, MODEL_A, MODEL_B]].isna().any().any():
        raise ValueError("Hay valores faltantes en columnas clave; revisar el archivo de entrada.")

    y = df[TARGET].to_numpy(dtype=int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("La variable de respuesta debe ser binaria (0/1).")

    score_a_raw = df[MODEL_A].to_numpy(dtype=float)
    score_b_raw = df[MODEL_B].to_numpy(dtype=float)

    # Transparencia: reportar AUC directo antes de cualquier correccion
    auc_a_direct = roc_auc_score(y, score_a_raw)
    auc_b_direct = roc_auc_score(y, score_b_raw)
    print("\n--- Orientacion directa (sin correccion) ---")
    print(f"Modelo A: AUC = {auc_a_direct:.4f}")
    print(f"Modelo B: AUC = {auc_b_direct:.4f}")
    if auc_b_direct < 0.5:
        print("Nota: ScoreLogisticoB orientado inversamente; se corrige con -score para comparar.")

    score_a, auc_a, note_a = _orient_score(y, score_a_raw, MODEL_A)
    score_b, auc_b, note_b = _orient_score(y, score_b_raw, MODEL_B)

    print(f"\nObservaciones: {len(df)}")
    print(f"Prevalencia incumplimiento: {y.mean():.4f}")
    print(note_a)
    print(note_b)

    rng = np.random.default_rng(SEED)
    ci_a = _bootstrap_auc_ci(y, score_a, N_BOOT, rng)
    ci_b = _bootstrap_auc_ci(y, score_b, N_BOOT, rng)
    delta, ci_delta, p_val = _bootstrap_delta_auc(y, score_a, score_b, N_BOOT, rng)

    print("\n--- Resultados (AUC) ---")
    print(f"Modelo A: AUC = {auc_a:.4f}  IC95% [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"Modelo B: AUC = {auc_b:.4f}  IC95% [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print(f"Delta AUC (A-B) = {delta:.4f}  IC95% [{ci_delta[0]:.4f}, {ci_delta[1]:.4f}]")
    if p_val < 0.001:
        print("p-valor bootstrap bilateral (H0: Delta=0): < 0.001")
    else:
        print(f"p-valor bootstrap bilateral (H0: Delta=0): {p_val:.6f}")

    best = "A" if auc_a > auc_b else "B"
    print("\n--- Conclusion ---")
    print(f"Modelo A (corregido): AUC = {auc_a:.4f}  IC95% [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"Modelo B (corregido): AUC = {auc_b:.4f}  IC95% [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    p_text = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
    print(f"Delta AUC (A-B) = {delta:.4f}  IC95% [{ci_delta[0]:.4f}, {ci_delta[1]:.4f}]  p {p_text}")
    if p_val < 0.05:
        print(
            f"\nEl modelo {best} presenta mayor poder predictivo "
            f"(diferencia estadisticamente significativa, p < 0.05)."
        )
        if best == "B":
            print("Esto se debe a que el score B venia en orientacion inversa; al corregirlo supera al A.")
    else:
        print(
            f"\nEl modelo {best} tiene AUC mayor, pero la diferencia no es "
            f"estadisticamente significativa al 5%."
        )

    # Curva ROC comparativa
    fpr_a, tpr_a, _ = roc_curve(y, score_a)
    fpr_b, tpr_b, _ = roc_curve(y, score_b)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, label=f"Modelo A (AUC={auc_a:.3f})", linewidth=2)
    plt.plot(fpr_b, tpr_b, label=f"Modelo B (AUC={auc_b:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curvas ROC - Modelos de Incumplimiento")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("problema3_roc_comparacion.png", dpi=150)
    print("\nGrafico guardado: problema3_roc_comparacion.png")


if __name__ == "__main__":
    main()
