# CASO HEART DISEASE

---

## Paso 2: Distribuciones bivariadas — covariables categóricas vs variable respuesta

Se analizó la proporción de enfermedad cardiaca (`heart_disease = 1`) dentro de cada categoría
de las 8 covariables categóricas del dataset.

### Tablas de contingencia (% con enfermedad cardiaca)

| Variable | Categorías | % con enfermedad por categoría |
|---|---|---|
| `sex` | 0 = Mujer, 1 = Hombre | Mujer: 25.8% / Hombre: 55.3% |
| `cp` | 1–4 (tipo de dolor) | 1: 30.4% / 2: 18.0% / 3: 20.9% / **4: 72.9%** |
| `fbs` | 0 = ≤120 mg/dl, 1 = >120 mg/dl | 0: 45.3% / 1: 48.9% |
| `restecg` | 0, 1, 2 | 0: 37.1% / **1: 75.0%** / 2: 54.1% |
| `exang` | 0 = No, 1 = Sí | 0: 30.9% / 1: 76.8% |
| `slope` | 1, 2, 3 | 1: 25.4% / 2: 65.0% / 3: 57.1% |
| `ca` | 0–3 (vasos coloreados) | 0: 26.1% / 1: 67.7% / 2: 81.6% / 3: 85.0% |
| `thal` | 3, 6, 7 | 3: 22.6% / 6: 66.7% / **7: 76.1%** |

---

### ¿Se observa algún inconveniente?

**Sí. Hay dos problemas concretos:**

#### Problema 1: Categoría con muy pocos datos — `restecg = 1`

La categoría `restecg = 1` (hipertrofia ventricular izquierda probable) tiene **solo 4 observaciones**
en todo el dataset (1 sin enfermedad, 3 con enfermedad). Esto genera:

- Estimaciones del coeficiente β inestables con errores estándar muy grandes.
- El 75% reportado no es representativo de la población real; es producto del tamaño muestral mínimo.
- En el modelo multivariado, `restecg` resulta **no significativo** (p = 0.198), en parte por esta razón.

> **Consecuencia:** en un análisis riguroso se debería considerar colapsar `restecg = 0` y `restecg = 1`
> en una sola categoría, o al menos reportar la inestabilidad del estimador.

---

#### Problema 2: `fbs` no discrimina la variable respuesta

La glucemia en ayunas (`fbs`) muestra porcentajes de enfermedad casi idénticos entre sus dos categorías:

- `fbs = 0` (glucemia normal): **45.3%** con enfermedad
- `fbs = 1` (glucemia alta): **48.9%** con enfermedad

La diferencia es de apenas **3.6 puntos porcentuales**, lo que indica que `fbs` prácticamente
no aporta información para distinguir pacientes con y sin enfermedad cardiaca.

Esto se confirma en el modelo bivariado (Paso 3): OR = 1.15 (casi 1), β₁ = 0.1421.
Y en el modelo multivariado (Paso 4): p-valor = 0.1422 → **no significativa** por test de Wald.

> **Consecuencia:** `fbs` es la variable con menor poder explicativo del conjunto. Incluirla no mejora
> la predicción y puede aumentar la varianza de otros estimadores.

---

### Resumen del inconveniente principal

| Variable | Inconveniente | Efecto en el modelo |
|---|---|---|
| `restecg` | Categoría `= 1` con solo 4 observaciones | Estimador inestable, p = 0.198 |
| `fbs` | No discrimina la respuesta (Δ% = 3.6) | OR ≈ 1, p = 0.142, no significativa |

---

## Paso 5: ¿Describe el modelo la presencia de enfermedad cardiaca?

### Probabilidades predichas por el modelo multivariado

El modelo de regresión logística multivariado (con todas las variables) produce probabilidades
predichas que se evaluaron visualmente y con métricas de discriminación.

### Resultados

| Métrica | Valor |
|---|---|
| **AUC-ROC** | **0.9240** |
| **Accuracy** (umbral 0.5) | **84.2%** |

### Interpretación

**Sí, el modelo describe bien la presencia de enfermedad cardiaca.** Las evidencias son:

1. **AUC-ROC = 0.92:** Un AUC superior a 0.90 indica excelente poder discriminativo. El modelo
   separa correctamente a los pacientes con y sin enfermedad cardiaca en el 92% de los pares posibles.

2. **Separación visual clara:** El boxplot de probabilidades predichas muestra medianas claramente
   separadas entre los dos grupos:
   - Pacientes sin enfermedad (Y=0): probabilidades predichas concentradas en valores **bajos** (< 0.5)
   - Pacientes con enfermedad (Y=1): probabilidades predichas concentradas en valores **altos** (> 0.5)

3. **Accuracy = 84.2%:** Con un umbral simple de 0.5, el modelo clasifica correctamente a 8 de
   cada 10 pacientes.

4. **Variables con mayor aporte predictivo** (significativas por Wald, p < 0.05):
   - `ca` (nº vasos coloreados): OR = 3.45 → mayor nº de vasos afectados aumenta 3.4× el riesgo
   - `sex` (sexo masculino): OR = 3.94 → los hombres tienen ~4× más riesgo
   - `cp` (tipo de dolor): OR = 1.84 → dolor tipo 4 (asintomático) es el más informativo
   - `thal` (defecto de talasemia): OR = 1.38
   - `exang` (angina inducida): OR = 2.81

### Conclusión

El modelo logístico multivariado **sí describe adecuadamente** la presencia de enfermedad cardiaca.
Con un AUC de 0.92 y accuracy de 84%, el modelo tiene un desempeño sólido sobre los datos de
entrenamiento. Las probabilidades predichas se alinean con la variable respuesta real, lo que
confirma que las covariables seleccionadas capturan los factores de riesgo más relevantes para
la enfermedad cardiaca en esta población.