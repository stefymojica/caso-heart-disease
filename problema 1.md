# Problema 1: Familia Exponencial de Distribuciones

## Definición

Una distribución pertenece a la **familia exponencial** si su fmp/fdp puede escribirse como:

$$p(x|\theta) = h(x)\exp(\eta(\theta)\,t(x) - a(\theta))$$

donde:
- $h(x)$: función base (solo depende de $x$)
- $\eta(\theta)$: parámetro natural
- $t(x)$: estadístico suficiente
- $a(\theta)$: función log-partición (normalizadora)

---

## 1. Distribución Bernoulli

**Contexto:** usada en regresión logística, modela eventos binarios $x \in \{0, 1\}$.

**Forma estándar:**

$$p(x|p) = p^x(1-p)^{1-x}$$

**Derivación:**

Aplicamos logaritmo y exponencial:

$$p(x|p) = \exp\left(x\log p + (1-x)\log(1-p)\right)$$

$$= \exp\left(x\log p - x\log(1-p) + \log(1-p)\right)$$

$$= \exp\left(x\log\frac{p}{1-p} + \log(1-p)\right)$$

**Identificación de componentes:**

| Componente | Expresión |
|---|---|
| $h(x)$ | $1$ |
| $\eta(\theta)$ | $\log\dfrac{p}{1-p}$ (log-odds) |
| $t(x)$ | $x$ |
| $a(\theta)$ | $-\log(1-p) = \log(1 + e^\eta)$ |

**Conclusión:** La Bernoulli pertenece a la familia exponencial. $\checkmark$

---

## 2. Distribución Normal

**Contexto:** usada en regresión lineal. Asumimos $\sigma^2$ conocida, $\theta = \mu$.

**Forma estándar:**

$$p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Derivación:** expandimos el cuadrado en el exponente:

$$-\frac{(x-\mu)^2}{2\sigma^2} = -\frac{x^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} - \frac{\mu^2}{2\sigma^2}$$

Entonces:

$$p(x|\mu) = \underbrace{\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{x^2}{2\sigma^2}\right)}_{h(x)} \cdot \exp\left(\frac{\mu}{\sigma^2}\cdot x - \frac{\mu^2}{2\sigma^2}\right)$$

**Identificación de componentes:**

| Componente | Expresión |
|---|---|
| $h(x)$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\dfrac{x^2}{2\sigma^2}\right)$ |
| $\eta(\theta)$ | $\dfrac{\mu}{\sigma^2}$ |
| $t(x)$ | $x$ |
| $a(\theta)$ | $\dfrac{\mu^2}{2\sigma^2}$ |

**Conclusión:** La Normal pertenece a la familia exponencial. $\checkmark$

---

## 3. Distribución Poisson

**Contexto:** usada en regresión Poisson para conteos $x \in \{0, 1, 2, \ldots\}$.

**Forma estándar:**

$$p(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$$

**Derivación:**

$$p(x|\lambda) = \frac{1}{x!}\exp\left(x\log\lambda - \lambda\right)$$

**Identificación de componentes:**

| Componente | Expresión |
|---|---|
| $h(x)$ | $\dfrac{1}{x!}$ |
| $\eta(\theta)$ | $\log\lambda$ |
| $t(x)$ | $x$ |
| $a(\theta)$ | $\lambda = e^\eta$ |

**Conclusión:** La Poisson pertenece a la familia exponencial. $\checkmark$

---

## Resumen comparativo

| Distribución | $h(x)$ | $\eta(\theta)$ | $t(x)$ | $a(\theta)$ |
|---|---|---|---|---|
| Bernoulli | $1$ | $\log\frac{p}{1-p}$ | $x$ | $\log(1+e^\eta)$ |
| Normal ($\sigma^2$ fija) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-x^2/2\sigma^2}$ | $\mu/\sigma^2$ | $x$ | $\mu^2/2\sigma^2$ |
| Poisson | $1/x!$ | $\log\lambda$ | $x$ | $e^\eta$ |

Las tres distribuciones admiten la forma canónica $h(x)\exp(\eta(\theta)t(x) - a(\theta))$, por lo tanto **pertenecen a la familia exponencial de distribuciones**.
