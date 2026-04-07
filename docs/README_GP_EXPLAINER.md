# Gaussian Process Regression: Theory and Implementation

**A comprehensive technical reference for the Gaussian Process (GP) regression system used to predict dissolved metal concentrations across the Phoenix metropolitan area.**

---

## Table of Contents

- [Motivation](#motivation)
- [What is a Gaussian Process?](#what-is-a-gaussian-process)
- [Mathematical Foundations](#mathematical-foundations)
  - [Prior Distribution over Functions](#prior-distribution-over-functions)
  - [Kernel (Covariance Function)](#kernel-covariance-function)
  - [Composite RBF Kernel Design](#composite-rbf-kernel-design)
  - [Covariance Matrix Construction](#covariance-matrix-construction)
  - [Observation Noise and Regularization](#observation-noise-and-regularization)
  - [Posterior Predictive Equations](#posterior-predictive-equations)
  - [Log Marginal Likelihood](#log-marginal-likelihood)
- [Implementation Pipeline](#implementation-pipeline)
  - [Coordinate Projection](#coordinate-projection)
  - [Target Normalization](#target-normalization)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Cholesky Decomposition and Numerical Stability](#cholesky-decomposition-and-numerical-stability)
  - [Training Procedure](#training-procedure)
  - [Prediction Procedure](#prediction-procedure)
  - [Safety Features](#safety-features)
- [Model Diagnostics](#model-diagnostics)
  - [Leave-One-Out Cross-Validation](#leave-one-out-cross-validation)
  - [Matrix Conditioning](#matrix-conditioning)
- [Design Decisions and Rationale](#design-decisions-and-rationale)
- [Worked Example](#worked-example)
- [References](#references)

---

## Motivation

Dissolved metals in urban water bodies (lakes, rivers, canals) serve as indicators of contamination from industrial runoff, aging infrastructure, and natural geological processes. Laboratory analysis of a single water sample costs $50 to $200 and requires days of processing. For a metropolitan area like Phoenix, spanning thousands of square kilometers, exhaustive sampling is economically impractical.

This project uses data from **6 real sampling sites** across the Phoenix metropolitan area to build a predictive spatial model using Gaussian Process regression. The GP produces continuous concentration predictions at any geographic coordinate, together with calibrated uncertainty estimates that communicate where the model is confident and where it is not.

The five analytes modeled are **Iron (Fe)**, **Chromium (Cr)**, **Manganese (Mn)**, **Molybdenum (Mo)**, and **Indium (In)**. Each analyte is modeled independently with its own optimized set of hyperparameters.

---

## What is a Gaussian Process?

A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution. In the context of regression, a GP defines a **distribution over functions** rather than parameterizing a single function form (as in linear or polynomial regression).

**Intuition via analogy:** Imagine stretching a flexible rubber sheet over a table and pushing pegs through it at six locations, each set to the measured metal concentration at that site. The sheet forms a smooth surface between the pegs. A GP is the mathematical equivalent — it finds the smoothest function consistent with (or near) the observed data, where "smoothness" is controlled by a kernel function.

When predicting at a new location, the GP computes a **weighted average** of the known values, with weights determined by spatial proximity through the kernel. Nearby sites contribute more; distant sites contribute less. Critically, the GP also outputs a **posterior variance** at each prediction point, which quantifies how uncertain the model is. Near training sites, uncertainty is nearly zero. Far from any training site, uncertainty approaches the prior variance.

---

## Mathematical Foundations

### Prior Distribution over Functions

$$
f(\mathbf{x}) \sim \mathcal{GP}\bigl(\mu(\mathbf{x}),\; k(\mathbf{x}, \mathbf{x}')\bigr)
$$

A GP is fully specified by a **mean function** $\mu(\mathbf{x})$ and a **kernel (covariance) function** $k(\mathbf{x}, \mathbf{x}')$.

In this implementation, $\mu(\mathbf{x}) = 0$ is used on normalized data. Before training, each analyte's observations are centered and scaled (see [Target Normalization](#target-normalization)), making the zero-mean prior assumption valid.

### Kernel (Covariance Function)

The kernel encodes assumptions about the spatial structure of the underlying function. The **Radial Basis Function (RBF)** kernel, also known as the squared exponential kernel, is defined as:

$$
k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert^2}{2 \ell^2}\right)
$$

where:
- $\sigma_f^2$ is the **signal variance**, controlling the vertical scale of variation
- $\ell$ is the **length scale**, controlling how quickly correlation decays with distance
- $\lVert \mathbf{x} - \mathbf{x}' \rVert$ is the Euclidean distance between two points

**Interpretation of the length scale:** A length scale of $\ell = 30$ km means that two points 30 km apart retain approximately 60% correlation ($e^{-0.5} \approx 0.607$). Points separated by $3\ell$ have effectively zero correlation.

### Composite RBF Kernel Design

This implementation uses a **composite kernel** consisting of two summed RBF components:

$$
k(\mathbf{x}, \mathbf{x}') = \underbrace{\sigma_f^2 \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert^2}{2 \ell^2}\right)}_{\text{Short-range component}} + \underbrace{\sigma_{\text{LR}}^2 \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert^2}{2 \ell_{\text{LR}}^2}\right)}_{\text{Long-range component}}
$$

| Parameter | Role | Value |
|-----------|------|-------|
| $\sigma_f$ | Short-range signal amplitude | Optimized via grid search from `[0.5, 1.0, 1.5, 2.0, 3.0, 5.0]` |
| $\ell$ | Short-range length scale (km) | Optimized from `[10, 20, 30, 50, 75, 100, 150, 200]` |
| $\sigma_{\text{LR}}$ | Long-range signal amplitude | **Fixed at 0.8** |
| $\ell_{\text{LR}}$ | Long-range length scale (km) | **Fixed at 150** |

**Why a composite kernel?** With only six training points spanning approximately 35 km, a single RBF kernel faces a dilemma. If the length scale is short enough to capture local variation between nearby sites, predictions in the gaps between distant sites collapse to the prior mean (zero in normalized space, which back-transforms to the training mean). If the length scale is long enough to maintain correlation across the full study area, the model cannot capture local differences.

The composite kernel resolves this by separating spatial variation into two scales:
- The **short-range component** captures local differences (e.g., one lake versus a nearby river)
- The **long-range component** maintains a smooth regional trend (e.g., the observed west-to-east concentration gradient)

The long-range parameters are fixed rather than optimized because there is insufficient data to reliably estimate both short-range and long-range hyperparameters simultaneously from six points.

### Covariance Matrix Construction

Given $n$ training points $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$, the covariance matrix $K$ is an $n \times n$ symmetric, positive semi-definite matrix:

$$
K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)
$$

For this project, $K$ is a $6 \times 6$ matrix. Each entry represents the kernel-evaluated spatial correlation between two training sites.

### Observation Noise and Regularization

Noise is added to the diagonal of the covariance matrix:

$$
\tilde{K} = K + \sigma_n^2 I + \epsilon I
$$

where:
- $\sigma_n^2$ is the **observation noise variance**, optimized from `[1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]`
- $\epsilon = 10^{-10}$ is a fixed **numerical jitter** for Cholesky stability

The noise term $\sigma_n^2$ accounts for genuine measurement uncertainty in the FAAS laboratory results. The jitter $\epsilon$ prevents the matrix from being numerically singular, which would cause the Cholesky decomposition to fail.

### Posterior Predictive Equations

Given a new query point $\mathbf{x}_*$, the GP posterior produces:

**Posterior mean (predicted concentration in normalized space):**

$$
\mu_* = \mathbf{k}_*^\top \tilde{K}^{-1} \mathbf{y}
$$

**Posterior variance (prediction uncertainty in normalized space):**

$$
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top \tilde{K}^{-1} \mathbf{k}_*
$$

where:
- $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), \ldots, k(\mathbf{x}_n, \mathbf{x}_*)]$ is the vector of covariances between the query point and each training point
- $k(\mathbf{x}_*, \mathbf{x}_*) = \sigma_f^2 + \sigma_{\text{LR}}^2$ is the prior variance at the query point

**Interpretation:** The posterior mean is a weighted sum of the training observations, where weights are derived from the spatial correlation structure encoded by the kernel. The posterior variance starts at the prior variance and shrinks proportionally to the correlation with nearby training points. Near a known site, uncertainty approaches zero. Far from all training sites, uncertainty approaches the prior variance.

### Log Marginal Likelihood

Hyperparameters are selected by maximizing the log marginal likelihood:

$$
\log p(\mathbf{y} \mid \mathbf{X}, \theta) = -\frac{1}{2} \mathbf{y}^\top \tilde{K}_\theta^{-1} \mathbf{y} - \frac{1}{2} \log |\tilde{K}_\theta| - \frac{n}{2} \log 2\pi
$$

This criterion has three terms:
1. **Data fit** ($-\frac{1}{2} \mathbf{y}^\top \tilde{K}^{-1} \mathbf{y}$): Penalizes models that do not explain the observed data well
2. **Complexity penalty** ($-\frac{1}{2} \log |\tilde{K}|$): Penalizes overly flexible models, serving as a built-in regularizer against overfitting
3. **Normalization constant** ($-\frac{n}{2} \log 2\pi$): Does not affect optimization

The log marginal likelihood implements an automatic Occam's razor — it favors the simplest model that adequately explains the data.

---

## Implementation Pipeline

### Coordinate Projection

Geographic coordinates (WGS-84 latitude and longitude) are projected to local kilometer units using an equirectangular approximation:

$$
x_{\text{km}} = (\text{lon} - \text{lon}_{\text{ref}}) \times 111.0 \times \cos(\text{lat}_{\text{ref}})
$$

$$
y_{\text{km}} = (\text{lat} - \text{lat}_{\text{ref}}) \times 111.0
$$

where $(\text{lat}_{\text{ref}}, \text{lon}_{\text{ref}})$ is the centroid of the six training sites.

**Why kilometers instead of raw degrees?** Kilometer units preserve true spatial distances and make the length-scale hyperparameter physically interpretable (e.g., $\ell = 30$ means "correlation drops significantly beyond 30 km"). Raw degrees would introduce anisotropy because one degree of longitude covers fewer kilometers than one degree of latitude at non-equatorial latitudes.

The $\cos(\text{lat})$ correction is adequate for Phoenix at approximately 33.5 degrees North. For study areas near the poles or spanning large latitude ranges, a UTM projection would be more appropriate.

### Target Normalization

Before training, each analyte's observations are z-score normalized:

$$
y_{\text{norm},i} = \frac{y_i - \bar{y}}{s_y}
$$

where $\bar{y}$ is the training mean and $s_y$ is the training standard deviation.

After prediction, outputs are denormalized:

$$
\hat{y} = \mu_* \cdot s_y + \bar{y}
$$

**Rationale:** The five analytes span vastly different concentration scales. Iron is measured in tens of mg/L while molybdenum is measured in thousandths of mg/L. Without normalization, a single set of kernel hyperparameters cannot serve all analytes. Z-score normalization places all analytes on a comparable scale, making the zero-mean prior assumption valid for each.

Negative predictions (which are physically impossible for concentration values) are clipped to zero after denormalization.

### Hyperparameter Optimization

The three free hyperparameters $\theta = (\sigma_f, \ell, \sigma_n)$ are optimized per analyte via exhaustive grid search:

| Parameter | Grid |
|-----------|------|
| $\sigma_f$ (signal amplitude) | `[0.5, 1.0, 1.5, 2.0, 3.0, 5.0]` |
| $\ell$ (length scale, km) | `[10, 20, 30, 50, 75, 100, 150, 200]` |
| $\sigma_n$ (noise std. dev.) | `[1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]` |

**Total candidates per analyte:** $6 \times 8 \times 6 = 288$

Grid search is chosen over gradient-based optimization because:
1. The log marginal likelihood surface is non-convex and may have local optima
2. With only 6 training points, each candidate evaluation takes microseconds
3. Grid search is deterministic and reproducible

The combination that maximizes the log marginal likelihood is selected.

### Cholesky Decomposition and Numerical Stability

Rather than computing $\tilde{K}^{-1}$ directly (which is numerically unstable for ill-conditioned matrices), the Cholesky decomposition is used:

1. **Decompose:** $\tilde{K} = L L^\top$ where $L$ is lower-triangular
2. **Forward-solve:** $L \mathbf{z} = \mathbf{y}$ to obtain $\mathbf{z}$
3. **Back-solve:** $L^\top \boldsymbol{\alpha} = \mathbf{z}$ to obtain $\boldsymbol{\alpha} = \tilde{K}^{-1} \mathbf{y}$
4. **Prediction mean:** $\mu_* = \mathbf{k}_*^\top \boldsymbol{\alpha}$
5. **Prediction variance:** Forward-solve $L \mathbf{v} = \mathbf{k}_*$, then $\sigma_*^2 = k_{**} - \mathbf{v}^\top \mathbf{v}$

**Diagnostic checks:**
- Eigenvalues are computed via the Jacobi iterative method; all must be positive (confirming positive definiteness)
- Condition number = $\lambda_{\max} / \lambda_{\min}$; safe if < $10^6$, a warning is logged if > $10^{10}$
- Jitter of $10^{-10}$ is always added to the diagonal

### Training Procedure

The full training procedure per analyte:

1. Extract raw observations $\mathbf{y}_{\text{raw}}$ for the analyte from the 6 sites
2. Compute training mean $\bar{y}$ and standard deviation $s_y$; normalize to $\mathbf{y}_{\text{norm}}$
3. Grid-search over $(\sigma_f, \ell, \sigma_n)$ to maximize log marginal likelihood
4. Build $\tilde{K}$ with best hyperparameters; compute Cholesky factor $L$
5. Solve for $\boldsymbol{\alpha} = \tilde{K}^{-1} \mathbf{y}_{\text{norm}}$ via forward/back substitution
6. Cache the trained model: $\{\mathbf{X}, \mathbf{y}_{\text{norm}}, \bar{y}, s_y, \boldsymbol{\alpha}, L, \theta\}$

All five models are trained at cold start and cached for subsequent requests. Training all five GPs takes less than 100 milliseconds.

### Prediction Procedure

For a query point $\mathbf{x}_*$:

1. Project to kilometers using the training centroid
2. Compute kernel vector $\mathbf{k}_*$ between the query and all training points
3. Compute posterior mean: $\mu_* = \mathbf{k}_*^\top \boldsymbol{\alpha}$ (normalized space)
4. Compute posterior variance: forward-solve $L \mathbf{v} = \mathbf{k}_*$, then $\sigma_*^2 = k_{**} - \mathbf{v}^\top \mathbf{v}$
5. Denormalize: $\hat{y} = \max(0, \; \mu_* \cdot s_y + \bar{y})$ and $\hat{\sigma} = \sqrt{\sigma_*^2} \cdot s_y$

### Safety Features

| Feature | Threshold | Behavior |
|---------|-----------|----------|
| Known-site snap | Query within 50 meters of a training site | Bypasses GP; returns exact measured values with zero uncertainty |
| Extrapolation flag | Nearest training site > 40 km | Sets `extrapolation: true` in response; frontend displays a warning |
| Negative clipping | Predicted mean < 0 | Clipped to 0 (concentrations cannot be negative) |

---

## Model Diagnostics

### Leave-One-Out Cross-Validation

LOOCV is the most rigorous validation procedure available for a six-point dataset. In each fold:

1. One site is removed from the training set
2. The GP is retrained on the remaining five sites with **full hyperparameter re-optimization**
3. The held-out site's concentration is predicted
4. The prediction error is recorded

This repeats for all six sites, producing six independent prediction errors per analyte.

**Key detail:** Hyperparameters are re-optimized in each fold. This prevents information leakage from the held-out site influencing the model configuration, providing a true out-of-sample assessment.

The LOOCV RMSE is reported per analyte and serves as the primary model quality metric.

### Matrix Conditioning

The diagnostic endpoint reports per-analyte matrix health:

| Metric | Computation | Acceptable Range |
|--------|-------------|-----------------|
| Eigenvalues | Jacobi iterative method on $\tilde{K}$ | All > 0 |
| Condition number | $\lambda_{\max} / \lambda_{\min}$ | < $10^6$ (warn > $10^{10}$) |
| Positive definiteness | $\min(\text{eigenvalues}) > 0$ | Must be `true` |

---

## Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Independent GPs per analyte (not multi-output) | Multi-output GPs require substantially more data to learn cross-covariance structure reliably. With 6 training points, independent models are more robust. |
| Composite kernel (not single RBF) | A single RBF cannot simultaneously capture local variation and maintain regional trends with only 6 points. |
| Fixed long-range parameters | Insufficient data to reliably optimize both short-range and long-range hyperparameters simultaneously. Fixing the long-range component reduces the optimization search space. |
| Grid search (not gradient-based optimization) | Non-convex likelihood surface; microsecond evaluation time per candidate makes exhaustive search feasible and deterministic. |
| Equirectangular projection (not UTM) | Adequate at Phoenix's latitude (33.5 degrees N); simpler to implement; physically interpretable length scales. |
| Z-score normalization (not min-max) | Preserves distributional properties; makes zero-mean prior assumption valid; robust to outliers. |

---

## Worked Example

This section walks through the complete GP pipeline using three synthetic training points, demonstrating every computation from raw data to final prediction.

### Setup

Three synthetic points in kilometer coordinates relative to a centroid:

| Point | x (km) | y (km) | Concentration (ppm) |
|-------|--------|--------|---------------------|
| P1 | 8.41 | -7.13 | 384.6 |
| P2 | -15.92 | 12.38 | 178.3 |
| P3 | 3.66 | 18.74 | 451.2 |

### Step 1: Normalize Targets

$$
\bar{y} = \frac{384.6 + 178.3 + 451.2}{3} = 338.033
$$

$$
s_y = \sqrt{\frac{(384.6 - 338.033)^2 + (178.3 - 338.033)^2 + (451.2 - 338.033)^2}{3}} = 115.193
$$

$$
\mathbf{y}_{\text{norm}} = [0.4041,\; -1.3870,\; 0.9822]
$$

### Step 2: Compute Pairwise Distances

$$
d_{12} = \sqrt{(8.41 + 15.92)^2 + (-7.13 - 12.38)^2} = 31.17 \text{ km}
$$

$$
d_{13} = \sqrt{(8.41 - 3.66)^2 + (-7.13 - 18.74)^2} = 26.29 \text{ km}
$$

$$
d_{23} = \sqrt{(-15.92 - 3.66)^2 + (12.38 - 18.74)^2} = 20.60 \text{ km}
$$

### Step 3: Build Covariance Matrix

Using $\sigma_f = 1.0$, $\ell = 30$ km, $\sigma_n = 0.001$:

| Pair | Distance | Short-range $k_s$ | Long-range $k_l$ | Total |
|------|----------|-------------------|-------------------|-------|
| (1,1) | 0 | 1.0000 | 0.6400 | 1.6400 |
| (1,2) | 31.17 | 0.5786 | 0.6262 | 1.2048 |
| (1,3) | 26.29 | 0.6884 | 0.6302 | 1.3186 |
| (2,3) | 20.60 | 0.7919 | 0.6340 | 1.4259 |

Adding noise + jitter to diagonal:

$$
\tilde{K} = \begin{bmatrix}
1.640001 & 1.2048 & 1.3186 \\
1.2048 & 1.640001 & 1.4259 \\
1.3186 & 1.4259 & 1.640001
\end{bmatrix}
$$

### Step 4: Cholesky Decomposition

$$
L = \begin{bmatrix}
1.28063 & 0 & 0 \\
0.94080 & 0.86884 & 0 \\
1.02965 & 0.52601 & 0.55058
\end{bmatrix}
$$

### Step 5: Solve for Alpha

Forward-solve $L\mathbf{z} = \mathbf{y}_{\text{norm}}$, then back-solve $L^\top \boldsymbol{\alpha} = \mathbf{z}$:

$$
\boldsymbol{\alpha} = [-0.10333,\; -5.57808,\; 5.53108]
$$

### Step 6: Predict at the Centroid

Query point $\mathbf{x}_* = (0, 0)$ (the centroid of training data):

Distances to training points: 11.03, 20.17, 19.10 km

Kernel vector: $\mathbf{k}_* = [1.5729,\; 1.4319,\; 1.4511]$

Posterior mean (normalized):

$$
\mu_* = 1.5729 \times (-0.10333) + 1.4319 \times (-5.57808) + 1.4511 \times 5.53108 = -0.1244
$$

### Step 7: Denormalize

$$
\hat{y} = \max(0,\; -0.1244 \times 115.193 + 338.033) = 323.7 \text{ ppm}
$$

The model predicts **323.7 ppm** at the centroid, close to the overall mean of 338.0 ppm. This is expected because the centroid is roughly equidistant from all three training points.

---

## References

1. Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. Available at [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/).

2. Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes.* PhD thesis, University of Cambridge.

3. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction.* MIT Press.

---

*Technical reference for a science-fair project investigating spatial interpolation of dissolved metal concentrations in Phoenix-area water bodies using Gaussian Process regression with 6 training sites.*
