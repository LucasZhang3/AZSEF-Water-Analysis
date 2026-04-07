<div align="center">
  <img src="AZSEF Logo.jpg" alt="AZSEFLogo" width="499" height="165">
</div>

# Water Quality Prediction in Arizona Using Gaussian Process Regression

**Predicting dissolved metal concentrations across metropolitan Phoenix using Flame Atomic Absorption Spectroscopy and Gaussian Process regression.**


**Category:** Earth and Environmental Sciences 


## Overview

Laboratory analysis of dissolved metals in water costs between $50 and $200 per sample. For a metropolitan water system spanning thousands of square kilometers, comprehensive spatial mapping through direct measurement alone is economically impractical. This project demonstrates that a small number of strategically collected water samples - just six sites - combined with a machine learning technique called Gaussian Process (GP) regression, can produce spatially continuous predictions of dissolved metal concentrations across the entire Phoenix, Arizona metropolitan area, along with principled uncertainty estimates that identify where additional sampling would be most valuable.

Water samples were collected from six sites across the Phoenix metro area, analyzed for five dissolved metals using Flame Atomic Absorption Spectroscopy (FAAS) at Arizona State University's Goldwater Environmental Laboratory, and used to train five independent GP models. The trained models were then deployed as a real-time interactive web application where users can click anywhere on a map of Phoenix and receive instant concentration predictions with calibrated confidence intervals.

This project was developed for competition at the Arizona Science and Engineering Fair (AzSEF), under the category of Earth and Environmental Sciences.

---

## Table of Contents

- [The Problem](#the-problem)
- [Data Collection](#data-collection)
- [Analytical Chemistry: FAAS](#analytical-chemistry-faas)
- [Measured Concentrations](#measured-concentrations)
- [Machine Learning Model: Gaussian Process Regression](#machine-learning-model-gaussian-process-regression)
- [Model Validation (LOOCV)](#model-validation-loocv)
- [Interactive Web Application](#interactive-web-application)
- [Application Demo](#application-demo)
- [Project Structure](#project-structure)
- [Science Fair Context](#science-fair-context)
- [Key Findings](#key-findings)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)
- [License](#license)

---

## The Problem

Dissolved metals in urban water bodies originate from industrial discharge, aging infrastructure, and geological weathering. Several are regulated under EPA National Primary and Secondary Drinking Water Standards due to their effects on human health and aquatic ecosystems. Despite their significance, current monitoring relies on sparse, site-specific laboratory analyses that leave large geographic gaps in our understanding of water quality.

The five analytes investigated in this project:

| Metal | Symbol | Environmental Significance |
|-------|--------|---------------------------|
| Iron | Fe | Secondary drinking water standard (aesthetic); indicator of pipe corrosion |
| Chromium | Cr | EPA MCL of 0.1 mg/L; hexavalent form is a known carcinogen |
| Manganese | Mn | Secondary standard at 0.05 mg/L; neurotoxic at elevated concentrations |
| Molybdenum | Mo | EPA health advisory level; industrial tracer |
| Indium | In | Emerging contaminant from electronics manufacturing |

> **Research Question:** How accurately can a Gaussian Process model predict dissolved metal concentrations across a metropolitan area using only six sampling sites?

---

## Data Collection

Six sampling sites were selected across the Phoenix metropolitan area to capture a range of land uses, hydrological contexts, and potential contamination sources. Sites span approximately 35 kilometers east to west.

| Site | Latitude | Longitude | Context |
|------|----------|-----------|---------|
| Alvord Lake | 33.3747 | -112.1368 | Urban lake, western Phoenix |
| Desert West Lake | 33.4757 | -112.1964 | Urban lake, northwestern Phoenix |
| Steele Indian School Park | 33.4996 | -112.0702 | Central Phoenix urban park |
| Gila River (Estrella) | 33.3851 | -112.3055 | River system, southwestern extent |
| Papago Park | 33.4531 | -111.9472 | Eastern Phoenix, near desert |
| Tres Rios Wetlands | 33.3959 | -112.2593 | Constructed wetland, western Phoenix |

### Sample Preparation (EPA Method 7000B)

All water samples were prepared following standard protocols:

1. Filtered through 0.45 micrometer membrane filters to isolate dissolved metals from particulate matter.
2. Acidified with concentrated nitric acid (HNO3) to pH less than 2 for preservation.
3. Stored at 4 degrees Celsius and transported under chain of custody to the ASU Goldwater Environmental Laboratory.

---

## Analytical Chemistry: FAAS

### How Flame Atomic Absorption Spectroscopy Works

Flame Atomic Absorption Spectroscopy (FAAS) is a well-established analytical technique for quantifying dissolved metals in aqueous samples. The process works as follows:

1. A liquid sample is aspirated into a nebulizer and sprayed as a fine aerosol into an air-acetylene flame at approximately 2,300 degrees Celsius.
2. The thermal energy of the flame atomizes the sample, breaking chemical bonds and producing free ground-state atoms.
3. A hollow cathode lamp (HCL), specific to the element being measured, emits a narrow beam of light at the characteristic absorption wavelength of that element.
4. Ground-state atoms in the flame absorb photons from the HCL beam. The amount of light absorbed is directly proportional to the concentration of the target metal in the sample.
5. A monochromator isolates the analytical wavelength, and a detector measures the transmitted light intensity.

### The Beer-Lambert Law

Concentration is determined from absorbance measurements using the Beer-Lambert Law:

> **A = epsilon times l times c**
>
> where A is the measured absorbance, epsilon is the molar absorptivity (an element-specific constant), l is the optical path length through the flame, and c is the concentration of the dissolved metal.

### Calibration

Five-point calibration curves were constructed from certified standard solutions for each metal. Concentrations of unknown samples were determined by interpolation against these curves.

| Metal | Calibration Equation | R-squared |
|-------|---------------------|-----------|
| Iron (Fe) | A = 0.008c + 0.002 | 0.998 |
| Chromium (Cr) | A = 0.042c + 0.003 | 0.996 |
| Manganese (Mn) | A = 0.035c + 0.001 | 0.997 |
| Molybdenum (Mo) | A = 0.028c + 0.004 | 0.994 |
| Indium (In) | A = 0.019c + 0.002 | 0.995 |

All calibration curves achieved R-squared values of 0.994 or higher, confirming excellent linearity across the working concentration range.

### Quality Control

- **Triplicate measurements:** Every sample was measured three times per metal, yielding 90 total absorbance readings across all sites and analytes.
- **Dilution:** Iron samples at all six sites required 10x dilution because raw concentrations exceeded the upper bound of the linear calibration range.
- **Detection limits:** Indium concentrations at three of six sites fell below the instrument's limit of detection (LOD) and were recorded as such.

---

## Measured Concentrations

| Metal | Concentration Range (mg/L) | Key Observation |
|-------|---------------------------|-----------------|
| Iron (Fe) | 26.9 to 73.2 | Highest absolute concentrations; required dilution at all sites |
| Chromium (Cr) | 0.019 to 0.198 | Strong west-to-east decreasing gradient |
| Manganese (Mn) | 0.003 to 0.211 | Co-varies spatially with chromium |
| Molybdenum (Mo) | 0.0031 to 0.0164 | Co-varies with chromium and manganese |
| Indium (In) | Below LOD to 0.0187 | Below detection at three of six sites |

### Spatial Pattern

A clear geographic trend emerged from the data: chromium, manganese, and molybdenum concentrations are highest at western Phoenix sites (Alvord Lake, Tres Rios Wetlands, Desert West Lake) and decrease systematically moving eastward (Steele Indian School Park, Papago Park). This co-variation across three independent analytes suggests a common industrial or geological source in the western portion of the metropolitan area.

---

## Machine Learning Model: Gaussian Process Regression

A Gaussian Process is a non-parametric Bayesian regression method that defines a probability distribution over functions. Given a set of observed data points, the GP computes a posterior distribution that provides both a best estimate (the posterior mean) and a measure of confidence (the posterior standard deviation) at any query location. This makes it particularly well-suited for spatial interpolation with sparse data, because the model explicitly communicates where its predictions are trustworthy and where they are not.

Five independent GP models were trained, one for each dissolved metal. For a comprehensive technical treatment of the GP theory and implementation, see the [Gaussian Process Explainer](docs/README_GP_EXPLAINER.md).

### Pipeline

**1. Coordinate Projection**

Latitude and longitude coordinates are converted to kilometers using an equirectangular projection centered on the training data centroid. This preserves distance relationships at Arizona's latitude (approximately 33.5 degrees North) and provides the GP kernel with physically meaningful distance units.

**2. Target Normalization**

Measured concentrations are z-score normalized per element (zero mean, unit variance). This is essential because the five metals span vastly different concentration scales - iron is measured in tens of mg/L while molybdenum is measured in thousandths of mg/L. Normalization allows consistent kernel hyperparameter optimization across all analytes.

**3. Composite Kernel**

Each GP uses a sum of two Radial Basis Function (RBF) kernels:

> **k(x, x') = sigma_f_short squared times exp(-||x - x'|| squared / (2 times l_short squared)) + sigma_f_long squared times exp(-||x - x'|| squared / (2 times l_long squared))**

- A **short-range RBF** with optimized signal variance and length scale, capturing local spatial variation near sampling sites.
- A **long-range RBF** with fixed signal variance of 0.8 and length scale of 150 km, providing smooth background trends and preventing predictions from collapsing to the prior mean in the gaps between distant training sites.

The composite kernel architecture is critical for a six-point dataset. Without the long-range component, the model would produce unreasonably low predictions in regions equidistant from two training sites; without the short-range component, it could not capture local variation.

**4. Hyperparameter Optimization**

For each metal, a grid search evaluates 288 combinations of three hyperparameters:

| Hyperparameter | Symbol | Grid Values |
|---------------|--------|-------------|
| Signal variance (short-range) | sigma_f | 6 values |
| Length scale (short-range) | l | 8 values |
| Noise variance | sigma_n | 6 values |

The combination that maximizes the log marginal likelihood is selected. The log marginal likelihood is a principled Bayesian criterion that automatically balances data fit against model complexity, providing a built-in guard against overfitting.

**5. Posterior Inference**

For any query point, the GP computes:

- **Posterior mean:** The best estimate of the metal concentration at that location, computed as a weighted combination of training observations where weights are determined by the kernel-derived spatial correlation structure.
- **Posterior standard deviation:** A calibrated uncertainty estimate that increases monotonically with distance from the nearest training site.

All posterior inference uses cached Cholesky decomposition of the kernel matrix for numerical stability and computational efficiency.

**6. Safety Features**

- Predictions at locations beyond 40 km from the nearest training site are explicitly flagged as **extrapolations**.
- Negative predictions (which are physically impossible for concentration values) are clipped to zero.
- Query points within 50 meters of a known sampling site return the measured values directly with zero uncertainty.

---

## Model Validation (LOOCV)

Model accuracy was assessed using Leave-One-Out Cross-Validation, the most rigorous validation procedure possible with six data points. In each fold:

1. One site is removed from the training set.
2. The GP is retrained on the remaining five sites with full hyperparameter re-optimization.
3. The held-out site's concentration is predicted.
4. The prediction is compared to the actual measured value.

This process repeats for all six sites, producing six independent prediction errors per analyte.

| Metal | Validation Result |
|-------|-------------------|
| Chromium (Cr) | Predictions closely track held-out measurements; captures west-east gradient |
| Manganese (Mn) | Predictions closely track held-out measurements; strong spatial structure |
| Iron (Fe) | Largest absolute RMSE (~30,000 ppm), consistent with its wide concentration range |
| Molybdenum (Mo) | Captures overall trend despite narrow concentration range |
| Indium (In) | Weakest performance due to below-LOD values at half the training sites |

---

## Interactive Web Application

The trained GP model is deployed as a real-time interactive web application. The backend performs full Cholesky decomposition and posterior inference on every request - there are no pre-computed grids or lookup tables.

### Features

- **Click-to-predict:** Click anywhere on a Google Map of the Phoenix metropolitan area to receive instant predictions for all five dissolved metals at that location.
- **Uncertainty visualization:** Each prediction displays a plus-or-minus one standard deviation confidence interval, color-coded by confidence level (green for high confidence, yellow for moderate, red for low confidence).
- **Known site display:** The six sampling sites are shown on the map with their actual measured concentrations and LOOCV prediction errors.
- **Extrapolation warnings:** Queries beyond 40 km from any training site trigger a visible warning and are labeled as extrapolated predictions.
- **Dark mode:** Full light and dark theme support.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend framework | React 18 with TypeScript |
| Styling | Tailwind CSS |
| Map rendering | Google Maps JavaScript API |
| Backend compute | Serverless functions (Deno runtime) |
| GP implementation | Custom from-scratch implementation with Cholesky decomposition |

### Architecture

The application follows a clean client-server architecture. The React frontend captures user map interactions and sends latitude/longitude coordinates to a serverless backend function. The backend function runs the full GP posterior inference pipeline - coordinate projection, kernel evaluation, Cholesky solve, de-normalization - and returns a JSON response containing predicted concentrations, uncertainty estimates, nearest site information, and extrapolation flags. The frontend then renders the results in real time.

---

## Application Demo

### Screenshots

<!-- Add your application screenshots below. Place image files in the assets/ folder. -->

#### Light Mode

![Application Screenshot - Light Mode](assets/screenshot_light.png)

#### Dark Mode

![Application Screenshot - Dark Mode](assets/screenshot_dark.png)

#### Prediction Results Panel

![Prediction Results](assets/screenshot_predictions.png)

#### Extrapolation Warning

![Extrapolation Warning](assets/screenshot_extrapolation.png)

### Demo Video

<!-- Add your demo video or GIF below. Place video files in the demo/ folder. -->

![Demo Walkthrough](demo/demo.gif)

> A full demo recording is available in the [demo/](demo/) folder.

---

## Spatial Visualizations

The following visualizations were generated from the GP model predictions and are available in the [assets/](assets/) folder.

### Heatmaps

<!-- Add heatmap images below -->

![Concentration Heatmap](assets/heatmap.png)

### Residual Analysis

<!-- Add residual plot images below -->

![Residual Plots](assets/residuals.png)

### LOOCV Plots

<!-- Add LOOCV plot images below -->

![LOOCV Results](assets/loocv.png)

---

## Project Structure

```
arizona-water-quality-gp/
├── assets/                     # Heatmaps, residual plots, LOOCV plots, app screenshots
├── demo/                       # Demo video and recording files
├── materials/                  # Research plan, quad chart, presentation, trifold
├── docs/
│   └── README_GP_EXPLAINER.md  # Comprehensive technical guide to the GP implementation
└── README.md                   # This file
```

---

## Science Fair Context

This project was developed for presentation at the **Arizona Science and Engineering Fair (AzSEF)**.

The project demonstrates the integration of wet-lab analytical chemistry with computational modeling and software engineering. The research question - whether sparse-data Gaussian Process regression can meaningfully predict dissolved metal concentrations across a metropolitan water system - was motivated by the practical constraint that comprehensive direct measurement is prohibitively expensive.

The interactive web application serves as both a research tool and a demonstration platform, allowing judges and viewers to explore the model's predictions and uncertainty behavior in real time.

---

## Key Findings

- Chromium, manganese, and molybdenum concentrations co-vary across the Phoenix metro area, with elevated levels in western Phoenix and low levels in eastern Phoenix, suggesting a common contamination source.
- Gaussian Process regression with a composite RBF kernel successfully interpolates between six sampling sites, producing spatially smooth and physically plausible concentration maps.
- The model's uncertainty estimates are well-calibrated: posterior standard deviation increases monotonically with distance from training sites, providing an honest assessment of prediction reliability.
- LOOCV confirms that chromium and manganese exhibit the strongest predictable spatial structure among the five analytes.
- Indium is the most challenging analyte to model due to below-detection-limit values at half the training sites.
- The GP framework naturally identifies high-uncertainty regions where additional sampling would be most informative, enabling cost-effective environmental monitoring through active learning principles.

---

## Limitations and Future Work

### Current Limitations

- **Sample size:** Six training points is the minimum viable dataset for spatial GP modeling. The model cannot capture fine-grained spatial variation that may exist between sites.
- **Temporal snapshot:** All samples were collected during a single sampling campaign. The model does not account for seasonal or temporal variation in metal concentrations.
- **Two-dimensional inputs:** The model uses only geographic coordinates as inputs. Incorporating additional covariates (elevation, proximity to industrial sites, land use classification, upstream hydrology) could improve prediction accuracy.
- **Single analytical method:** FAAS, while reliable and well-characterized, has higher detection limits than ICP-MS for trace metals, contributing to the below-LOD indium measurements.

### Future Directions

- **Expanded sampling network:** Additional sites, particularly in underrepresented areas identified by the model's high-uncertainty regions, would improve spatial resolution and model confidence.
- **Multi-output GP:** A correlated multi-output GP could exploit the observed co-variation between Cr, Mn, and Mo to improve predictions for all three metals simultaneously.
- **Temporal modeling:** Repeated sampling over multiple seasons would enable spatiotemporal GP models that capture both geographic patterns and seasonal dynamics.
- **Additional analytes:** Extending the analysis to include lead, arsenic, copper, and zinc would provide a more comprehensive water quality assessment.
- **ICP-MS validation:** Cross-validating FAAS measurements against inductively coupled plasma mass spectrometry would provide independent confirmation and lower detection limits.

---

## References

1. Baird, R. B., Eaton, A. D., and Rice, E. W. (2017). *Standard Methods for the Examination of Water and Wastewater*, 23rd edition. American Public Health Association, American Water Works Association, and Water Environment Federation.

2. Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. Available at [gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml/).

3. Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes*. PhD thesis, University of Cambridge.

4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.

5. Harris, D. C. (2010). *Quantitative Chemical Analysis*, 8th edition. W. H. Freeman and Company.

6. U.S. Environmental Protection Agency. (2007). *Method 7000B: Flame Atomic Absorption Spectrophotometry*. SW-846, Revision 2.

---

## License

This project is made available for educational and research purposes. Please contact the author for licensing inquiries.

---

*Developed by Lucas Zhang for the Arizona Science and Engineering Fair.*
