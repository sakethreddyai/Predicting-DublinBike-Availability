# Predicting DublinBike Availability

**Spatio-temporal graph neural networks for short-term DublinBikes availability forecasting, enhanced with urban POI context and Explainable AI.**

## Overview

This project explores the prediction of bike availability across the DublinBikes network using **Spatio-Temporal Graph Convolutional Networks (ST-GCNs)**.

DublinBikes stations are represented as nodes in a spatial graph, allowing the models to learn both **temporal patterns in station availability** and **spatial relationships between stations**. The project further investigates whether **Points of Interest (POIs)** around stations can improve prediction performance.

The project also incorporates **Explainable AI (XAI)** to analyse which temporal features, POI categories, and neighbouring stations contribute to model predictions.

## Model Architecture

```text
DublinBikes Data
       ↓
Data Processing
       ↓
Station Graph ─────── POI / H3 Context
       ↓                    ↓
       └───────┬────────────┘
               ↓
        ST-GCN / AST-GCN
               ↓
        POI-STGCN / POI-ASTGCN
               ↓
     30-min Availability Forecast
               ↓
       ┌───────┴────────┐
       ↓                ↓
   Performance        XAI Analysis
    Evaluation       Temporal / POI /
                     Spatial Features
```

## Key Results

| Model             |   Test MAE |
| ----------------- | ---------: |
| Naive Persistence |    11.5031 |
| ST-GCN            |     3.8842 |
| AST-GCN           |     3.0828 |
| POI-STGCN         |     1.2425 |
| **POI-ASTGCN**    | **1.0639** |

The best-performing **POI-ASTGCN** reduced MAE by **90.75% compared with the naive persistence baseline**.

### Model Comparison

![Model Comparison](Predicting%20DublinBike%20Availability%20LateX/figures/model_comparison_all.png)

## Approach

The project compares four graph-based forecasting approaches:

* **ST-GCN:** captures spatial and temporal dependencies across stations.
* **AST-GCN:** introduces temporal attention to focus on more informative historical observations.
* **POI-STGCN:** incorporates static Points of Interest as additional station-level context.
* **POI-ASTGCN:** combines POI context with temporal attention.

The final dataset contains **115 DublinBikes stations**, with historical observations sampled at **10-minute intervals**. The models use **3 hours of historical observations to forecast the following 30 minutes**.

## Explainable AI

Prediction accuracy is only part of the evaluation. The project uses explainability techniques to investigate how the models make their predictions.

* **Integrated Gradients** for temporal and POI feature attribution.
* **PGExplainer-style analysis** for spatial/edge-level explanations.
* Station-level error analysis to identify differences in model reliability across the network.

These analyses help investigate whether the models rely on meaningful temporal, spatial, and urban-context information.

### Temporal Feature Importance

![Temporal Importance](Predicting%20DublinBike%20Availability%20LateX/figures/temporal_importance_comparison.png)

### POI Feature Attribution

![POI Attribution](Predicting%20DublinBike%20Availability%20LateX/figures/poi_feature_attribution.png)

## Dataset

Historical DublinBikes data was collected through the **Smart Dublin DublinBikes API**, covering **May 2024 to August 2025**.

The final modelling dataset contains:

* 115 stations
* 10-minute observation intervals
* 18 historical time steps as model input
* 3 future time steps as prediction targets
* 70/15/15 chronological train, validation, and test split

## Technologies

* **Python**
* **PyTorch**
* **PyTorch Geometric**
* **Pandas / NumPy**
* **Scikit-learn**
* **Captum**
* **H3**
* **Matplotlib**
* **Jupyter Notebook**

## Repository Contents

```text
├── Predicting DublinBike Availability.ipynb
├── Predicting DublinBike Availability.pdf
├── Predicting DublinBike Availability Code.pdf
│
└── Predicting DublinBike Availability LateX/
    ├── main.tex
    ├── references.bib
    └── figures/
```

The Jupyter notebook contains the project implementation and experiments, while the accompanying PDF and LaTeX source provide the full research documentation.

## Research Focus

This project was developed as part of a **Master's thesis in Human-Centred Artificial Intelligence at TU Dublin**.

The work focuses on combining:

**Spatio-Temporal Graph Neural Networks + Urban Context + Explainable AI**

to improve both the predictive performance and interpretability of bike-sharing availability forecasting.

## Future Work

Potential extensions include:

* Incorporating dynamic contextual features such as weather and events.
* Evaluating the approach across longer forecasting horizons.
* Improving uncertainty estimation and station-level reliability analysis.
* Exploring real-time deployment for operational bike redistribution.

## Author

**Saketh Reddy Chandupatla**

Master's in Human-Centred Artificial Intelligence
Technological University Dublin
