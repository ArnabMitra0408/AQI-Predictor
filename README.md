# AQI Predictor: Ensemble vs. Hybrid Models

## Overview
This project investigates advanced modeling approaches for hourly Air Quality Index (AQI) forecasting. We compare two methodologies:
- **Ensemble Model:** Averages predictions from a Random Forest and an LSTM model.
- **Hybrid Model:** Uses Random Forest for initial predictions and LSTM to refine the residual errors.

The hybrid approach outperforms the ensemble model in accuracy, generalization, and robustness.

## Data
We use two datasets:
1. **Air Quality Data:** Sourced from OpenWeatherMap API (2023-10-13 to 2024-10-13), covering 50 U.S. states with pollutant concentrations (CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3) and AQI levels.
2. **Weather Data:** Sourced from OpenMeteo API (same date range), including temperature, humidity, rainfall, wind speed, and soil moisture.

These datasets were preprocessed, merged, and standardized for modeling.

## Novel Metrics
To evaluate the models, we introduce:

### Health Risk Assessment Score (HRAS)
Assesses model accuracy in predicting AQI health risk categories based on WHO guidelines.

\[ HRAS = \frac{\text{Number of Correctly Predicted Risk Levels}}{\text{Total Predictions}} \]

### Hourly/Temporal Prediction Accuracy (HTPA)
Measures accuracy across different hours, accounting for time-based variations like rush hour pollution.

\[ HTPA = \frac{\text{Correct Hourly Predictions}}{\text{Total Hourly Predictions}} \]

## Model Architectures
### Hybrid Model
1. **Random Forest** (200 estimators, max depth = 20, min samples split = 5) predicts initial AQI values.
2. **LSTM Model** (2 LSTM layers with 64 hidden units) learns temporal patterns from Random Forest residual errors.
3. The final AQI prediction is obtained by adjusting Random Forest outputs using LSTM corrections.

### Ensemble Model
1. **Random Forest** and **LSTM** trained separately.
2. Final prediction is computed as the average of both model outputs.

## Results
The hybrid model significantly outperforms the ensemble model:

| Metric  | Ensemble Model | Hybrid Model |
|---------|---------------|--------------|
| R²      | 0.74864       | 0.99985      |
| MAE     | 3.42725       | 0.15967      |
| HTPA    | 0.81510       | 0.99968      |
| HRAS    | 0.93352       | 0.99645      |

## Conclusion
- The **Hybrid Model** provides superior AQI predictions by combining feature selection (Random Forest) with temporal learning (LSTM).
- The **Ensemble Model** offers efficiency but struggles with overfitting and lower generalization.
- Future work will explore scaling to larger datasets and additional hybridization strategies.

## Repository Structure
```
├── data/                # Processed datasets
├── models/              # Trained model weights
├── src/                 # Model training scripts
├── results/             # Performance metrics & visualizations
├── README.md            # Project documentation
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ArnabMitra0408/AQI-Hourly-Prediction.git
   cd AQI-Hourly-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models:
   ```bash
   python train_hybrid.py
   python train_ensemble.py
   ```
4. Evaluate performance:
   ```bash
   python evaluate.py
   ```
