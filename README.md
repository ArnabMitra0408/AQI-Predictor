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

$$ HRAS = \frac{\text{Number of Correctly Predicted Risk Levels}}{\text{Total Predictions}} $$

### Hourly/Temporal Prediction Accuracy (HTPA)
Measures accuracy across different hours, accounting for time-based variations like rush hour pollution.

$$ HTPA = \frac{\text{Correct Hourly Predictions}}{\text{Total Hourly Predictions}} $$

## Model Architectures
### Hybrid Model
1. **Random Forest** (200 estimators, max depth = 20, min samples split = 5) predicts initial AQI values.
2. **LSTM Model** (2 LSTM layers with 64 hidden units) learns temporal patterns from Random Forest residual errors.
3. The final AQI prediction is obtained by adjusting Random Forest outputs using LSTM corrections.
![Hybrid_Model_Architecture](https://github.com/ArnabMitra0408/AQI-Predictor/blob/main/Plots_And_Metrics/Hybrid_Model_Architecture.png)

### Ensemble Model
1. **Random Forest** and **LSTM** trained separately.
2. Final prediction is computed as the average of both model outputs.

![Ensemble_Model_Architecture](https://github.com/ArnabMitra0408/AQI-Predictor/blob/main/Plots_And_Metrics/Ensemble_Model_Architecture.png)

## Results
The hybrid model significantly outperforms the ensemble model:

| Metric  | Ensemble Model | Hybrid Model |
|---------|---------------|--------------|
| R²      | 0.74864       | 0.99985      |
| MAE     | 3.42725       | 0.15967      |
| HTPA    | 0.81510       | 0.99968      |
| HRAS    | 0.93352       | 0.99645      |


![Ensemble Model Predictions (Actual vs Predicted)](https://github.com/ArnabMitra0408/AQI-Predictor/blob/main/Plots_And_Metrics/EnsembleModelPredictions(Acutal_Vs_Predicted).png)


![Hybrid Model Predictions (Actual vs Predicted)](https://github.com/ArnabMitra0408/AQI-Predictor/blob/main/Plots_And_Metrics/HybridModelPredictions(Acutal_Vs_Predicted).png)



## Conclusion
- The **Hybrid Model** provides superior AQI predictions by combining feature selection (Random Forest) with temporal learning (LSTM).
- The **Ensemble Model** offers efficiency but struggles with overfitting and lower generalization.
- Future work will explore scaling to larger datasets and additional hybridization strategies.

## Repository Structure
```
├── Notebooks/                # Contains all the notebooks for experimentaions
├── Plots_and_Metrics/              # Contains all the plots and the metrics files
├── data_scripts/                 # Contains all the code to fetch the raw_data
├── data_store/             # Location where all the data (raw and processed) are stored)
├── ensemble/            # Contains the code for the ensemble model
├── hybrid/            # Contains the code for the hybrid model model

```

## Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/ArnabMitra0408/AQI-Hourly-Prediction.git](https://github.com/ArnabMitra0408/AQI-Predictor.git)
   cd AQI-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Gather Data:
   ```bash
   python data_scripts/aqi_data_gathering.py
   python data_scripts/aqi_data_processing.py
   python data_scripts/weather_data_collection.py
   python data_scripts/weather_data_processing.py
   python data_scripts/data_merge.py
   
   ```
4. Train the models:
   ```bash
   python ensemble/ensemble.py
   python hybrid/hybrid.py
   ```
