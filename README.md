# Australian Extreme Rainfall Forecasting Research (Phase 1)

This repository contains the data, modeling scripts, and an interactive dashboard for the first phase of our research into AI-powered rainfall forecasting for Australian catchments.

## 🚀 Live Dashboard
You can view the interactive research results here:
[https://paudelsanjay500-arch.github.io/weather_forecaster/](https://paudelsanjay500-arch.github.io/weather_forecaster/)

## 📊 Phase 1: LSTM Neural Network
The first phase focuses on a **Long Short-Term Memory (LSTM)** network designed to capture temporal dependencies in multi-source meteorological data (2000-2018).

### Key Performance Metrics:
- **Prediction Accuracy:** 91.20%
- **NSE (Efficiency):** 0.6967
- **RMSE:** 3.5343 mm
- **False Alarm Rate (FAR):** 26.9%

## 📁 Repository Structure
- `index.html`: The interactive Research Dashboard.
- `Master_Rainfall_Dataset.zip`: Compressed master dataset (BoM, NOAA, CAMELS-AUS).
- `train_lstm_phase1.py`: The core Python script used for training the LSTM model.
- `analyze_dataset.py`: Statistical analysis and feature correlation script.
- `prediction_validation.png`: Visual proof of model performance (Actual vs. Predicted).

## 🛠️ Methodology
1. **Data Integration:** Combined surface observations (BoM) with climate indices (ENSO, IOD) and hydrometeorology.
2. **Preprocessing:** Applied Min-Max normalization and created 14-day lookback sequences.
3. **Training:** Deep LSTM architecture with dropout regularization.
4. **Validation:** Evaluated across 5 geographically distinct catchments.

---
© 2026 Australian Rainfall Research Project
