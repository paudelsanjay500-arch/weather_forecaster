import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import json

def calculate_hydrological_metrics(y_true, y_pred, threshold):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    nse = 1 - (numerator / denominator)
    
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    far = fp / (fp + tp) if (fp + tp) > 0 else 0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return nse, csi, far, pod, accuracy, precision, recall, f1

def train_phase3_hybrid():
    print("Loading Master Dataset for Phase 3 (CNN-LSTM)...")
    df = pd.read_csv('Master_Rainfall_Dataset_Final.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    features = ['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'SOI', 'DMI', 'N34_A']
    data = df[features].values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    window = 14
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i, 0]) 
        
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Training CNN-LSTM Hybrid Model...")
    model = Sequential([
        # Pattern Extraction (Spatial)
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window, len(features))),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Temporal Sequencing
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        
        # Final Prediction
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    predictions = model.predict(X_test)
    
    y_test_inv = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    predictions_inv = predictions.flatten() * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    threshold = np.percentile(y_test_inv, 90)
    
    nse, csi, far, pod, accuracy, precision, recall, f1 = calculate_hydrological_metrics(y_test_inv, predictions_inv, threshold)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)
    
    results = {
        "Phase": "Phase 3: CNN-LSTM Hybrid",
        "Accuracy": round(float(accuracy * 100), 2),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "NSE": round(float(nse), 4),
        "CSI": round(float(csi), 4),
        "FAR": round(float(far * 100), 2),
        "POD": round(float(pod * 100), 2),
        "Threshold": round(float(threshold), 2)
    }
    
    with open('hybrid_metrics.json', 'w') as f:
        json.dump(results, f)
    
    print("\n--- PHASE 3 PERFORMANCE (CNN-LSTM) ---")
    print(results)

if __name__ == "__main__":
    train_phase3_hybrid()
