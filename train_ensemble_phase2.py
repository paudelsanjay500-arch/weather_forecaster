import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import json

def calculate_hydrological_metrics(y_true, y_pred, threshold):
    # NSE (Nash-Sutcliffe Efficiency)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    nse = 1 - (numerator / denominator)
    
    # Event Detection Metrics
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

def train_phase2_ensemble():
    print("Loading Master Dataset for Phase 2 (Ensemble)...")
    df = pd.read_csv('Master_Rainfall_Dataset_Final.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    features = ['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'SOI', 'DMI', 'N34_A']
    data = df[features].values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create Sequences (Window = 14 days)
    X, y = [], []
    window = 14
    for i in range(window, len(scaled_data)):
        # Flatten the 14x7 window into a 1D array of 98 features for tree-based models
        X.append(scaled_data[i-window:i].flatten())
        y.append(scaled_data[i, 0]) 
        
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Training XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    predictions = xgb.predict(X_test)
    
    # Inverse Scale
    y_test_inv = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    predictions_inv = predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    threshold = np.percentile(y_test_inv, 90)
    
    nse, csi, far, pod, accuracy, precision, recall, f1 = calculate_hydrological_metrics(y_test_inv, predictions_inv, threshold)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)
    
    results = {
        "Phase": "Phase 2: XGBoost Ensemble",
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
    
    with open('ensemble_metrics.json', 'w') as f:
        json.dump(results, f)
    
    print("\n--- PHASE 2 PERFORMANCE (XGBoost) ---")
    print(results)

if __name__ == "__main__":
    train_phase2_ensemble()
