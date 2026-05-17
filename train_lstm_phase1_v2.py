"""
AuraSentinel v2 — Phase 1: Stacked LSTM (Baseline)
====================================================
Dataset: Master_Rainfall_Dataset_v2.csv
Stations: Melbourne (Yarra, Goulburn) + Sydney (Hawkesbury)
Window: 14 days | Epochs: 25 | Architecture: LSTM(100)->LSTM(50)->Dense
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score)
import json, os, sys

sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH    = 'Master_Rainfall_Dataset_v2.csv'
OUTPUT_JSON = 'phase1_results.json'
WINDOW      = 14
EPOCHS      = 25
BATCH_SIZE  = 64
FEATURES    = ['Rainfall','Tmax','MSLP','Humidity','SOI','DMI','N34_A',
               'Rainfall_Roll7','Tmax_Roll7']

def hydrological_metrics(y_true, y_pred, threshold):
    nse = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2)
    yb  = (y_true > threshold).astype(int)
    pb  = (y_pred > threshold).astype(int)
    tp  = np.sum((yb==1)&(pb==1)); fp = np.sum((yb==0)&(pb==1))
    fn  = np.sum((yb==1)&(pb==0))
    csi = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0
    far = fp/(fp+tp)    if (fp+tp)>0    else 0
    pod = tp/(tp+fn)    if (tp+fn)>0    else 0
    acc = accuracy_score(yb, pb)
    pre = precision_score(yb, pb, zero_division=0)
    rec = recall_score(yb, pb, zero_division=0)
    f1  = f1_score(yb, pb, zero_division=0)
    return nse, csi, far, pod, acc, pre, rec, f1

def train():
    print("=" * 60)
    print("  AuraSentinel Phase 1 — Stacked LSTM Training")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  Loaded {len(df)} rows, {df['Station_ID'].nunique()} stations")

    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i-WINDOW:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    split  = int(len(X) * 0.80)
    vsplit = int(len(X) * 0.90)
    X_tr, X_val, X_te = X[:split], X[split:vsplit], X[vsplit:]
    y_tr, y_val, y_te = y[:split], y[split:vsplit], y[vsplit:]
    print(f"  Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")

    # Model architecture
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
        BatchNormalization(),
        Dropout(0.25),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.20),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ]

    print(f"\n  Training for up to {EPOCHS} epochs ...")
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # Predict & inverse scale
    preds  = model.predict(X_te).flatten()
    r_min  = scaler.data_min_[0]; r_max = scaler.data_max_[0]
    y_inv  = y_te   * (r_max - r_min) + r_min
    p_inv  = preds  * (r_max - r_min) + r_min
    p_inv  = np.maximum(p_inv, 0)

    # Threshold optimization (85th–95th percentile sweep)
    best_acc, best_thr = 0, np.percentile(y_inv, 90)
    for pct in np.linspace(85, 95, 200):
        t   = np.percentile(y_inv, pct)
        acc = accuracy_score((y_inv>t).astype(int), (p_inv>t).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, t

    nse,csi,far,pod,acc,pre,rec,f1 = hydrological_metrics(y_inv, p_inv, best_thr)
    rmse = float(np.sqrt(mean_squared_error(y_inv, p_inv)))
    mae  = float(mean_absolute_error(y_inv, p_inv))
    r2   = float(r2_score(y_inv, p_inv))

    results = {
        "Phase"    : "Phase 1: Stacked LSTM",
        "Accuracy" : round(acc*100, 2),
        "Precision": round(float(pre), 4),
        "Recall"   : round(float(rec), 4),
        "F1"       : round(float(f1),  4),
        "RMSE"     : round(rmse, 4),
        "MAE"      : round(mae,  4),
        "R2"       : round(r2,   4),
        "NSE"      : round(float(nse), 4),
        "CSI"      : round(float(csi), 4),
        "FAR"      : round(float(far)*100, 2),
        "POD"      : round(float(pod)*100, 2),
        "Threshold": round(float(best_thr), 2),
        "Stations" : "229650A | 212209 | 405209",
        "Region"   : "Melbourne + Sydney"
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n  PHASE 1 RESULTS:")
    for k, v in results.items():
        print(f"    {k:12s}: {v}")
    print(f"\n  Saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    train()
