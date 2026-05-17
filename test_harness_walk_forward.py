import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                     Dropout, BatchNormalization, Bidirectional,
                                     GlobalAveragePooling1D, Multiply, Activation,
                                     Flatten, Reshape)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import sys

# Silence TF warnings for clean output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CSV_PATH = 'Master_Rainfall_Dataset_v2.csv'
WINDOW = 14
EPOCHS = 15
BATCH_SIZE = 64
N_SPLITS = 5
FEATURES = ['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'SOI', 'DMI', 'N34_A', 'Rainfall_Roll7', 'Tmax_Roll7']

# 1. Model Definitions
def build_lstm(window, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_hybrid(window, n_features):
    inp = Input(shape=(window, n_features))
    c = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inp)
    c = BatchNormalization()(c)
    c = Conv1D(128, kernel_size=2, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.2)(c)

    b = Bidirectional(LSTM(64, return_sequences=True))(c)
    b = BatchNormalization()(b)

    attn = Dense(1, activation='tanh')(b)
    attn = Flatten()(attn)
    attn = Activation('softmax')(attn)
    attn = Reshape((window, 1))(attn)
    ctx  = Multiply()([b, attn])
    ctx  = GlobalAveragePooling1D()(ctx)

    x = Dense(32, activation='relu')(ctx)
    x = Dropout(0.15)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

# 2. Test Harness Logic
def run_test_harness():
    print("=" * 60)
    print("  WALK-FORWARD VALIDATION TEST HARNESS & T-TEST")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i-WINDOW:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    lstm_rmses = []
    hybrid_rmses = []
    
    lstm_all_abs_errors = []
    hybrid_all_abs_errors = []

    split_num = 1
    for train_index, test_index in tscv.split(X):
        print(f"\n--- Processing Split {split_num}/{N_SPLITS} ---")
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        
        print(f"Train Size: {len(X_tr)} | Test Size: {len(X_te)}")

        # Train Baseline LSTM
        print("Training Baseline LSTM...")
        lstm_model = build_lstm(WINDOW, len(FEATURES))
        lstm_model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        
        # Train Hybrid
        print("Training Hybrid CNN-BiLSTM-Attention...")
        hybrid_model = build_hybrid(WINDOW, len(FEATURES))
        hybrid_model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # Inverse Transform
        r_min = scaler.data_min_[0]
        r_max = scaler.data_max_[0]
        y_inv = y_te * (r_max - r_min) + r_min

        lstm_preds = np.maximum(lstm_model.predict(X_te, verbose=0).flatten() * (r_max - r_min) + r_min, 0)
        hybrid_preds = np.maximum(hybrid_model.predict(X_te, verbose=0).flatten() * (r_max - r_min) + r_min, 0)

        # Calculate Split Metrics
        lstm_rmse = np.sqrt(mean_squared_error(y_inv, lstm_preds))
        hybrid_rmse = np.sqrt(mean_squared_error(y_inv, hybrid_preds))
        
        lstm_rmses.append(lstm_rmse)
        hybrid_rmses.append(hybrid_rmse)

        print(f"Result Split {split_num} -> LSTM RMSE: {lstm_rmse:.4f} | Hybrid RMSE: {hybrid_rmse:.4f}")

        # Store absolute errors for paired T-Test
        lstm_all_abs_errors.extend(np.abs(y_inv - lstm_preds))
        hybrid_all_abs_errors.extend(np.abs(y_inv - hybrid_preds))
        
        split_num += 1

    # 3. Statistical T-Test
    print("\n" + "=" * 60)
    print("  STATISTICAL SIGNIFICANCE (PAIRED T-TEST)")
    print("=" * 60)
    
    # Calculate Paired T-Test on absolute errors
    t_stat, p_value = stats.ttest_rel(lstm_all_abs_errors, hybrid_all_abs_errors)
    
    avg_lstm_rmse = np.mean(lstm_rmses)
    avg_hybrid_rmse = np.mean(hybrid_rmses)

    report = [
        "WALK-FORWARD VALIDATION & STATISTICAL SIGNIFICANCE REPORT",
        "=========================================================",
        f"Validation Strategy: {N_SPLITS}-Fold TimeSeriesSplit (Walk-Forward)",
        "",
        "AVERAGE PERFORMANCE ACROSS ALL SPLITS:",
        f"- Baseline LSTM Average RMSE: {avg_lstm_rmse:.4f} mm",
        f"- Proposed Hybrid Average RMSE: {avg_hybrid_rmse:.4f} mm",
        "",
        "PAIRED T-TEST RESULTS (on Absolute Errors):",
        f"- T-Statistic: {t_stat:.4f}",
        f"- P-Value: {p_value:.4e}",
        ""
    ]

    if p_value < 0.05:
        report.append("CONCLUSION: The p-value is < 0.05. The performance improvement of the Proposed Hybrid model over the Baseline LSTM is STATISTICALLY SIGNIFICANT. It is not due to random variation.")
    else:
        report.append("CONCLUSION: The p-value is >= 0.05. The performance difference is NOT statistically significant.")

    report_text = "\n".join(report)
    print(report_text)

    with open('statistical_significance_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\nSaved report to 'statistical_significance_report.txt'")

if __name__ == '__main__':
    run_test_harness()
