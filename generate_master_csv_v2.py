"""
AuraSentinel v2 - Master Dataset Generator
===========================================
THESIS CORRECTION: Using stations NEAREST to Melbourne & Sydney

STATIONS SELECTED (from CAMELS-AUS id_name_metadata.csv):
  1. 229650A  Aldermans Creek at RD 32      (Yarra River, VIC) -> Melbourne ~60km NE
  2. 212209   Nepean River at Maguires Xing  (Hawkesbury,  NSW) -> Sydney    ~60km SW
  3. 405209   Acheron River at Taggerty      (Goulburn,    VIC) -> Melbourne ranges ~100km NE

DATA SOURCES (all locally available in CAMELS-AUS dataset):
  - precipitation_SILO.csv       -> Daily rainfall (mm)
  - tmax_SILO.csv                -> Max temperature (degC)
  - mslp_SILO.csv                -> Mean sea level pressure (hPa)
  - rh_tmax_SILO.csv             -> Relative humidity (%)
  - dmi_data.txt                 -> Indian Ocean Dipole (NOAA)
  - soi_data.txt                 -> Southern Oscillation Index (NOAA)
  - sstoi_indices.txt            -> NINO3.4 SST anomaly (NOAA)

ZERO MISSING VALUES POLICY:
  Step1: Monthly mean imputation (preserves seasonality)
  Step2: Global median fallback
  Step3: Zero fill safety net
"""

import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
STATIONS = {
    '229650A': {
        'name':   'Aldermans Creek, Yarra River (Melbourne, VIC)',
        'lat':    -37.7950,
        'lng':    145.5792,
        'region': 'Melbourne',
    },
    '212209': {
        'name':   'Nepean River, Hawkesbury (Sydney, NSW)',
        'lat':    -34.0372,
        'lng':    150.4736,
        'region': 'Sydney',
    },
    '405209': {
        'name':   'Acheron River, Goulburn (Melbourne Ranges, VIC)',
        'lat':    -37.3178,
        'lng':    145.7139,
        'region': 'Melbourne',
    },
}

DATE_START = '2000-01-01'
DATE_END   = '2018-12-31'

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))           # AuraSentinel_Research_Final/
ROOT_DIR   = os.path.dirname(BASE_DIR)                            # sanjay_antigravity/
HYDRO_BASE = os.path.join(ROOT_DIR, '05_hydrometeorology', '05_hydrometeorology')

PRECIP_FILE = os.path.join(HYDRO_BASE, '01_precipitation_timeseries', 'precipitation_SILO.csv')
TMAX_FILE   = os.path.join(HYDRO_BASE, '03_Other', 'SILO', 'tmax_SILO.csv')
MSLP_FILE   = os.path.join(HYDRO_BASE, '03_Other', 'SILO', 'mslp_SILO.csv')
RH_FILE     = os.path.join(HYDRO_BASE, '03_Other', 'SILO', 'rh_tmax_SILO.csv')

DMI_FILE    = os.path.join(ROOT_DIR, 'dmi_data.txt')
SOI_FILE    = os.path.join(ROOT_DIR, 'soi_data.txt')
SST_FILE    = os.path.join(ROOT_DIR, 'sstoi_indices.txt')

OUTPUT_CSV  = os.path.join(BASE_DIR, 'Master_Rainfall_Dataset_v2.csv')

# ─── CLIMATE INDEX PARSERS ────────────────────────────────────────────────────
def parse_dmi(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 13:
            yr = int(parts[0])
            for mo, val in enumerate(parts[1:], 1):
                v = float(val)
                if v > -90:
                    data.append({'Year': yr, 'Month': mo, 'DMI': v})
    return pd.DataFrame(data)

def parse_soi(path):
    data = []
    reading = False
    with open(path, 'r') as f:
        for line in f:
            if 'YEAR' in line and 'JAN' in line:
                reading = True
                continue
            if reading:
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        yr = int(parts[0])
                        for mo, val in enumerate(parts[1:13], 1):
                            data.append({'Year': yr, 'Month': mo, 'SOI': float(val)})
                    except ValueError:
                        break
    return pd.DataFrame(data)

def parse_sst(path):
    df = pd.read_csv(path, sep=r'\s+', skiprows=1,
                     names=['Year','Month','N12','N12_A','N3','N3_A','N4','N4_A','N34','N34_A'])
    return df[['Year', 'Month', 'N34_A']].copy()

# ─── EXTRACT STATION COLUMN ──────────────────────────────────────────────────
def extract_var(raw_df, col_name, station_id):
    out = pd.DataFrame()
    out['Date'] = pd.to_datetime(raw_df[['year', 'month', 'day']])
    out[col_name] = pd.to_numeric(raw_df[station_id], errors='coerce')
    return out

# ─── IMPUTATION (ZERO-NULL POLICY) ───────────────────────────────────────────
def impute_zero_null(df, numeric_cols):
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            monthly_mean = df.groupby('Month')[col].transform('mean')
            df[col] = df[col].fillna(monthly_mean)
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].fillna(0.0)
    return df

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
def generate_master():
    print("=" * 65)
    print("  AuraSentinel v2 — Fresh Master Dataset Generation")
    print("  Stations: Melbourne (Yarra/Goulburn) + Sydney (Hawkesbury)")
    print("=" * 65)

    # 1. Load climate indices
    print("\n[1/5] Loading NOAA climate indices ...")
    dmi     = parse_dmi(DMI_FILE)
    soi     = parse_soi(SOI_FILE)
    sst     = parse_sst(SST_FILE)
    indices = dmi.merge(soi, on=['Year','Month']).merge(sst, on=['Year','Month'])
    print(f"      Climate records loaded: {len(indices)}")

    # 2. Load all hydro files (once, memory-shared)
    print("\n[2/5] Loading CAMELS-AUS hydrometeorology files (this may take ~30s) ...")
    precip_raw = pd.read_csv(PRECIP_FILE)
    tmax_raw   = pd.read_csv(TMAX_FILE)
    mslp_raw   = pd.read_csv(MSLP_FILE)
    rh_raw     = pd.read_csv(RH_FILE)
    print("      All hydro files loaded.")

    # 3. Process each station
    print("\n[3/5] Processing each station ...")
    all_frames = []

    for sid, meta in STATIONS.items():
        print(f"\n   [{sid}] {meta['name']}")

        # Verify columns exist
        for raw, label in [(precip_raw,'precipitation'),(tmax_raw,'tmax'),
                           (mslp_raw,'mslp'),(rh_raw,'humidity')]:
            if sid not in raw.columns:
                raise ValueError(f"Station {sid} not in {label} file!")

        # Extract variables
        precip = extract_var(precip_raw, 'Rainfall',  sid)
        tmax   = extract_var(tmax_raw,   'Tmax',      sid)
        mslp   = extract_var(mslp_raw,   'MSLP',      sid)
        rh     = extract_var(rh_raw,     'Humidity',  sid)

        # Merge on Date
        sdf = (precip
               .merge(tmax,  on='Date', how='left')
               .merge(mslp,  on='Date', how='left')
               .merge(rh,    on='Date', how='left'))

        # Date filter
        sdf = sdf[(sdf['Date'] >= DATE_START) & (sdf['Date'] <= DATE_END)].copy()
        sdf = sdf.sort_values('Date').reset_index(drop=True)

        # Merge climate indices (monthly)
        sdf['Year']  = sdf['Date'].dt.year
        sdf['Month'] = sdf['Date'].dt.month
        sdf = sdf.merge(indices, on=['Year','Month'], how='left')

        # Feature engineering: Lag features
        for i in range(1, 8):
            sdf[f'Rainfall_Lag{i}'] = sdf['Rainfall'].shift(i)

        # Rolling features (7-day)
        sdf['Rainfall_Roll7'] = sdf['Rainfall'].rolling(window=7, min_periods=1).mean()
        sdf['Tmax_Roll7']     = sdf['Tmax'].rolling(window=7, min_periods=1).mean()

        # Target: High Rain Event (90th percentile of this station's rainfall)
        threshold = sdf['Rainfall'].quantile(0.90)
        sdf['High_Rain_Event'] = (sdf['Rainfall'] > threshold).astype(int)
        sdf['Threshold_90']    = round(threshold, 4)

        # Zero-null imputation
        numeric_cols = ['Rainfall','Tmax','MSLP','Humidity','DMI','SOI','N34_A',
                        'Rainfall_Roll7','Tmax_Roll7'] + [f'Rainfall_Lag{i}' for i in range(1,8)]
        null_before = sdf[numeric_cols].isnull().sum().sum()
        sdf = impute_zero_null(sdf, numeric_cols)
        null_after = sdf[numeric_cols].isnull().sum().sum()

        # Station metadata
        sdf['Station_ID']   = sid
        sdf['Station_Name'] = meta['name']
        sdf['Region']       = meta['region']
        sdf['Lat']          = meta['lat']
        sdf['Lng']          = meta['lng']

        all_frames.append(sdf)

        print(f"      Rows      : {len(sdf)}")
        print(f"      Date range: {sdf['Date'].min().date()} to {sdf['Date'].max().date()}")
        print(f"      Nulls     : {null_before} before -> {null_after} after imputation")
        print(f"      Threshold : {threshold:.2f} mm  |  Extreme events: {sdf['High_Rain_Event'].sum()} ({sdf['High_Rain_Event'].mean()*100:.1f}%)")
        print(f"      Rain mean : {sdf['Rainfall'].mean():.2f} mm  |  Max: {sdf['Rainfall'].max():.2f} mm")

    # 4. Concatenate
    print("\n[4/5] Concatenating all stations ...")
    master = pd.concat(all_frames, ignore_index=True)

    # Final null verification
    total_nulls = master.isnull().sum().sum()
    print(f"      Total rows : {len(master)}")
    print(f"      Total nulls: {total_nulls}  {'--- ZERO NULL GUARANTEE MET ---' if total_nulls == 0 else 'WARNING: nulls present!'}")

    # 5. Save
    print("\n[5/5] Saving master dataset ...")
    master.to_csv(OUTPUT_CSV, index=False)
    print(f"      Saved to: {OUTPUT_CSV}")
    print(f"      File size: {os.path.getsize(OUTPUT_CSV)/1024/1024:.2f} MB")

    print("\n" + "=" * 65)
    print("  DATASET GENERATION COMPLETE")
    print("  Stations: Melbourne + Sydney (CAMELS-AUS genuine data)")
    print("  Missing Values: ZERO")
    print("=" * 65)

    return master


if __name__ == '__main__':
    generate_master()
