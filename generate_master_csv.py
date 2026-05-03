import pandas as pd
import numpy as np
import os

def parse_dmi(file_path):
    # DMI format: Year, Jan, Feb, ..., Dec
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header
            parts = line.split()
            if len(parts) == 13:
                year = int(parts[0])
                for month, val in enumerate(parts[1:], 1):
                    if float(val) > -90: # Handle missing
                        data.append({'Year': year, 'Month': month, 'DMI': float(val)})
    return pd.DataFrame(data)

def parse_soi(file_path):
    # SOI format: Year, Jan, Feb, ..., Dec
    data = []
    start_reading = False
    with open(file_path, 'r') as f:
        for line in f:
            if "YEAR" in line and "JAN" in line:
                start_reading = True
                continue
            if start_reading:
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        year = int(parts[0])
                        for month, val in enumerate(parts[1:13], 1):
                            data.append({'Year': year, 'Month': month, 'SOI': float(val)})
                    except:
                        break
    return pd.DataFrame(data)

def parse_sst(file_path):
    # SST format: YR MON NINO1+2 ANOM ... NINO3.4 ANOM
    df = pd.read_csv(file_path, sep='\s+', skiprows=1, names=['Year', 'Month', 'N12', 'N12_A', 'N3', 'N3_A', 'N4', 'N4_A', 'N34', 'N34_A'])
    return df[['Year', 'Month', 'N34_A']]

def generate_master():
    print("Loading Climate Indices...")
    dmi = parse_dmi('dmi_data.txt')
    soi = parse_soi('soi_data.txt')
    sst = parse_sst('sstoi_indices.txt')
    
    indices = dmi.merge(soi, on=['Year', 'Month']).merge(sst, on=['Year', 'Month'])
    
    print("Loading CAMELS-AUS Hydrometeorology...")
    station_id = "919003A"
    
    # Base path
    base_hydro = "05_hydrometeorology/05_hydrometeorology"
    
    def get_station_data(file_rel_path, col_name):
        df = pd.read_csv(os.path.join(base_hydro, file_rel_path))
        df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
        return df[['Date', station_id]].rename(columns={station_id: col_name})

    precip = get_station_data("01_precipitation_timeseries/precipitation_SILO.csv", "Rainfall")
    tmax = get_station_data("03_Other/SILO/tmax_SILO.csv", "Tmax")
    mslp = get_station_data("03_Other/SILO/mslp_SILO.csv", "MSLP")
    rh = get_station_data("03_Other/SILO/rh_tmax_SILO.csv", "Humidity")
    
    print("Merging Datasets...")
    master = precip.merge(tmax, on='Date').merge(mslp, on='Date').merge(rh, on='Date')
    
    # Filter for 2000-2024
    master = master[(master['Date'].dt.year >= 2000) & (master['Date'].dt.year <= 2024)]
    
    # Merge with monthly indices
    master['Year'] = master['Date'].dt.year
    master['Month'] = master['Date'].dt.month
    master = master.merge(indices, on=['Year', 'Month'], how='left')
    
    print("Calculating Lags and Thresholds...")
    # Lags for Rainfall
    for i in range(1, 8):
        master[f'Rainfall_Lag{i}'] = master['Rainfall'].shift(i)
    
    # High Rainfall Event (90th percentile)
    threshold = master['Rainfall'].quantile(0.90)
    master['High_Rain_Event'] = (master['Rainfall'] > threshold).astype(int)
    
    # Interpolate missing values instead of dropping them
    master = master.interpolate(method='linear')
    master = master.bfill().ffill()
    
    # Add compulsory station identification columns
    master['Station_ID'] = '919003A'
    master['Station_Name'] = 'Mitchell River'
    
    output_file = "Master_Rainfall_Dataset_Final.csv"
    master.to_csv(output_file, index=False)
    print(f"Master Dataset generated successfully: {output_file}")
    print(f"Total Rows: {len(master)}")
    print(f"Variables: {list(master.columns)}")

if __name__ == "__main__":
    generate_master()
