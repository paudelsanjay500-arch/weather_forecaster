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
    
    stations = {
        '919003A': 'Mitchell River',
        '912101A': 'Gregory River',
        '925001A': 'Wenlock River'
    }
    
    base_hydro = "05_hydrometeorology/05_hydrometeorology"
    all_station_data = []
    
    print("Loading CAMELS-AUS Hydrometeorology for multiple stations...")
    
    # Load raw files once to save memory
    precip_raw = pd.read_csv(os.path.join(base_hydro, "01_precipitation_timeseries/precipitation_SILO.csv"))
    tmax_raw = pd.read_csv(os.path.join(base_hydro, "03_Other/SILO/tmax_SILO.csv"))
    mslp_raw = pd.read_csv(os.path.join(base_hydro, "03_Other/SILO/mslp_SILO.csv"))
    rh_raw = pd.read_csv(os.path.join(base_hydro, "03_Other/SILO/rh_tmax_SILO.csv"))
    
    def extract_and_format(raw_df, col_name, station_id):
        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(raw_df[['year', 'month', 'day']])
        df[col_name] = raw_df[station_id]
        return df

    for station_id, station_name in stations.items():
        print(f"Processing Station: {station_name} ({station_id})...")
        
        precip = extract_and_format(precip_raw, 'Rainfall', station_id)
        tmax = extract_and_format(tmax_raw, 'Tmax', station_id)
        mslp = extract_and_format(mslp_raw, 'MSLP', station_id)
        rh = extract_and_format(rh_raw, 'Humidity', station_id)
        
        # Merge station data
        station_df = precip.merge(tmax, on='Date', how='left')\
                           .merge(mslp, on='Date', how='left')\
                           .merge(rh, on='Date', how='left')
        
        # Filter dates (MSLP only goes up to 2018, so we keep 2000-2024 and impute the rest)
        station_df = station_df[(station_df['Date'].dt.year >= 2000) & (station_df['Date'].dt.year <= 2024)].copy()
        
        # Merge with climate indices
        station_df['Year'] = station_df['Date'].dt.year
        station_df['Month'] = station_df['Date'].dt.month
        station_df = station_df.merge(indices, on=['Year', 'Month'], how='left')
        
        # Lags
        for i in range(1, 8):
            station_df[f'Rainfall_Lag{i}'] = station_df['Rainfall'].shift(i)
            
        # Threshold calculation for the High_Rain_Event (Target Variable)
        threshold = station_df['Rainfall'].quantile(0.90)
        station_df['High_Rain_Event'] = (station_df['Rainfall'] > threshold).astype(int)
        
        # -------------------------------------------------------------
        # MISSING VALUE IMPUTATION (Professor's Feedback: Mean/Median)
        # -------------------------------------------------------------
        # To preserve seasonality, we fill missing values with the Monthly Mean of that specific feature.
        # This is scientifically much better than the global mean.
        numeric_cols = ['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'DMI', 'SOI', 'N34_A'] + [f'Rainfall_Lag{i}' for i in range(1, 8)]
        
        for col in numeric_cols:
            if station_df[col].isnull().sum() > 0:
                # Calculate monthly mean
                monthly_means = station_df.groupby('Month')[col].transform('mean')
                # Fill missing with monthly mean
                station_df[col] = station_df[col].fillna(monthly_means)
                # If any still missing (e.g. an entire month was missing), fallback to global median
                station_df[col] = station_df[col].fillna(station_df[col].median())
        
        # Add ID columns
        station_df['Station_ID'] = station_id
        station_df['Station_Name'] = station_name
        
        all_station_data.append(station_df)
        
    print("Concatenating all stations into Master Dataset...")
    master = pd.concat(all_station_data, ignore_index=True)
    
    output_file = "AuraSentinel_Research_Final/Master_Rainfall_Dataset_Final.csv"
    master.to_csv(output_file, index=False)
    print(f"Master Dataset generated successfully: {output_file}")
    print(f"Total Rows: {len(master)}")
    print(f"Variables: {list(master.columns)}")

if __name__ == "__main__":
    generate_master()
