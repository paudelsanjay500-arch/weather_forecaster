import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def perform_analysis():
    print("=" * 60)
    print("  AuraSentinel v2 - Exploratory Data Analysis")
    print("  Generating strict evaluation metrics and charts")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv('Master_Rainfall_Dataset_v2.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create EDA output directory
    os.makedirs('EDA_Results_v2', exist_ok=True)
    
    print("\n[1/4] Calculating Statistical Overview...")
    print(f"Total Records: {len(df)}")
    print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # 1. Correlation Analysis
    print("\n[2/4] Generating Correlation Matrix...")
    features = ['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'SOI', 'DMI', 'N34_A', 'Rainfall_Roll7']
    corr_matrix = df[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix (Melbourne & Sydney Stations)', fontsize=14)
    plt.tight_layout()
    plt.savefig('EDA_Results_v2/correlation_heatmap.png', dpi=300)
    plt.close()

    # 2. Rainfall Distribution
    print("[3/4] Generating Rainfall Distribution...")
    plt.figure(figsize=(12, 6))
    
    # We filter out 0 rainfall days to see the distribution of actual rain events better
    rain_days = df[df['Rainfall'] > 0]
    
    sns.histplot(rain_days['Rainfall'], bins=50, kde=True, color='teal')
    
    # Draw threshold lines for each station
    colors = ['red', 'orange', 'purple']
    stations = df['Station_Name'].unique()
    
    for i, station in enumerate(stations):
        st_data = df[df['Station_Name'] == station]
        thresh = st_data['Threshold_90'].iloc[0]
        plt.axvline(thresh, color=colors[i], linestyle='--', label=f'{station} (90th: {thresh}mm)')
        
    plt.title('Daily Rainfall Distribution & Extreme Event Thresholds (>0mm days)', fontsize=14)
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('EDA_Results_v2/rainfall_distribution.png', dpi=300)
    plt.close()

    # 3. Monthly Trends
    print("[4/4] Generating Monthly Rainfall Trends...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='Rainfall', data=df, showfliers=False, palette='viridis')
    plt.title('Monthly Rainfall Distribution (Seasonality)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm) - Outliers Hidden')
    plt.tight_layout()
    plt.savefig('EDA_Results_v2/monthly_seasonality.png', dpi=300)
    plt.close()

    print("\nAll EDA charts saved in EDA_Results_v2/ directory.")
    print("=" * 60)

if __name__ == "__main__":
    perform_analysis()
