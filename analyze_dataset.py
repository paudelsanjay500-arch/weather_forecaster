import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_analysis():
    # Load dataset
    df = pd.read_csv('Master_Rainfall_Dataset_Final.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("--- DATASET OVERVIEW ---")
    print(f"Total Records: {len(df)}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print("\nMissing Values:\n", df.isnull().sum())
    
    print("\n--- STATISTICAL SUMMARY ---")
    print(df[['Rainfall', 'Tmax', 'MSLP', 'Humidity', 'SOI', 'DMI', 'N34_A']].describe())
    
    # 1. Correlation Analysis
    print("\n--- CORRELATION ANALYSIS (Target: Rainfall) ---")
    corr = df.corr()['Rainfall'].sort_values(ascending=False)
    print(corr)
    
    # 2. Extreme Events Analysis
    extreme_count = df['High_Rain_Event'].sum()
    print(f"\nTotal Extreme Rainfall Events (90th percentile): {extreme_count}")
    print(f"Percentage of Extreme Events: {(extreme_count/len(df))*100:.2f}%")
    
    # 3. Monthly Trends
    monthly_avg = df.groupby('Month')['Rainfall'].mean()
    print("\n--- MONTHLY RAINFALL TRENDS ---")
    print(monthly_avg)
    
    # Generate Visualizations (Saved as files)
    print("\nGenerating Visualizations...")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Rainfall Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rainfall'], bins=50, kde=True, color='blue')
    plt.axvline(df['Rainfall'].quantile(0.90), color='red', linestyle='--', label='90th Percentile')
    plt.title('Rainfall Distribution & Extreme Threshold')
    plt.legend()
    plt.savefig('rainfall_distribution.png')
    plt.close()
    
    # Time Series of Extreme Events (Last 5 years)
    recent_df = df[df['Date'].dt.year >= 2010]
    plt.figure(figsize=(15, 6))
    plt.plot(recent_df['Date'], recent_df['Rainfall'], color='gray', alpha=0.5, label='Daily Rainfall')
    extreme_days = recent_df[recent_df['High_Rain_Event'] == 1]
    plt.scatter(extreme_days['Date'], extreme_days['Rainfall'], color='red', s=20, label='Extreme Events')
    plt.title('Rainfall Time Series & Extreme Events (Post-2010)')
    plt.legend()
    plt.savefig('rainfall_timeseries.png')
    plt.close()

    print("Analysis complete. Visualizations saved as .png files.")

if __name__ == "__main__":
    perform_analysis()
