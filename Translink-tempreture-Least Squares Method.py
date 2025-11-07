import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

CHECKIN_FILE_2023 = r"D:\H\Y1Q1\programming\project\2023 transline.txt"
TEMP_FILE_2023 = r"D:\H\Y1Q1\programming\project\Daily rainfall and tempratur2023.txt"
YEAR = 2023


def load_checkin_weekly(file_path, target_year):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Check-in file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, sep=';', decimal='.', usecols=['Datum', 'Uur', 'Aantal_check_ins'])
    except Exception as e:
        print(f"Error reading check-in file {file_path}: {e}")
        return None
    
    df['Datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Uur'].astype(str), format='%d-%m-%Y %H', errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)

    df_year = df[df.index.year == target_year].copy()
    
    df_daily = df_year['Aantal_check_ins'].resample('D').sum()

    return df_daily.resample('W-MON').sum().dropna()

def load_temperature_weekly(file_path, target_year):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weather file not found: {file_path}")

    try:
        df_temp = pd.read_csv(
            file_path, 
            skiprows=9, 
            sep=r'\s*,\s*|\s+',
            engine='python',
            header=None,
            usecols=[1, 2], 
            names=['DATE', 'TX']
        ).copy()
        
        df_temp = df_temp.dropna(subset=['DATE', 'TX'])
        df_temp['TX'] = pd.to_numeric(df_temp['TX'], errors='coerce')
        df_temp['TX'].replace(-9999, np.nan, inplace=True) 

        df_temp['Datetime'] = pd.to_datetime(df_temp['DATE'].astype(str), format='%Y%m%d', errors='coerce')
        df_temp = df_temp.dropna(subset=['Datetime'])
        df_temp.set_index('Datetime', inplace=True)
      
        df_year = df_temp[df_temp.index.year == target_year].copy()

        daily_max_temp = df_year['TX'] / 10

        return daily_max_temp.resample('W-MON').mean().dropna()

    except Exception as e:
        print(f"Critical error processing weather file {file_path}: {e}")
        return None

def scatter_temp_vs_checkin_weekly(checkin_w, temperature_w, year: int):
    combined_data = pd.concat([checkin_w.rename('Total_Check_ins'), temperature_w.rename('Avg_Temperature')], axis=1).dropna()
    X = combined_data['Avg_Temperature']    
    Y = combined_data['Total_Check_ins']

    r, p_value = stats.pearsonr(X.values, Y.values)
    R_squared = r**2

    coefficients = np.polyfit(X, Y, 1) 
    polynomial = np.poly1d(coefficients)

    X_min, X_max = X.min(), X.max()
    x_fit = np.linspace(X_min, X_max, 100) 

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(X, Y, color='tab:blue', s=60, alpha=0.7, label=f'Weekly Check-ins (Data Points)') 
    
    fit_label = (f'Linear Fit: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}'
                 f' ($R^2$={R_squared:.3f})'
                  f' ($R$={r:.3f})')
                  
    ax.plot(x_fit, polynomial(x_fit), color='tab:red', linestyle='-', linewidth=2, label=fit_label) 

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Weekly Average Maximum Temperature (°C)', fontsize=12)
    ax.set_ylabel('Weekly Total Check-ins', fontsize=12)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'{year} Full Year: Weekly Total Check-ins vs Weekly Avg Max Temperature', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.tight_layout()
    plt.show()

def main_analysis_weekly_full_year():
    
    print(f"Processing weekly data for the full year {YEAR}...")
    
    try:
        checkin_w = load_checkin_weekly(CHECKIN_FILE_2023, YEAR)
        temp_w = load_temperature_weekly(TEMP_FILE_2023, YEAR)
        
        if checkin_w is None or temp_w is None or checkin_w.empty or temp_w.empty:
            print(f"Skipping {YEAR} analysis due to incomplete or missing data for the target year.")
            return

        scatter_temp_vs_checkin_weekly(checkin_w, temp_w, YEAR)
        
        print(f"\nAnalysis complete for {YEAR} full year.")
        print(f"Total weeks plotted: {len(checkin_w.index.intersection(temp_w.index))}") 
        
    except FileNotFoundError as e:
        print(f"File not found error:{e}")
    except Exception as e:
        print(f"Error:{e}")

if __name__ == "__main__":
    main_analysis_weekly_full_year()