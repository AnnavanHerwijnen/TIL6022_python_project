# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CHECKIN_PATH = r"D:\H\Y1Q1\programming\project\2024 transline.txt"
TEMP_PATH    = r"D:\H\Y1Q1\programming\project\2024 temperature data.txt"
YEAR         = 2024

def load_checkin_weekly(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    df = pd.read_csv(file, sep=';', decimal='.', usecols=['Datum', 'Uur', 'Aantal_check_ins'])
    df['Datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Uur'].astype(str), format='%d-%m-%Y %H')
    daily = df.set_index('Datetime')['Aantal_check_ins'].resample('D').sum()
    return daily.resample('W').mean()   

def load_temp_weekly(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    df = pd.read_csv(file, skiprows=32, sep=r'\s*,\s*|\s+', engine='python',
                     header=None, usecols=[1, 2, 5], names=['DATE', 'HH', 'TX'])
    
    df = df.dropna(subset=['DATE', 'HH', 'TX'])

    df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + df['HH'].astype(str).str.zfill(2),
                                     format='%Y%m%d%H', errors='coerce')
   
    daily_max = (df.set_index('Datetime')['TX'].replace(-9999, np.nan) / 10).resample('D').max()
    return daily_max.resample('W').mean()   

def scatter_temp_vs_checkin(checkin, temp, year: int):
    common = checkin.index.intersection(temp.index)
    checkin_aligned, temp_aligned = checkin.loc[common], temp.loc[common]

    r, p_value = stats.pearsonr(temp_aligned.dropna(), checkin_aligned.dropna())
    R_squared = r**2
  
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(temp_aligned, checkin_aligned, 
               color='tab:blue', 
               s=60, 
               alpha=0.7, 
               label='Avg Weekend Check-ins (Data Points)') 

    coefficients = np.polyfit(temp_aligned.dropna(), checkin_aligned.dropna(), 1) 
    polynomial = np.poly1d(coefficients)
    
    temp_min = temp_aligned.min()
    temp_max = temp_aligned.max()
    x_fit = np.linspace(temp_min, temp_max, 100) 
    
    fit_label = (f'Linear Fit: y={coefficients[0]:.2f}x + {coefficients[1]:.2f}'
                 f' ($R^2$={R_squared:.3f})')
                 
    ax.plot(x_fit, polynomial(x_fit), 
            color='tab:red', 
            linestyle='-', 
            linewidth=2, 
            label=fit_label) 

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Avg Weekend Max Temperature (°C)')
    ax.set_ylabel('Avg Weekend Check-ins')
    ax.legend(loc='best')
    ax.set_title(f'{year} Avg Weekly Check-ins vs Avg Weekly Max Temperature (with Linear Fit)')
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        checkin_w = load_checkin_weekly(CHECKIN_PATH)
        temp_w    = load_temp_weekly(TEMP_PATH)
      
        scatter_temp_vs_checkin(checkin_w, temp_w, YEAR)

    except FileNotFoundError as e:
        print(f"File not found:{e}")
    except Exception as e:
        print(f"Erro{e}")