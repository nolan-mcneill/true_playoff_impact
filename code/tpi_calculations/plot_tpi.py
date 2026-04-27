import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, trapezoid

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, 'data', 'tpi_results', 'lebron_tpi_results.csv')
raw_data_path = os.path.join(base_dir, 'data', 'prod_score', 'lebron_data', 'lebron_tpi_metrics_final.csv')
graphs_dir = os.path.join(base_dir, 'data', 'tpi_results', 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# 1. Plot Career Metrics (Normalized)
df_res = pd.read_csv(results_path)
df_res = df_res.dropna() # Drop years without playoffs (2019, 2022)

# Normalize metrics
df_res['Norm_TPI'] = df_res['Total_TPI'] / df_res['Total_TPI'].mean()
df_res['Norm_Prod'] = df_res['Prod_Score'] / df_res['Prod_Score'].mean()
df_res['Norm_Res'] = df_res['Res_Score'] / df_res['Res_Score'].mean()
df_res['Norm_Fatigue'] = df_res['Fatigue_Avg'] / df_res['Fatigue_Avg'].mean()

plt.figure(figsize=(12, 6))
plt.plot(df_res['Year'], df_res['Norm_TPI'], marker='o', label='TPI', linewidth=2)
plt.plot(df_res['Year'], df_res['Norm_Prod'], marker='s', label='Prod Score', linewidth=2)
plt.plot(df_res['Year'], df_res['Norm_Res'], marker='^', label='Res Score', linewidth=2)
plt.plot(df_res['Year'], df_res['Norm_Fatigue'], marker='x', label='Fatigue', linewidth=2)

plt.title('LeBron James: Normalized Career Playoff Metrics', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Normalized Value (1.0 = Career Average)', fontsize=12)
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
plt.xticks(np.arange(df_res['Year'].min(), df_res['Year'].max()+1, 2))
plt.legend()
plt.grid(alpha=0.3)

career_plot_path = os.path.join(graphs_dir, 'tpi_career_normalized.png')
plt.savefig(career_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved career metrics plot to {career_plot_path}")
plt.close()

# 2. Fatigue Validation for 2018
# Engine constants from tpi_v2.py
R_BASE, ETA, LMBDA, BETA, ZETA, PHI = 0.33, 0.15, 0.12, 0.02, 0.05, 1.05

def fatigue_derivative(Af, t, r_eff):
    recovery_rate = r_eff * (1 + ETA * Af) * np.exp(-LMBDA * Af)
    return -Af * recovery_rate

def get_fatigue_curve(row):
    days = [float(x) for x in str(row['Schedule_Days']).split(',')]
    shifts = [float(x) for x in str(row['TZ_Shifts']).split(',')]
    current_af = (pow(row['RS_MPG'] / 36, 1.4) * (row['RS_USG'] / 25)) * (row['Games_Played'] / 82)
    
    time_points = []
    fatigue_vals = []
    
    # Initial state
    time_points.append(0)
    fatigue_vals.append(current_af)

    for i in range(len(days)):
        work_base = pow(row['MPG'] / 36, 1.4) * (row['USG'] / 25)
        impulse = (work_base * PHI) * np.exp(BETA * current_af)
        current_af += impulse
        
        # Log immediately after game impulse
        time_points.append(days[i])
        fatigue_vals.append(current_af)
        
        if i < len(days) - 1:
            t_span = np.linspace(days[i], days[i+1], 50)
            r_eff = R_BASE * pow(1 - ZETA, abs(shifts[i]))
            decay_path = odeint(fatigue_derivative, current_af, t_span, args=(r_eff,))
            
            # Log the recovery curve
            time_points.extend(t_span[1:]) # Skip first to avoid duplicate
            fatigue_vals.extend(decay_path.flatten()[1:])
            
            current_af = decay_path[-1][0]
            
    return time_points, fatigue_vals

df_raw = pd.read_csv(raw_data_path)
row_2018 = df_raw[df_raw['Year'] == 2018].iloc[0]

t_vals, f_vals = get_fatigue_curve(row_2018)

plt.figure(figsize=(12, 6))
plt.plot(t_vals, f_vals, color='firebrick', linewidth=2)
# Highlight game days
game_days = [float(x) for x in str(row_2018['Schedule_Days']).split(',')]
plt.scatter(game_days, [f_vals[t_vals.index(d)] for d in game_days], color='black', zorder=5, label='Game Played')

plt.title('Fatigue Model Validation: 2018 Playoff Run', fontsize=14)
plt.xlabel('Days into Playoffs', fontsize=12)
plt.ylabel('Accumulated Fatigue', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

fatigue_plot_path = os.path.join(graphs_dir, 'fatigue_validation_2018.png')
plt.savefig(fatigue_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved fatigue validation plot to {fatigue_plot_path}")
plt.close()
