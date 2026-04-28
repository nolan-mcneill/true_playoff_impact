import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, 'data', 'tpi_results', 'lebron_tpi_results.csv')
graphs_dir = os.path.join(base_dir, 'data', 'tpi_results', 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Fatigue Constants (from tpi_career_fatigue.py)
ALPHA_ON, BETA_ON, GAMMA_BENCH, R_CONST = 0.04, 0.08, 0.008, 0.01

def model_on(f, t, usg): 
    intensity = pow(usg / 25.0, 2.0)
    return ALPHA_ON * intensity - BETA_ON * f
def model_off(f, t): return -GAMMA_BENCH * f

# Load data
df = pd.read_csv(results_path)
df = df.dropna(subset=['Total_TPI'])

# Set aesthetic style
plt.style.use('dark_background')
accent_color = '#e74c3c' # LeBron Red
secondary_color = '#f1c40f' # Laker Gold
tpi_color = '#00ccff' # Bright Cyan
bg_color = '#0f0f0f'

# 1. Component Breakdown (Brighter & Thicker)
df_norm = df.copy()
for col in ['Prod_Score', 'Res_Score', 'Fatigue_Avg', 'TPI_per_G']:
    df_norm[col] = df_norm[col] / df_norm[col].mean()

plt.figure(figsize=(14, 7), facecolor=bg_color)
ax = plt.gca()
ax.set_facecolor(bg_color)

# Brighter, thicker component lines
plt.plot(df_norm['Year'], df_norm['Prod_Score'], label='Production', color='#2ecc71', linewidth=2.5, marker='.', alpha=0.9)
plt.plot(df_norm['Year'], df_norm['Res_Score'], label='Resistance', color='#9b59b6', linewidth=2.5, marker='.', alpha=0.9)
plt.plot(df_norm['Year'], df_norm['Fatigue_Avg'], label='Fatigue', color='#e67e22', linewidth=2.5, marker='.', alpha=0.9)

# TPI per Game (Slightly thicker than components)
plt.plot(df_norm['Year'], df_norm['TPI_per_G'], label='TOTAL TPI PER GAME', color=tpi_color, linewidth=4.0, marker='H', markersize=10, zorder=10)

plt.axhline(1.0, color='white', linestyle=':', alpha=0.3)
plt.title('LeBron James: TPI Component Breakdown (Normalized)', fontsize=16, fontweight='bold', pad=15)
plt.legend(loc='upper left', frameon=True, facecolor=bg_color)
plt.xticks(df['Year'])
plt.grid(alpha=0.1)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'tpi_component_drivers.png'), dpi=300)
plt.close()

# 2. Fatigue Explainer: 2018 Finals Deep Dive
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor=bg_color)
ax1.set_facecolor(bg_color)
ax2.set_facecolor(bg_color)

# --- Subplot 1: Game 1 (48 mins ON) ---
usg_g1 = 35.0
f_vals_g1 = [0.45] # Start with some existing fatigue
t_game = np.linspace(0, 53, 530) # 48 mins + OT
for i in range(len(t_game)-1):
    dt = t_game[i+1] - t_game[i]
    df_delta = model_on(f_vals_g1[i], t_game[i], usg_g1)
    f_vals_g1.append(f_vals_g1[i] + df_delta * dt)

ax1.plot(t_game, f_vals_g1, color='#e67e22', linewidth=3)
ax1.fill_between(t_game, f_vals_g1, color='#e67e22', alpha=0.2)
ax1.set_title('2018 Finals Game 1: Intra-Game Micro-Fatigue (48 Mins ON)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Minutes (Game Clock)', alpha=0.7)
ax1.set_ylabel('Fatigue (Af)', alpha=0.7)
ax1.grid(alpha=0.1)
ax1.annotate('Maximum Intensity Load', xy=(48, f_vals_g1[-1]), xytext=(30, 0.65),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5))

# --- Subplot 2: 2018 Finals Series (4 Games) ---
# Simulating 4 games with 2 days rest between
t_series = np.linspace(0, 10, 1000) # 10 days
f_series = [0.45]
game_days = [0, 2.1, 5, 7.1]
for i in range(len(t_series)-1):
    dt = t_series[i+1] - t_series[i]
    is_game = False
    for gd in game_days:
        if gd <= t_series[i] < gd + (48/1440.0): # 48 mins game
            is_game = True
            break
    
    if is_game:
        dfdt = model_on(f_series[i], t_series[i], usg_g1) * 1440.0 # Scale to days
    else:
        dfdt = -R_CONST * f_series[i]
    
    f_series.append(f_series[i] + dfdt * dt)

ax2.plot(t_series, f_series, color='#e74c3c', linewidth=3)
ax2.fill_between(t_series, f_series, color='#e74c3c', alpha=0.2)
ax2.set_title('2018 Finals: Series Macro-Fatigue Accumulation', fontsize=14, fontweight='bold')
ax2.set_xlabel('Days in Series', alpha=0.7)
ax2.set_ylabel('Fatigue (Af)', alpha=0.7)
for gd in game_days: ax2.axvline(gd, color='white', alpha=0.2, linestyle='--')
ax2.grid(alpha=0.1)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(graphs_dir, 'fatigue_engine_mechanics.png'), dpi=300)
plt.close()

# 3. Career Profile
fig3, ax3_1 = plt.subplots(figsize=(14, 8), facecolor=bg_color)
ax3_1.set_facecolor(bg_color)
ax3_1.bar(df['Year'], df['Total_TPI'], color=accent_color, alpha=0.5, label='Total TPI')
ax3_2 = ax3_1.twinx()
ax3_2.plot(df['Year'], df['TPI_per_G'], color=secondary_color, marker='o', linewidth=3, label='TPI per G')
ax3_1.set_title('LeBron James: TPI Career Profile', fontsize=18, fontweight='bold')
ax3_1.set_xticks(df['Year'])
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'lebron_career_tpi_profile.png'), dpi=300)
plt.close()

print(f"Refined visuals saved to {graphs_dir}")
