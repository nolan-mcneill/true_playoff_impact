import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, 'data', 'tpi_results', 'lebron_tpi_results.csv')
graphs_dir = os.path.join(base_dir, 'data', 'tpi_results', 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Load data
df = pd.read_csv(results_path)
df = df.dropna(subset=['Total_TPI'])

# Set aesthetic style
plt.style.use('dark_background')
accent_color = '#e74c3c' # LeBron Red
secondary_color = '#f1c40f' # Laker Gold
bg_color = '#121212'

# 1. Career TPI Progression (Bar + Line)
fig, ax1 = plt.subplots(figsize=(14, 8), facecolor=bg_color)
ax1.set_facecolor(bg_color)

# Bar chart for Total TPI
bars = ax1.bar(df['Year'], df['Total_TPI'], color=accent_color, alpha=0.7, label='Total TPI')

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{int(height)}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

# Line chart for TPI per Game
ax2 = ax1.twinx()
ax2.plot(df['Year'], df['TPI_per_G'], color=secondary_color, marker='o', linewidth=3, markersize=8, label='TPI per Game')

# Formatting
ax1.set_xlabel('Year', fontsize=13, color='white')
ax1.set_ylabel('Total TPI (Cumulative)', fontsize=13, color=accent_color)
ax2.set_ylabel('TPI per Game (Efficiency/Impact)', fontsize=13, color=secondary_color)
plt.title('LeBron James: True Playoff Impact (TPI) Career Profile', fontsize=18, fontweight='bold', pad=20)

# Grid and Legend
ax1.grid(axis='y', linestyle='--', alpha=0.2)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor=bg_color)

plt.xticks(df['Year'])
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'lebron_career_tpi_profile.png'), dpi=300)
plt.close()

# 2. Fatigue vs Impact (Bubble Chart)
plt.figure(figsize=(10, 8), facecolor=bg_color)
ax = plt.gca()
ax.set_facecolor(bg_color)

# Bubble size based on GP
scatter = plt.scatter(df['Fatigue_Avg'], df['TPI_per_G'], 
            s=df['GP']*20, c=df['Year'], cmap='Reds', 
            alpha=0.8, edgecolors='white', linewidth=1)

# Annotate years
for i, txt in enumerate(df['Year']):
    plt.annotate(txt, (df['Fatigue_Avg'].iloc[i], df['TPI_per_G'].iloc[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.title('The Exertion-Impact Frontier', fontsize=16, fontweight='bold')
plt.xlabel('Average On-Court Fatigue (Af)', fontsize=12)
plt.ylabel('TPI per Game', fontsize=12)
plt.colorbar(scatter, label='Year')
plt.grid(linestyle='--', alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'fatigue_impact_frontier.png'), dpi=300)
plt.close()

# 3. Component Breakdown (Normalized)
df_norm = df.copy()
for col in ['Prod_Score', 'Res_Score', 'Fatigue_Avg']:
    df_norm[col] = df_norm[col] / df_norm[col].mean()

plt.figure(figsize=(14, 6), facecolor=bg_color)
ax = plt.gca()
ax.set_facecolor(bg_color)

plt.plot(df_norm['Year'], df_norm['Prod_Score'], label='Production', marker='o', alpha=0.6)
plt.plot(df_norm['Year'], df_norm['Res_Score'], label='Resistance', marker='s', alpha=0.6)
plt.plot(df_norm['Year'], df_norm['Fatigue_Avg'], label='Fatigue', marker='x', color=accent_color, linewidth=2.5)

plt.axhline(1.0, color='white', linestyle=':', alpha=0.3)
plt.title('Career Metric Drivers (Normalized)', fontsize=15, fontweight='bold')
plt.legend()
plt.xticks(df['Year'])
plt.grid(alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'tpi_component_drivers.png'), dpi=300)
plt.close()

print(f"All career graphs saved to {graphs_dir}")
