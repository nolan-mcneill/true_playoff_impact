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

# --- 3. Micro-Fatigue Validation for 2018 ---
minutes_csv_path = os.path.join(base_dir, 'data', 'fatigue_metric', "lebron's_playoff_minutes_2018.csv")

# Constants for micro-fatigue
ALPHA_ON = 0.04   # Increased for visible gain
BETA_ON = 0.08    # Moderate exponential scaling
GAMMA_BENCH = 0.008 # Slower intra-game recovery
LAMBDA_BENCH = 0.01

def on_court_derivative(F, t, usg):
    return ALPHA_ON * (usg / 25.0) * np.exp(BETA_ON * F)

def off_court_derivative(F, t):
    return -GAMMA_BENCH * F * np.exp(-LAMBDA_BENCH * F)

def parse_game_minutes(row_str_series):
    stretches = []
    for val in row_str_series:
        if pd.isna(val) or val == '—' or val == '-': continue
        val = str(val).strip()
        if not val.startswith('ON:') and not val.startswith('OFF:'): continue
        parts = val.split(':')
        state = parts[0].strip()
        times = parts[1].strip().split('~')
        if len(times) != 2: continue
        
        start_str = times[0].replace('*', '').strip()
        end_str = times[1].replace('*', '').strip()
        
        if end_str == '7' and start_str == '36': end_str = '37'
        if end_str == 'SS48': end_str = '48'
        if end_str == '-': continue
        if start_str == '' or end_str == '': continue
        
        try:
            stretches.append((float(start_str), float(end_str), state))
        except ValueError:
            pass
    return stretches

def generate_game_timeline(stretches):
    raw_chunks = []
    for start, end, state in stretches:
        raw_chunks.append({'start': start, 'end': end, 'state': state})
        
    quarter_boundaries = [12.0, 24.0, 36.0, 48.0, 53.0, 58.0, 63.0]
    
    split_chunks = []
    for chunk in raw_chunks:
        t = chunk['start']
        end = chunk['end']
        while t < end:
            next_t = end
            for qb in quarter_boundaries:
                if t < qb < end:
                    next_t = qb
                    break
            split_chunks.append({'start': t, 'end': next_t, 'state': chunk['state']})
            t = next_t
            
    final_timeline = []
    for i, chunk in enumerate(split_chunks):
        dur = chunk['end'] - chunk['start']
        if dur > 0:
            state_type = 'ON' if chunk['state'] == 'ON' else 'OFF_BENCH'
            final_timeline.append((dur, state_type))
            
        if chunk['end'] in quarter_boundaries:
            if i < len(split_chunks) - 1:
                boundary = chunk['end']
                if boundary == 24.0:
                    final_timeline.append((15.0, 'OFF_BREAK'))
                elif boundary in [12.0, 36.0, 48.0, 53.0, 58.0]:
                    final_timeline.append((3.0, 'OFF_BREAK'))
                    
    return final_timeline

def simulate_chunk(F0, duration, state_type, usg):
    if duration <= 0: return [F0], [0]
    t_span = np.linspace(0, duration, max(2, int(duration * 2)))
    
    if state_type == 'ON':
        path = odeint(on_court_derivative, F0, t_span, args=(usg,))
    else:
        path = odeint(off_court_derivative, F0, t_span)
        
    return path.flatten().tolist(), t_span.tolist()

def get_micro_fatigue_curve(row, minutes_df):
    days = [float(x) for x in str(row['Schedule_Days']).split(',')]
    shifts = [float(x) for x in str(row['TZ_Shifts']).split(',')]
    usg = float(row['USG'])
    
    current_af = (pow(row['RS_MPG'] / 36, 1.4) * (row['RS_USG'] / 25))
    
    global_time = [0]
    global_fatigue = [current_af]
    
    game_start_times = []
    game_start_fatigues = []
    game_end_times = []
    game_end_fatigues = []
    
    for i in range(len(days)):
        if i >= len(minutes_df): break
            
        game_row = minutes_df.iloc[i]
        stretches = parse_game_minutes(game_row[['Stretch 1', 'Stretch 2', 'Stretch 3', 'Stretch 4', 'Stretch 5', 'Stretch 6', 'Stretch 7']])
        timeline = generate_game_timeline(stretches)
        
        current_game_time = days[i]
        
        # Log the start of the game
        game_start_times.append(current_game_time)
        game_start_fatigues.append(current_af)
        
        global_time.append(current_game_time)
        global_fatigue.append(current_af)
        
        # Run the internal micro-fatigue math
        for dur, state_type in timeline:
            f_path_seg, t_span_seg = simulate_chunk(current_af, dur, state_type, usg)
            
            # Append intermediate points for rich visualization
            for t_rel, f_val in zip(t_span_seg[1:], f_path_seg[1:]):
                global_time.append(current_game_time + (t_rel / 1440.0))
                global_fatigue.append(f_val)
                
            current_af = f_path_seg[-1]
            current_game_time += (t_span_seg[-1] / 1440.0)
            
        # Log the final fatigue at the end of the game
        game_end_times.append(current_game_time)
        game_end_fatigues.append(current_af)
            
        if i < len(days) - 1:
            rest_duration_days = days[i+1] - current_game_time
            if rest_duration_days > 0:
                t_span_rest = np.linspace(current_game_time, days[i+1], 50)
                r_eff = R_BASE * pow(1 - ZETA, abs(shifts[i]))
                t_span_rel = np.linspace(0, rest_duration_days, 50)
                decay_path = odeint(fatigue_derivative, current_af, t_span_rel, args=(r_eff,))
                
                global_time.extend(t_span_rest[1:])
                global_fatigue.extend(decay_path.flatten()[1:])
                current_af = decay_path[-1][0]
                
    return {
        'global_time': global_time,
        'global_fatigue': global_fatigue,
        'game_starts': (game_start_times, game_start_fatigues),
        'game_ends': (game_end_times, game_end_fatigues)
    }

if os.path.exists(minutes_csv_path):
    minutes_df = pd.read_csv(minutes_csv_path)
    if not minutes_df.empty:
        curve_data = get_micro_fatigue_curve(row_2018, minutes_df)
        t_vals_micro = curve_data['global_time']
        f_vals_micro = curve_data['global_fatigue']
        start_times, start_fatigues = curve_data['game_starts']
        end_times, end_fatigues = curve_data['game_ends']
        
        plt.figure(figsize=(14, 7), facecolor='#f8f9fa')
        ax = plt.gca()
        ax.set_facecolor('#ffffff')
        
        # 1. Plot continuous fatigue (recovery + in-game progression)
        plt.plot(t_vals_micro, f_vals_micro, color='#2c3e50', linewidth=1.5, alpha=0.8, label='Continuous Fatigue Path', zorder=2)
        
        # 4. Impulse arrows (to show direction of change during the game)
        for i in range(len(start_times)):
            dy = end_fatigues[i] - start_fatigues[i]
            # Use arrows to show the jump during the game
            plt.arrow(start_times[i], start_fatigues[i], 0, dy, 
                      width=0.1, head_width=0.5, head_length=0.15, 
                      fc='#e74c3c' if dy > 0 else '#3498db', 
                      ec='#e74c3c' if dy > 0 else '#3498db',
                      alpha=0.9, length_includes_head=True, zorder=5)
            
            # Label specific games the user asked about (1-based index)
            game_num = i + 1
            if game_num in [8, 16, 20, 22]:
                plt.annotate(f'G{game_num}', (start_times[i], end_fatigues[i]), 
                             textcoords="offset points", xytext=(0,10), ha='center',
                             fontsize=10, fontweight='bold', color='#c0392b')

        plt.title('LeBron 2018 Macro-Fatigue: Micro-ODE Integration (Recovery Based)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Days into 2018 Playoffs', fontsize=12, labelpad=10)
        plt.ylabel('Fatigue Units (Af)', fontsize=12, labelpad=10)
        
        # Add legend with nice formatting
        plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        
        # Grid and spines
        plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis to start at 0
        plt.ylim(0, max(f_vals_micro) * 1.2)
        
        micro_plot_path = os.path.join(graphs_dir, 'micro_fatigue_validation_2018.png')
        plt.savefig(micro_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved micro-fatigue validation plot to {micro_plot_path}")
        plt.close()

        # --- 4. Zoomed Single Game Plot ---
        def get_single_game_micro_curve(row, game_row):
            usg = float(row['USG'])
            current_af = (pow(row['RS_MPG'] / 36, 1.4) * (row['RS_USG'] / 25))
            
            stretches = parse_game_minutes(game_row[['Stretch 1', 'Stretch 2', 'Stretch 3', 'Stretch 4', 'Stretch 5', 'Stretch 6', 'Stretch 7']])
            timeline = generate_game_timeline(stretches)
            
            t_minutes = [0]
            fatigue_vals = [current_af]
            current_time_mins = 0
            
            for dur, state_type in timeline:
                f_path, t_span = simulate_chunk(current_af, dur, state_type, usg)
                local_t_span = [current_time_mins + tm for tm in t_span]
                
                t_minutes.extend(local_t_span[1:])
                fatigue_vals.extend(f_path[1:])
                
                current_af = f_path[-1]
                current_time_mins = local_t_span[-1]
                
            return t_minutes, fatigue_vals

        game_1_row = minutes_df.iloc[0]
        t_mins_single, f_vals_single = get_single_game_micro_curve(row_2018, game_1_row)
        
        plt.figure(figsize=(12, 6))
        plt.plot(t_mins_single, f_vals_single, color='darkorange', linewidth=2)
        
        plt.title('In-Game Micro-Fatigue: 2018 Round 1, Game 1 vs Pacers', fontsize=14)
        plt.xlabel('Biological Time (Game Minutes + Break Minutes)', fontsize=12)
        plt.ylabel('Accumulated Fatigue', fontsize=12)
        plt.grid(alpha=0.3)
        
        zoomed_plot_path = os.path.join(graphs_dir, 'micro_fatigue_zoomed_game1.png')
        plt.savefig(zoomed_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved zoomed micro-fatigue plot to {zoomed_plot_path}")
        plt.close()
