import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint

# Constants
R_BASE, ETA, LMBDA, BETA, ZETA, PHI = 0.33, 0.15, 0.12, 0.02, 0.05, 1.05
ALPHA_ON = 0.04
BETA_ON = 0.08
GAMMA_BENCH = 0.008
LAMBDA_BENCH = 0.01

def on_court_derivative(F, t, usg):
    return ALPHA_ON * (usg / 25.0) * np.exp(BETA_ON * F)

def off_court_derivative(F, t):
    return -GAMMA_BENCH * F * np.exp(-LAMBDA_BENCH * F)

def fatigue_derivative(Af, t, r_eff):
    recovery_rate = r_eff * (1 + ETA * Af) * np.exp(-LMBDA * Af)
    return -Af * recovery_rate

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
        try:
            stretches.append((float(start_str), float(end_str), state))
        except: pass
    return stretches

def generate_game_timeline(stretches):
    raw_chunks = [{'start': s, 'end': e, 'state': st} for s, e, st in stretches]
    quarter_boundaries = [12.0, 24.0, 36.0, 48.0, 53.0, 58.0, 63.0]
    split_chunks = []
    for chunk in raw_chunks:
        t, end = chunk['start'], chunk['end']
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
            final_timeline.append((dur, 'ON' if chunk['state'] == 'ON' else 'OFF_BENCH'))
        if chunk['end'] in quarter_boundaries and i < len(split_chunks) - 1:
            boundary = chunk['end']
            if boundary == 24.0: final_timeline.append((15.0, 'OFF_BREAK'))
            elif boundary in [12.0, 36.0, 48.0, 53.0, 58.0]: final_timeline.append((3.0, 'OFF_BREAK'))
    return final_timeline

def simulate_chunk(F0, duration, state_type, usg):
    t_span = np.linspace(0, duration, 2)
    if state_type == 'ON':
        path = odeint(on_court_derivative, F0, t_span, args=(usg,))
    else:
        path = odeint(off_court_derivative, F0, t_span)
    return path[-1][0]

def debug_2018():
    base_dir = r"c:\Users\mcnei\CS\Personal_coding\true_playoff_impact"
    raw_data_path = os.path.join(base_dir, 'data', 'prod_score', 'lebron_data', 'lebron_tpi_metrics_final.csv')
    minutes_csv_path = os.path.join(base_dir, 'data', 'fatigue_metric', "lebron's_playoff_minutes_2018.csv")
    
    df_raw = pd.read_csv(raw_data_path)
    row_2018 = df_raw[df_raw['Year'] == 2018].iloc[0]
    minutes_df = pd.read_csv(minutes_csv_path)
    
    days = [float(x) for x in str(row_2018['Schedule_Days']).split(',')]
    shifts = [float(x) for x in str(row_2018['TZ_Shifts']).split(',')]
    usg = float(row_2018['USG'])
    current_af = (pow(row_2018['RS_MPG'] / 36, 1.4) * (row_2018['RS_USG'] / 25))
    
    print(f"{'Game':<5} | {'Start F':<8} | {'End F':<8} | {'Impulse':<8} | {'Net Change':<10}")
    print("-" * 50)
    
    prev_end_af = current_af
    for i in range(len(days)):
        game_row = minutes_df.iloc[i]
        stretches = parse_game_minutes(game_row[['Stretch 1', 'Stretch 2', 'Stretch 3', 'Stretch 4', 'Stretch 5', 'Stretch 6', 'Stretch 7']])
        timeline = generate_game_timeline(stretches)
        
        game_start_af = current_af
        for dur, state_type in timeline:
            current_af = simulate_chunk(current_af, dur, state_type, usg)
        
        game_end_af = current_af
        impulse = game_end_af - game_start_af
        net_change = game_end_af - prev_end_af
        
        print(f"{i+1:<5} | {game_start_af:<8.3f} | {game_end_af:<8.3f} | {impulse:<8.3f} | {net_change:<10.3f}")
        
        if i < len(days) - 1:
            rest_duration = days[i+1] - (days[i] + 70/1440.0) # approx 70 mins game
            if rest_duration > 0:
                r_eff = R_BASE * pow(1 - ZETA, abs(shifts[i]))
                t_span = np.linspace(0, rest_duration, 2)
                decay_path = odeint(fatigue_derivative, current_af, t_span, args=(r_eff,))
                current_af = decay_path[-1][0]
        prev_end_af = game_end_af

debug_2018()
