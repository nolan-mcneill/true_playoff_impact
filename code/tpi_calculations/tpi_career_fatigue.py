import pandas as pd
import numpy as np
import os
from scipy.integrate import odeint
from datetime import datetime, timedelta

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, 'data', 'tpi_results', 'lebron_tpi_results.csv')
minutes_path = os.path.join(base_dir, 'data', 'fatigue_metric', "lebron's_playoff_minutes_full_career.csv")
rs_dates_path = os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_reg_season_game_dates.csv')
pbp_path = os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_pbp_usg_components.csv')
metrics_path = os.path.join(base_dir, 'data', 'prod_score', 'lebron_data', 'lebron_tpi_metrics_final.csv')

# Playoff Start Dates (from scraped data)
PO_START_DATES = {
    '2006': '2006-04-22', '2007': '2007-04-22', '2008': '2008-04-19', '2009': '2009-04-18', 
    '2010': '2010-04-17', '2011': '2011-04-16', '2012': '2012-04-28', '2013': '2013-04-21', 
    '2014': '2014-04-20', '2015': '2015-04-19', '2016': '2016-04-17', '2017': '2017-04-15', 
    '2018': '2018-04-15', '2020': '2020-08-18', '2021': '2021-05-23', '2023': '2023-04-16', 
    '2024': '2024-04-20', '2025': '2025-04-19', '2026': '2026-04-18'
}

# Fatigue Constants
ALPHA_ON, BETA_ON, GAMMA_BENCH, R_CONST = 0.04, 0.08, 0.008, 0.1

def model_on(f, t, usg):
    intensity = pow(usg / 25.0, 2.0)
    return ALPHA_ON * intensity - BETA_ON * f

def model_off(f, t):
    return -GAMMA_BENCH * f

def simulate_segment(f_start, mode, duration, usg):
    if duration <= 0: return f_start, 0, 0
    t = np.linspace(0, duration, 10)
    if mode == 'ON':
        f_vals = odeint(model_on, f_start, t, args=(usg,))
    else:
        f_vals = odeint(model_off, f_start, t)
    avg_f = np.mean(f_vals)
    return f_vals[-1][0], avg_f * duration, duration

def parse_stretch(s):
    if pd.isna(s) or s == '-': return None
    try:
        mode, times = s.split(': ')
        start, end = map(int, times.split('~'))
        return mode, start, end
    except: return None

def simulate_game(f_start, stretches, usg_by_period, timeouts_by_period):
    f_current = f_start
    stretch_to_period = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6}
    game_fatigue_sum, game_minutes, last_period = 0, 0, 1
    for i, s in enumerate(stretches):
        data = parse_stretch(s)
        if not data: continue
        mode, start, end = data
        period = stretch_to_period.get(i, 4)
        if period > last_period:
            b_dur = 15.0 if period == 3 else (5.0 if period > 4 else 2.5)
            f_current, _, _ = simulate_segment(f_current, 'OFF', b_dur, 0)
            last_period = period
        usg = usg_by_period.get(period, 30.0)
        tos = sorted([t/60.0 for t in timeouts_by_period.get(period, [])])
        current_pos = start
        for to_time in tos:
            if start < to_time < end:
                dur = to_time - current_pos
                f_current, s_sum, s_min = simulate_segment(f_current, mode, dur, usg)
                game_fatigue_sum += s_sum; game_minutes += s_min
                f_current, _, _ = simulate_segment(f_current, 'OFF', 3.0, 0)
                current_pos = to_time
        dur = end - current_pos
        if dur > 0:
            f_current, s_sum, s_min = simulate_segment(f_current, mode, dur, usg)
            game_fatigue_sum += s_sum; game_minutes += s_min
    return f_current, (game_fatigue_sum / game_minutes if game_minutes > 0 else 0)

def parse_rs_dates():
    rs_map = {}
    if not os.path.exists(rs_dates_path): return {}
    with open(rs_dates_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    for line in lines[1:]:
        parts = line.strip().split(',')
        if not parts: continue
        year_str = parts[0]
        start_year = int(year_str.split('-')[0])
        end_year = 2000 + int(year_str.split('-')[1]) if '-' in year_str else start_year
        dates = []
        for val in parts[1:]:
            if not val or '/' not in val: continue
            try:
                mm, dd = map(int, val.split('/'))
                y = start_year if mm >= 10 else end_year
                dates.append(datetime(y, mm, dd))
            except: continue
        rs_map[str(end_year)] = sorted(dates)
    return rs_map

def run_career_simulation():
    df_min = pd.read_csv(minutes_path)
    df_pbp = pd.read_csv(pbp_path)
    df_pbp['TO_List'] = df_pbp['Timeouts'].apply(lambda x: [float(t) for t in str(x).split(',')] if pd.notna(x) and str(x) != "" else [])
    df_met = pd.read_csv(metrics_path)
    rs_game_map = parse_rs_dates()
    
    results = []
    for year in sorted(df_min['Year'].unique()):
        year_str = str(year)
        rs_games = rs_game_map.get(year_str, [])
        po_start_str = PO_START_DATES.get(year_str)
        po_start = datetime.strptime(po_start_str, '%Y-%m-%d') if po_start_str else None
        rs_usg = df_met[df_met['Year']==year]['RS_USG'].iloc[0] if not df_met[df_met['Year']==year].empty else 30.0
        
        f_accum = 0.0
        last_date = None
        # Regular Season
        for g_date in rs_games:
            if last_date:
                gap_days = (g_date - last_date).days
                f_accum *= np.exp(-R_CONST * gap_days)
            # Simulate RS game (generic 38 min load)
            f_accum, _ = simulate_game(f_accum, ["ON: 0~10", "OFF: 0~4", "ON: 4~10", "ON: 0~10", "OFF: 0~4", "ON: 4~10"], {1:rs_usg, 2:rs_usg, 3:rs_usg, 4:rs_usg}, {})
            last_date = g_date
            
        if po_start and last_date:
            gap_days = (po_start - last_date).days
            f_accum *= np.exp(-R_CONST * max(0, gap_days))
            
        year_po = df_min[df_min['Year'] == year]
        df_year_pbp = df_pbp[df_pbp['Year'] == year].copy()
        
        # Calculate Usage
        df_year_pbp['L_Score'] = df_year_pbp['L_FGA'] + 0.44*df_year_pbp['L_FTA'] + df_year_pbp['L_TOV']
        df_year_pbp['T_Score'] = df_year_pbp['T_FGA'] + 0.44*df_year_pbp['T_FTA'] + df_year_pbp['T_TOV']
        df_year_pbp['USG_Calc'] = 100 * (df_year_pbp['L_Score'] / df_year_pbp['T_Score'].replace(0, 1))
        
        pbp_map = {idx: (g.set_index('Period')['USG_Calc'].to_dict(), g.set_index('Period')['TO_List'].to_dict()) 
                   for idx, (gid, g) in enumerate(df_year_pbp.groupby('Game_ID'))}
        
        po_fatigues = []
        for idx, (i, row) in enumerate(year_po.iterrows()):
            if idx > 0: f_accum *= np.exp(-R_CONST * 2.0) # Approx 2 day rest in PO
            usg_map, to_map = pbp_map.get(idx, ({1:35,2:35,3:35,4:35}, {}))
            f_accum, avg_f = simulate_game(f_accum, [row[f'Stretch {k}'] for k in range(1, 8)], usg_map, to_map)
            po_fatigues.append(avg_f)
            
        if po_fatigues:
            results.append({'Year': year, 'Fatigue_Avg': np.mean(po_fatigues)})
            
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_career_fatigue_results.csv'), index=False)
    print("Dynamic RS-to-PO fatigue simulation complete.")

if __name__ == "__main__":
    run_career_simulation()
