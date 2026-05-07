import pandas as pd
import numpy as np
import os
from scipy.integrate import odeint
from datetime import datetime, timedelta

# Path configuration
# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_dir, 'data', 'results', 'lebron_tpi_results.csv')
minutes_path = os.path.join(base_dir, 'data', 'raw', "lebron's_playoff_minutes_full_career.csv")
rs_dates_path = os.path.join(base_dir, 'data', 'raw', 'lebron_reg_season_game_dates.csv')
pbp_path = os.path.join(base_dir, 'data', 'processed', 'lebron_pbp_usg_components.csv')
metrics_path = os.path.join(base_dir, 'data', 'raw', 'lebron_tpi_metrics_final.csv')

PO_START_DATES = {
    '2006': '2006-04-22', '2007': '2007-04-22', '2008': '2008-04-19', '2009': '2009-04-18', 
    '2010': '2010-04-17', '2011': '2011-04-16', '2012': '2012-04-28', '2013': '2013-04-21', 
    '2014': '2014-04-20', '2015': '2015-04-19', '2016': '2016-04-17', '2017': '2017-04-15', 
    '2018': '2018-04-15', '2020': '2020-08-18', '2021': '2021-05-23', '2023': '2023-04-16', 
    '2024': '2024-04-20', '2025': '2025-04-19', '2026': '2026-04-18'
}

from bio_model import bio_model_ode, bio_model_off, bio_model_rest, get_P

def simulate_segment(state, mode, duration, p_stats=None):
    if duration <= 0: return state, state, 0
    
    if mode == 'REST':
        t = np.linspace(0, duration, 10)
    else:
        t = np.linspace(0, duration, max(2, int(duration * 2)))
        
    if mode == 'ON':
        period_min = p_stats.get('Period_Min', 12.0)
        ast_pm = p_stats.get('L_AST', 0) / period_min
        Contact_Rate = (p_stats.get('L_FD', 0) + p_stats.get('L_PF', 0) + p_stats.get('L_FTA', 0)) / period_min
        Explosion_Rate = (p_stats.get('L_TRB', 0) + p_stats.get('L_STL', 0) + p_stats.get('L_BLK', 0)) / period_min
        pace = p_stats.get('Pace', 95.0)
        
        I_mov = (pace / 100.0) + (ast_pm * 0.2)
        I_col = (Contact_Rate * 0.7) + (Explosion_Rate * 0.3)
        I_total = (I_mov * 0.5) + (I_col * 0.5)
        
        f_vals = odeint(bio_model_ode, state, t, args=(I_total, I_col))
    elif mode == 'OFF':
        f_vals = odeint(bio_model_off, state, t)
    else: # REST (days in minutes)
        f_vals = odeint(bio_model_rest, state, t)
        
    f_vals[:, 0:5] = np.clip(f_vals[:, 0:5], 0.0, 1.0)
    f_vals[:, 5] = np.maximum(f_vals[:, 5], 0.0)
    avg_state = np.mean(f_vals, axis=0)
    return f_vals[-1].tolist(), avg_state.tolist(), duration


def parse_stretch(s):
    if pd.isna(s) or s == '-': return None
    try:
        mode, times = s.split(': ')
        start, end = map(int, times.split('~'))
        return mode, start, end
    except: return None

def simulate_game(state, stretches, pbp_map, pace):
    stretch_to_period = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6}
    game_P_sum = 0.0
    game_minutes = 0.0
    game_states = []
    
    last_period = 1
    for i, s in enumerate(stretches):
        data = parse_stretch(s)
        if not data: continue
        mode, start, end = data
        period = stretch_to_period.get(i, 4)
        
        if period > last_period:
            b_dur = 15.0 if period == 3 else (5.0 if period > 4 else 2.5)
            state, _, _ = simulate_segment(state, 'OFF', b_dur)
            last_period = period
            
        p_stats = pbp_map.get(period, {})
        p_stats['Period_Min'] = 5.0 if period > 4 else 12.0
        p_stats['Pace'] = pace
        tos = sorted([t/60.0 for t in p_stats.get('Timeouts', [])])
        
        curr_pos = start
        for to_time in tos:
            if start < to_time < end:
                dur = to_time - curr_pos
                state, avg_s, s_min = simulate_segment(state, mode, dur, p_stats)
                if mode == 'ON':
                    game_P_sum += get_P(avg_s) * s_min
                    game_minutes += s_min
                    game_states.append(avg_s)
                state, _, _ = simulate_segment(state, 'OFF', 3.0)
                curr_pos = to_time
        dur = end - curr_pos
        if dur > 0:
            state, avg_s, s_min = simulate_segment(state, mode, dur, p_stats)
            if mode == 'ON':
                game_P_sum += get_P(avg_s) * s_min
                game_minutes += s_min
                game_states.append(avg_s)
                
    avg_P = game_P_sum / game_minutes if game_minutes > 0 else get_P(state)
    avg_game_state = np.mean(game_states, axis=0).tolist() if game_states else state
    return state, avg_P, avg_game_state

def parse_rs_dates():
    rs_map = {}
    if not os.path.exists(rs_dates_path): return {}
    with open(rs_dates_path, 'r') as f:
        lines = f.readlines()
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
    df_pbp['Timeouts'] = df_pbp['Timeouts'].apply(lambda x: [float(t) for t in str(x).split(',')] if pd.notna(x) and str(x) != "" else [])
    df_met = pd.read_csv(metrics_path)
    rs_game_map = parse_rs_dates()
    
    results = []
    bio_averages = []
    
    for year in sorted(df_min['Year'].unique()):
        year_str = str(year)
        rs_games = rs_game_map.get(year_str, [])
        po_start_str = PO_START_DATES.get(year_str)
        po_start = datetime.strptime(po_start_str, '%Y-%m-%d') if po_start_str else None
        rs_pace = df_met[df_met['Year']==year]['RS_Pace'].iloc[0] if not df_met[df_met['Year']==year].empty else 95.0
        po_pace = df_met[df_met['Year']==year]['PO_Pace'].iloc[0] if not df_met[df_met['Year']==year].empty else 90.0
        
        # Initial State: [PCr, Gly, Lac, M, CNS, Phi]
        state = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        last_date = None
        
        # Simulate Regular Season
        for g_date in rs_games:
            if last_date:
                gap_days = (g_date - last_date).days
                if gap_days > 0:
                    state, _, _ = simulate_segment(state, 'REST', gap_days * 1440.0)
            
            # Generic 38 min RS game (generic stats)
            rs_p_stats = {'Pace': rs_pace, 'L_AST': 2.0, 'L_FD': 2.0, 'L_PF': 0.5, 'L_FTA': 2.0, 'L_TRB': 2.0, 'L_STL': 0.5, 'L_BLK': 0.2}
            rs_pbp_map = {p: rs_p_stats for p in range(1, 5)}
            state, _, _ = simulate_game(state, ["ON: 0~10", "OFF: 0~4", "ON: 4~10", "ON: 0~10", "OFF: 0~4", "ON: 4~10"], rs_pbp_map, rs_pace)
            last_date = g_date
            
        if po_start and last_date:
            gap_days = max(0, (po_start - last_date).days)
            if gap_days > 0:
                state, _, _ = simulate_segment(state, 'REST', gap_days * 1440.0)
                
        year_po = df_min[df_min['Year'] == year]
        df_year_pbp = df_pbp[df_pbp['Year'] == year].copy()
        
        pbp_map = {}
        for gid, g in df_year_pbp.groupby('Game_ID'):
            g_dict = {}
            for _, row in g.iterrows():
                g_dict[row['Period']] = row.to_dict()
            pbp_map[gid] = g_dict
            
        po_Ps = []
        po_states = []
        for idx, (i, row) in enumerate(year_po.iterrows()):
            if idx > 0: 
                state, _, _ = simulate_segment(state, 'REST', 2.0 * 1440.0) # Approx 2 day rest
            gid = list(pbp_map.keys())[idx] if idx < len(pbp_map) else None
            game_pbp = pbp_map.get(gid, {})
            state, avg_P, avg_state = simulate_game(state, [row[f'Stretch {k}'] for k in range(1, 8)], game_pbp, po_pace)
            po_Ps.append(avg_P)
            po_states.append(avg_state)
            
        if po_Ps:
            avg_P_season = np.mean(po_Ps)
            results.append({'Year': year, 'Fatigue_Avg': avg_P_season})
            avg_season_states = np.mean(po_states, axis=0)
            bio_averages.append({
                'Year': year, 'PCr': avg_season_states[0], 'Gly': avg_season_states[1], 
                'Lac': avg_season_states[2], 'M': avg_season_states[3], 
                'CNS': avg_season_states[4], 'Phi': avg_season_states[5], 'P': avg_P_season
            })
            
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'data', 'results', 'lebron_career_fatigue_results.csv'), index=False)
    
    # Save bio averages to txt file
    with open(os.path.join(base_dir, 'data', 'results', 'lebron_playoff_bio_averages.txt'), 'w') as f:
        f.write("Year\tPCr\tGly\tLac\tM\tCNS\tPhi\tP(Prod)\n")
        f.write("-" * 65 + "\n")
        for b in bio_averages:
            f.write(f"{b['Year']}\t{b['PCr']:.3f}\t{b['Gly']:.3f}\t{b['Lac']:.3f}\t{b['M']:.3f}\t{b['CNS']:.3f}\t{b['Phi']:.3f}\t{b['P']:.3f}\n")
            
    print("Bio-fatigue simulation complete.")

if __name__ == "__main__":
    run_career_simulation()
