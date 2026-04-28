import pandas as pd
import numpy as np
from scipy.integrate import odeint
import os

# Micro-Fatigue Constants (Optimized)
ALPHA_ON = 0.04
BETA_ON = 0.08
GAMMA_BENCH = 0.008

# Macro-Fatigue Constants (Recovery)
R_CONST = 0.01

def model_on(f, t, usg):
    intensity = usg / 25.0
    dfdt = ALPHA_ON * intensity - BETA_ON * f
    return dfdt

def model_off(f, t):
    dfdt = -GAMMA_BENCH * f
    return dfdt

def parse_stretch(stretch_str):
    if pd.isna(stretch_str) or stretch_str == '-':
        return None
    try:
        mode, times = stretch_str.split(': ')
        start, end = map(int, times.split('~'))
        return mode, start, end
    except:
        return None

def calculate_usg_rate(l_fga, l_fta, l_tov, t_fga, t_fta, t_tov):
    player_pts = l_fga + 0.44 * l_fta + l_tov
    team_pts = t_fga + 0.44 * t_fta + t_tov
    if team_pts == 0:
        return 30.0
    usg = 100 * (player_pts / team_pts)
    return max(5.0, min(60.0, usg))

def simulate_segment(f_start, mode, duration, usg):
    if duration <= 0:
        return f_start, 0, 0
    num_steps = int(duration) + 1
    if num_steps < 2: num_steps = 2
    t_span = np.linspace(0, duration, num_steps)
    if mode == 'ON':
        f_vals = odeint(model_on, f_start, t_span, args=(usg,))
        # Sum fatigue for average calculation
        # Skip last point to avoid double counting segments
        seg_fatigue_sum = np.sum(f_vals[:-1])
        seg_minutes = duration
    else:
        f_vals = odeint(model_off, f_start, t_span)
        seg_fatigue_sum = 0
        seg_minutes = 0
    return f_vals[-1][0], seg_fatigue_sum, seg_minutes

def simulate_game(f_start, stretches, usg_by_period, timeouts_by_period):
    t_current = f_start
    f_current = f_start
    
    stretch_to_period = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6}
    
    game_fatigue_sum = 0
    game_minutes = 0
    
    for i, s in enumerate(stretches):
        data = parse_stretch(s)
        if not data: continue
        
        mode, start, end = data
        period = stretch_to_period.get(i, 4)
        usg = usg_by_period.get(period, 30.0)
        
        # Timeouts in this period (seconds converted to minutes)
        raw_to = timeouts_by_period.get(period, [])
        tos = sorted([t/60.0 for t in raw_to])
        
        # Current stretch interval: [start, end]
        # We need to inject timeouts that fall within this interval
        current_pos = start
        for to_time in tos:
            if start < to_time < end:
                # 1. Simulate from current_pos to timeout
                dur = to_time - current_pos
                f_current, s_sum, s_min = simulate_segment(f_current, mode, dur, usg)
                game_fatigue_sum += s_sum
                game_minutes += s_min
                
                # 2. Add 3-minute Timeout (Always OFF)
                # Note: Timeout duration is 3 mins, but we don't count it in 'game_minutes'
                f_current, _, _ = simulate_segment(f_current, 'OFF', 3.0, usg)
                
                current_pos = to_time
        
        # Final part of the stretch
        dur = end - current_pos
        if dur > 0:
            f_current, s_sum, s_min = simulate_segment(f_current, mode, dur, usg)
            game_fatigue_sum += s_sum
            game_minutes += s_min
            
    avg_game_fatigue = game_fatigue_sum / game_minutes if game_minutes > 0 else 0
    return f_current, avg_game_fatigue

def main():
    minutes_file = "data/fatigue_metric/lebron's_playoff_minutes_full_career.csv"
    pbp_file = "data/fatigue_metric/lebron_pbp_usg_components.csv"
    
    df_min = pd.read_csv(minutes_file)
    df_pbp = pd.read_csv(pbp_file)
    
    df_pbp['USG'] = df_pbp.apply(lambda r: calculate_usg_rate(r['L_FGA'], r['L_FTA'], r['L_TOV'], r['T_FGA'], r['T_FTA'], r['T_TOV']), axis=1)
    
    # Process Timeouts column
    def parse_timeouts(to_str):
        if pd.isna(to_str) or to_str == "": return []
        return [float(x) for x in str(to_str).split(',')]
    
    df_pbp['TO_List'] = df_pbp['Timeouts'].apply(parse_timeouts)
    
    pbp_games = {}
    for (year, gid), group in df_pbp.groupby(['Year', 'Game_ID']):
        usg_map = group.set_index('Period')['USG'].to_dict()
        to_map = group.set_index('Period')['TO_List'].to_dict()
        if year not in pbp_games: pbp_games[year] = []
        pbp_games[year].append((usg_map, to_map))
    
    for year in pbp_games:
        # Sort by Game_ID
        unique_gids = sorted(df_pbp[df_pbp['Year']==year]['Game_ID'].unique())
        gid_to_data = {gid: (u, t) for gid, u, t in zip(unique_gids, [g[0] for g in pbp_games[year]], [g[1] for g in pbp_games[year]])}
        pbp_games[year] = [gid_to_data[gid] for gid in unique_gids]
    
    results = []
    
    for year in sorted(df_min['Year'].unique()):
        year_rows = df_min[df_min['Year'] == year]
        game_avg_fatigues = []
        f_accumulated = 0
        
        for idx, (i, row) in enumerate(year_rows.iterrows()):
            year_pbp = pbp_games.get(year, [])
            if idx < len(year_pbp):
                usg_map, to_map = year_pbp[idx]
            else:
                usg_map, to_map = {1:30, 2:30, 3:30, 4:30}, {}
            
            stretches = [row[f'Stretch {k}'] for k in range(1, 8)]
            f_accumulated, avg_f = simulate_game(f_accumulated, stretches, usg_map, to_map)
            game_avg_fatigues.append(avg_f)
            
            recovery_time = 48 * 60
            f_accumulated = f_accumulated * np.exp(-R_CONST * recovery_time / 1440.0)
            
        results.append({
            'Year': year,
            'Fatigue_Avg': np.mean(game_avg_fatigues)
        })
        
    output_df = pd.DataFrame(results)
    output_df.to_csv("data/fatigue_metric/lebron_career_fatigue_results.csv", index=False)
    print("Career fatigue simulation (with timeouts) complete!")
    print(output_df)

if __name__ == "__main__":
    main()
