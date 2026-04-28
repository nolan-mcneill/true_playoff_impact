import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime, timedelta

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_path = os.path.join(base_dir, 'data', 'tpi_results', 'lebron_tpi_results.csv')
minutes_path = os.path.join(base_dir, 'data', 'fatigue_metric', "lebron's_playoff_minutes_full_career.csv")
graphs_dir = os.path.join(base_dir, 'data', 'tpi_results', 'graphs')
rs_dates_path = os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_reg_season_game_dates.csv')
pbp_path = os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_pbp_usg_components.csv')
os.makedirs(graphs_dir, exist_ok=True)

# Constants - BIOLOGICAL DIFFERENTIATION
ALPHA_ON = 0.048 
BETA_TIMEOUT = 0.018 # Per minute
BETA_REST_DAY = 0.15 # Per day
BETA_PLAYING = 0.001 # Per minute

plt.style.use('dark_background')
color_on, color_off, color_rest, bg_color = '#ff4d4d', '#2ecc71', '#3498db', '#0a0a0a'

def simulate_segment_plot(f_start, mode, duration, usg, t_offset):
    if duration <= 0: return f_start, [], [], mode
    num_steps = max(15, int(duration * 2))
    t_span = np.linspace(0, duration, num_steps)
    if mode == 'ON':
        f_vals = odeint(lambda f, t: ALPHA_ON * pow(usg/25,2) - BETA_PLAYING*f, f_start, t_span)
    else:
        f_vals = odeint(lambda f, t: -BETA_TIMEOUT*f, f_start, t_span)
    t_global = t_offset + t_span/1440.0
    return f_vals[-1][0], t_global.tolist(), f_vals.flatten().tolist(), mode

def parse_stretch(s):
    if pd.isna(s) or s == '-': return None
    try:
        m, t = s.split(': '); start, end = map(int, t.split('~'))
        return m, start, end
    except: return None

def simulate_game_plot(f_start, stretches, usg_map, to_map, t_day_start):
    f_curr, wall_min, segments = f_start, 0.0, []
    p_data = {}
    for i, s in enumerate(stretches):
        d = parse_stretch(s)
        if not d: continue
        p = {0:1,1:2,2:2,3:3,4:4,5:4,6:5,7:6}.get(i, 4)
        p_data.setdefault(p, []).append(d)
    
    if not p_data: return f_start, [], []
    for p in range(1, max(p_data.keys())+1):
        if p > 1:
            b_dur = 15.0 if p==3 else (5.0 if p>4 else 2.5)
            f_curr, ts, fs, m = simulate_segment_plot(f_curr, 'OFF', b_dur, 0, t_day_start + wall_min/1440.0)
            segments.append({'t':ts, 'f':fs, 'm':m}); wall_min += b_dur
        for (mode, start, end) in sorted(p_data.get(p, []), key=lambda x: x[1]):
            curr_c = start
            for to in sorted([t/60.0 for t in to_map.get(p, [])]):
                if curr_c < to < end:
                    dur = to - curr_c
                    f_curr, ts, fs, m = simulate_segment_plot(f_curr, mode, dur, usg_map.get(p, 30), t_day_start + wall_min/1440.0)
                    segments.append({'t':ts, 'f':fs, 'm':m}); wall_min += dur
                    f_curr, ts, fs, m = simulate_segment_plot(f_curr, 'OFF', 3.0, 0, t_day_start + wall_min/1440.0)
                    segments.append({'t':ts, 'f':fs, 'm':m}); wall_min += 3.0
                    curr_c = to
            dur = end - curr_c
            if dur > 0:
                f_curr, ts, fs, m = simulate_segment_plot(f_curr, mode, dur, usg_map.get(p, 30), t_day_start + wall_min/1440.0)
                segments.append({'t':ts, 'f':fs, 'm':m}); wall_min += dur
    return f_curr, segments

def parse_rs_dates():
    rs_map = {}
    if not os.path.exists(rs_dates_path): return {}
    with open(rs_dates_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(',')
        if not parts: continue
        year_str = parts[0]
        start_y = int(year_str.split('-')[0]); end_y = 2000 + int(year_str.split('-')[1]) if '-' in year_str else start_y
        dates = []
        for val in parts[1:]:
            if not val or '/' not in val: continue
            try:
                mm, dd = map(int, val.split('/'))
                dates.append(datetime(start_y if mm >= 10 else end_y, mm, dd))
            except: continue
        rs_map[str(end_y)] = sorted(dates)
    return rs_map

# Data Loading
df_min = pd.read_csv(minutes_path)
df_pbp = pd.read_csv(pbp_path)
df_pbp['TO'] = df_pbp['Timeouts'].apply(lambda x: [float(t) for t in str(x).split(',')] if pd.notna(x) and str(x)!="" else [])
df_pbp['USG_C'] = 100 * ((df_pbp['L_FGA'] + 0.44*df_pbp['L_FTA'] + df_pbp['L_TOV']) / (df_pbp['T_FGA'] + 0.44*df_pbp['T_FTA'] + df_pbp['T_TOV']).replace(0,1))
pbp_2018 = {idx: (g.set_index('Period')['USG_C'].to_dict(), g.set_index('Period')['TO'].to_dict()) 
            for idx, (gid, g) in enumerate(df_pbp[df_pbp['Year']==2018].groupby('Game_ID'))}
rs_game_map = parse_rs_dates()

def simulate_2018_full():
    rs_games = rs_game_map.get('2018', [])
    po_data = df_min[df_min['Year']==2018]
    t0 = rs_games[0]
    f_accum, segs_rs, segs_po, po_info = 0.0, [], [], []
    # RS
    last_d = t0
    for g_d in rs_games:
        gap = (g_d - last_d).total_seconds() / 86400.0
        if gap > 0:
            tr = np.linspace(0, gap, 8); fs = f_accum * np.exp(-BETA_REST_DAY * tr)
            segs_rs.append({'t': [((last_d-t0).total_seconds()+x*86400)/86400 for x in tr], 'f': fs.tolist(), 'm': 'REST'})
            f_accum = fs[-1]
        t_off = (g_d - t0).total_seconds() / 86400.0
        f_accum, segs = simulate_game_plot(f_accum, ["ON: 0~10","OFF: 0~3","ON: 3~10","ON: 0~9","OFF: 0~3","ON: 3~9"], {1:30,2:30,3:30,4:30}, {}, t_off)
        segs_rs.extend(segs); f_accum = segs[-1]['f'][-1]; last_d = g_d
    
    po_start_date = datetime(2018, 4, 15)
    t_po_start = (po_start_date-t0).total_seconds()/86400.0
    gap_po = (po_start_date - last_d).total_seconds() / 86400.0
    if gap_po > 0:
        tr = np.linspace(0, gap_po, 10); fs = f_accum * np.exp(-BETA_REST_DAY * tr)
        segs_po.append({'t': [(last_d-t0).total_seconds()/86400+x for x in tr], 'f': fs.tolist(), 'm': 'REST'})
        f_accum = fs[-1]
    
    days_el = t_po_start
    for idx, (i, row) in enumerate(po_data.iterrows()):
        if idx > 0:
            tr = np.linspace(0, 2.0, 20); fs = f_accum * np.exp(-BETA_REST_DAY * tr)
            segs_po.append({'t': [days_el + x for x in tr], 'f': fs.tolist(), 'm': 'REST'})
            f_accum = fs[-1]; days_el += 2.0
        g_start_idx = len(segs_po)
        usg, to = pbp_2018.get(idx, ({1:35,2:35,3:35,4:35}, {}))
        f_accum, segs = simulate_game_plot(f_accum, [row[f'Stretch {k}'] for k in range(1, 8)], usg, to, days_el)
        segs_po.extend(segs); f_accum = segs_po[-1]['f'][-1]; days_el = segs_po[-1]['t'][-1]
        po_info.append({'start': g_start_idx, 'end': len(segs_po), 'round': row['Rnd']})
    return segs_rs, segs_po, po_info, t_po_start

segs_rs, segs_po, po_info, t_po_start = simulate_2018_full()

fig, axes = plt.subplots(5, 1, figsize=(20, 65), facecolor=bg_color)
for ax in axes: ax.set_facecolor(bg_color); ax.grid(alpha=0.03)

# LEVEL 1: Intra-Game
g1 = [g for g in po_info if g['round'] == 'F'][0]
g1_segs = segs_po[g1['start']:g1['end']]
t0_g = g1_segs[0]['t'][0]
for s in g1_segs:
    axes[0].plot((np.array(s['t'])-t0_g)*1440, s['f'], color=(color_on if s['m']=='ON' else color_off), linewidth=5, solid_capstyle='round')
axes[0].set_title('LEVEL 1: Intra-Game Mechanics (Finals G1)', fontsize=24, fontweight='bold', pad=25)

# LEVEL 2: Series (Snake Zoom 30x)
f_meta = [g for g in po_info if g['round'] == 'F']
s_segs = segs_po[f_meta[0]['start']-1:f_meta[-1]['end']]
current_x = 0
for s in s_segs:
    dt = np.array(s['t']) - s['t'][0]
    is_game = s['m'] in ['ON', 'OFF']
    x_v = current_x + (dt * 30.0 if is_game else dt)
    if is_game: axes[1].axvspan(x_v[0], x_v[-1], color='white', alpha=0.1, linewidth=0)
    axes[1].plot(x_v, s['f'], color=(color_on if s['m']=='ON' else (color_off if s['m']=='OFF' else color_rest)), linewidth=3, solid_capstyle='round')
    current_x = x_v[-1]
axes[1].set_title('LEVEL 2: Series Fatigue [Snake Zoom 30x]', fontsize=24, fontweight='bold', pad=25)
axes[1].set_xticks([])

# LEVEL 3: Postseason (Snake Zoom 30x)
current_x = 0
for s in segs_po:
    dt = np.array(s['t']) - s['t'][0]
    is_game = s['m'] in ['ON', 'OFF']
    x_v = current_x + (dt * 30.0 if is_game else dt)
    if is_game: axes[2].axvspan(x_v[0], x_v[-1], color='white', alpha=0.1, linewidth=0)
    axes[2].plot(x_v, s['f'], color=(color_on if is_game else color_rest), linewidth=2, solid_capstyle='round')
    current_x = x_v[-1]
axes[2].set_title('LEVEL 3: Full Postseason [Snake Zoom 30x]', fontsize=24, fontweight='bold', pad=25)
axes[2].set_xticks([])

# LEVEL 4: Regular Season
for s in segs_rs:
    c = (color_on if s['m'] in ['ON','OFF'] else color_rest)
    axes[3].plot(s['t'], s['f'], color=c, linewidth=1.5, alpha=0.8)
axes[3].set_title('LEVEL 4: Regular Season Profile (Actual Game Dates)', fontsize=24, fontweight='bold', pad=25)

# LEVEL 5: OVERALL SEASON (RS + PO) [Snake Zoom 15x]
all_segs = segs_rs + segs_po
current_x = 0
for s in all_segs:
    dt = np.array(s['t']) - s['t'][0]
    is_game = s['m'] in ['ON', 'OFF']
    x_v = current_x + (dt * 15.0 if is_game else dt) # 15x stretch for the whole season
    if is_game: axes[4].axvspan(x_v[0], x_v[-1], color='white', alpha=0.08, linewidth=0)
    axes[4].plot(x_v, s['f'], color=(color_on if is_game else color_rest), linewidth=1, alpha=0.7)
    current_x = x_v[-1]
# Mark Playoff Start in Snake Coordinates
# Need to find the exact current_x at the bridge
# Let's just draw a vertical line where segs_rs ends
rs_end_x = 0
for s in segs_rs:
    dt = np.array(s['t']) - s['t'][0]
    rs_end_x += (dt * 15.0 if s['m'] in ['ON','OFF'] else dt)[-1]
axes[4].axvline(rs_end_x, color='white', linestyle='--', alpha=0.6)
axes[4].text(rs_end_x, -0.05, 'PLAYOFFS START', color='white', fontweight='bold', ha='center', va='top', transform=axes[4].get_xaxis_transform())
axes[4].set_title('LEVEL 5: Overall Season Fatigue (RS + PO) [Snake Zoom 15x]', fontsize=24, fontweight='bold', pad=25)
axes[4].set_xticks([]); axes[4].set_ylim(0, max([v for s in all_segs for v in s['f']])*1.25)

plt.tight_layout(pad=16.0); plt.savefig(os.path.join(graphs_dir, 'fatigue_engine_mechanics.png'), dpi=300); plt.close()
print("Final 5-level high-fidelity visuals with full season context saved.")
