import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime

# ─── Path configuration ───────────────────────────────────────────────────────
base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
graphs_dir   = os.path.join(base_dir, 'data', 'results', 'graphs')
results_path = os.path.join(base_dir, 'data', 'results', 'lebron_tpi_results.csv')
minutes_path = os.path.join(base_dir, 'data', 'raw', "lebron's_playoff_minutes_full_career.csv")
rs_dates_path= os.path.join(base_dir, 'data', 'raw', 'lebron_reg_season_game_dates.csv')
pbp_path     = os.path.join(base_dir, 'data', 'processed', 'lebron_pbp_usg_components.csv')
metrics_path = os.path.join(base_dir, 'data', 'raw', 'lebron_tpi_metrics_final.csv')
os.makedirs(graphs_dir, exist_ok=True)

from bio_model import bio_model_ode, bio_model_off, bio_model_rest, get_P

def simulate_segment_plot(state, mode, duration, p_stats=None):
    """Returns (final_state, t_sparse, hist_sparse_Nx6).

    Simulation correctness: always integrates over the FULL duration so the
    returned final_state is accurate and the next segment inherits the correct
    bio-state. No time periods are skipped or truncated.

    Plotting efficiency: the returned t/hist arrays are downsampled to at most
    PLOT_POINTS so REST corridors render as smooth curves, not vertical bars.

    duration units:  MINUTES for ON/OFF  |  HOURS for REST
    """
    PLOT_POINTS = 12

    if duration <= 0:
        return state, np.array([]), np.empty((0, 6))

    if mode == 'REST':
        # duration is in MINUTES
        n_full = max(PLOT_POINTS, min(120, int(duration/60.0) + 1))
        t_full = np.linspace(0, duration, n_full)
        hist_full = odeint(bio_model_rest, state, t_full)
        # Downsample for plotting only
        idx = np.round(np.linspace(0, n_full - 1, PLOT_POINTS)).astype(int)
        t_plot, hist_plot = t_full[idx], hist_full[idx]
    elif mode == 'ON':
        n = max(2, int(duration * 2))
        t_full = np.linspace(0, duration, n)
        pm      = p_stats.get('Period_Min', 12.0)
        pace    = p_stats.get('Pace', 95.0)
        ast_pm  = p_stats.get('L_AST', 0) / pm
        Contact = (p_stats.get('L_FD',0) + p_stats.get('L_PF',0) + p_stats.get('L_FTA',0)) / pm
        Explode = (p_stats.get('L_TRB',0) + p_stats.get('L_STL',0) + p_stats.get('L_BLK',0)) / pm
        I_mov   = (pace/100.0) + (ast_pm*0.2)
        I_col   = (Contact*0.7) + (Explode*0.3)
        I_tot   = (I_mov*0.5) + (I_col*0.5)
        hist_full = odeint(bio_model_ode, state, t_full, args=(I_tot, I_col))
        t_plot, hist_plot = t_full, hist_full
    else:  # OFF (timeout / between-period break)
        n = max(2, int(duration * 2))
        t_full = np.linspace(0, duration, n)
        hist_full = odeint(bio_model_off, state, t_full)
        t_plot, hist_plot = t_full, hist_full

    hist_full[:, :5] = np.clip(hist_full[:, :5], 0.0, 1.0)
    hist_full[:, 5]  = np.maximum(hist_full[:, 5], 0.0)
    hist_plot[:, :5] = np.clip(hist_plot[:, :5], 0.0, 1.0)
    hist_plot[:, 5]  = np.maximum(hist_plot[:, 5], 0.0)
    return hist_full[-1].tolist(), t_plot, hist_plot

def parse_stretch(s):
    if pd.isna(s) or s == '-': return None
    try:
        m, times = s.split(': '); st, en = map(int, times.split('~'))
        return m, st, en
    except: return None

def simulate_game_plot(state, stretches, pbp_map, pace, t_day_start):
    """Returns (final_state, segments) where each segment = {t, hist, mode}."""
    stretch_to_period = {0:1,1:2,2:2,3:3,4:4,5:4,6:5,7:6}
    segments = []
    wall_min = 0.0
    last_period = 1

    for i, s in enumerate(stretches):
        data = parse_stretch(s)
        if not data: continue
        mode, start, end = data
        period = stretch_to_period.get(i, 4)

        if period > last_period:
            b_dur = 15.0 if period == 3 else (5.0 if period > 4 else 2.5)  # minutes
            state, t_seg, hist = simulate_segment_plot(state, 'OFF', b_dur)
            segments.append({'t': t_day_start + (wall_min + t_seg)/1440.0, 'hist': hist, 'mode': 'OFF'})
            wall_min += b_dur
            last_period = period

        p_stats = pbp_map.get(period, {})
        p_stats['Period_Min'] = 5.0 if period > 4 else 12.0
        p_stats['Pace'] = pace
        tos = sorted([t_/60.0 for t_ in p_stats.get('Timeouts', [])])

        curr = start
        for to_t in tos:
            if start < to_t < end:
                dur = to_t - curr
                state, t_seg, hist = simulate_segment_plot(state, mode, dur, p_stats)
                segments.append({'t': t_day_start + (wall_min + t_seg)/1440.0, 'hist': hist, 'mode': mode})
                wall_min += dur
                # timeout break
                state, t_seg, hist = simulate_segment_plot(state, 'OFF', 3.0)
                segments.append({'t': t_day_start + (wall_min + t_seg)/1440.0, 'hist': hist, 'mode': 'OFF'})
                wall_min += 3.0
                curr = to_t
        dur = end - curr
        if dur > 0:
            state, t_seg, hist = simulate_segment_plot(state, mode, dur, p_stats)
            segments.append({'t': t_day_start + (wall_min + t_seg)/1440.0, 'hist': hist, 'mode': mode})
            wall_min += dur

    return state, segments

def parse_rs_dates():
    rs_map = {}
    if not os.path.exists(rs_dates_path): return {}
    with open(rs_dates_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(',')
        if not parts: continue
        year_str = parts[0]
        start_y = int(year_str.split('-')[0])
        end_y   = 2000+int(year_str.split('-')[1]) if '-' in year_str else start_y
        dates = []
        for val in parts[1:]:
            if not val or '/' not in val: continue
            try:
                mm, dd = map(int, val.split('/'))
                dates.append(datetime(start_y if mm >= 10 else end_y, mm, dd))
            except: continue
        rs_map[str(end_y)] = sorted(dates)
    return rs_map

# ─── Load Data ────────────────────────────────────────────────────────────────
df_min  = pd.read_csv(minutes_path)
df_pbp  = pd.read_csv(pbp_path)
df_met  = pd.read_csv(metrics_path)
df_pbp['Timeouts'] = df_pbp['Timeouts'].apply(
    lambda x: [float(t) for t in str(x).split(',')] if pd.notna(x) and str(x) != "" else [])
rs_game_map = parse_rs_dates()

def build_pbp_map(year):
    """Build {game_idx: {period: stats_dict}} for a given year."""
    df_y = df_pbp[df_pbp['Year'] == year].copy()
    result = {}
    for idx, (gid, g) in enumerate(df_y.groupby('Game_ID')):
        result[idx] = {row['Period']: row.to_dict() for _, row in g.iterrows()}
    return result

# ─── 2018 Full Simulation ─────────────────────────────────────────────────────
def simulate_2018_full():
    rs_games = rs_game_map.get('2018', [])
    po_data  = df_min[df_min['Year'] == 2018]
    pbp_map_2018 = build_pbp_map(2018)
    rs_pace = df_met[df_met['Year']==2018]['RS_Pace'].iloc[0] if not df_met[df_met['Year']==2018].empty else 95.0
    po_pace = df_met[df_met['Year']==2018]['PO_Pace'].iloc[0] if not df_met[df_met['Year']==2018].empty else 90.0

    state    = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    t0       = rs_games[0] if rs_games else datetime(2017, 10, 17)
    segs_rs, segs_gap, segs_po, po_info = [], [], [], []
    last_date = None

    # Regular Season
    for g_date in rs_games:
        if last_date:
            gap = (g_date - last_date).days
            if gap > 0:
                state, t_seg, hist = simulate_segment_plot(state, 'REST', gap * 1440.0)
                t_base = (g_date - t0).days - gap
                segs_rs.append({'t': t_base + t_seg / 1440.0, 'hist': hist, 'mode': 'REST'})
        rs_ps = {'Pace': rs_pace, 'L_AST':2.0,'L_FD':2.0,'L_PF':0.5,'L_FTA':2.0,'L_TRB':2.0,'L_STL':0.5,'L_BLK':0.2}
        rs_pbp = {p: rs_ps for p in range(1,5)}
        t_off = (g_date - t0).days
        state, segs = simulate_game_plot(state, ["ON: 0~10","OFF: 0~4","ON: 4~10","ON: 0~10","OFF: 0~4","ON: 4~10"], rs_pbp, rs_pace, t_off)
        segs_rs.extend(segs)
        last_date = g_date

    # Rest before playoffs (Gap)
    po_start = datetime(2018, 4, 15)
    if last_date:
        gap = (po_start - last_date).days
        if gap > 0:
            state, t_seg, hist = simulate_segment_plot(state, 'REST', gap * 1440.0)
            t_base = (po_start - t0).days - gap
            segs_gap.append({'t': t_base + t_seg / 1440.0, 'hist': hist, 'mode': 'REST'})

    days_el = (po_start - t0).days
    segments_for_po = [] # temporary to get po_info indices relative to segs_po
    for idx, (i, row) in enumerate(po_data.iterrows()):
        if idx > 0:
            state, t_seg, hist = simulate_segment_plot(state, 'REST', 2.0 * 1440.0)
            s_rec = {'t': days_el + t_seg / 1440.0, 'hist': hist, 'mode': 'REST'}
            segs_po.append(s_rec)
            segments_for_po.append(s_rec)
            days_el += 2.0
        
        g_start_idx = len(segments_for_po)
        game_pbp = pbp_map_2018.get(idx, {})
        state, segs = simulate_game_plot(state, [row[f'Stretch {k}'] for k in range(1,8)], game_pbp, po_pace, days_el)
        segs_po.extend(segs)
        segments_for_po.extend(segs)
        if segs:
            days_el = float(np.max(segs[-1]['t']))
        po_info.append({'start': g_start_idx, 'end': len(segments_for_po), 'round': row['Rnd']})

    return segs_rs, segs_gap, segs_po, po_info

segs_rs, segs_gap, segs_po, po_info = simulate_2018_full()
segs_full = segs_rs + segs_gap + segs_po

# ─── Calculate Normalization Factors ─────────────────────────────────────────
# Scale metrics to their observed maximum (% of Max) for visual consistency
all_data = [s['hist'] for s in segs_full if s['hist'].size > 0]
if all_data:
    stacked = np.vstack(all_data)
    maxes = np.max(stacked, axis=0)[:5]
    # Scaling factor = 1.0 / max_observed (prevent division by zero)
    SCALE_FACTORS = 1.0 / np.where(maxes < 1e-4, 1.0, maxes)
else:
    SCALE_FACTORS = np.ones(5)
print(f"Bio-Metric Scaling Factors (% of Max): {dict(zip(['PCr', 'Gly', 'Lac', 'M', 'CNS'], SCALE_FACTORS))}")

# ─── Plotting Helpers ─────────────────────────────────────────────────────────
plt.style.use('dark_background')
BG   = '#080810'
COLS = {
    'P':   '#FFFFFF',   # Production multiplier — bold white
    'PCr': '#00e5ff',   # Cyan
    'Gly': '#ff00ff',   # Magenta
    'Lac': '#69ff47',   # Green
    'M':   '#ff9100',   # Orange
    'CNS': '#ffff00',   # Yellow
}
METRICS = ['PCr', 'Gly', 'Lac', 'M', 'CNS']  # indices 0-4 in state
SUB_ALPHA = 0.65
SUB_LW    = 1.4
MAIN_LW   = 2.8

def plot_segments(ax, segments, zoom=1.0, x_offset=0.0, show_submetrics=True, local_normalize=False, use_minutes=False):
    """Plot 5 bio sub-metrics from a list of segment dicts."""
    current_x = x_offset
    
    # If local_normalize, find max within these segments only
    l_factors = SCALE_FACTORS
    if local_normalize:
        all_h = [s['hist'] for s in segments if s['hist'].size > 0]
        if all_h:
            stack = np.vstack(all_h)
            l_max = np.max(stack, axis=0)[:5]
            l_factors = 1.0 / np.where(l_max < 1e-4, 1.0, l_max)

    t0_global = segments[0]['t'][0] if segments else 0
    for seg in segments:
        t_arr = np.asarray(seg['t'])
        hist  = seg['hist']
        if hist.shape[0] == 0 or len(t_arr) == 0: continue
        is_play = seg['mode'] in ('ON', 'OFF')
        
        if use_minutes:
            x = (t_arr - t0_global) * 1440.0
        else:
            dt = t_arr - t_arr[0]
            x  = current_x + (dt * zoom if is_play else dt)
            
        if seg['mode'] == 'ON':
            ax.axvspan(x[0], x[-1], color='#ffffff', alpha=0.15, linewidth=0)
        elif seg['mode'] == 'OFF':
            ax.axvspan(x[0], x[-1], color='#ffffff', alpha=0.05, linewidth=0)
        # Sub-metrics (Scaled + Smoothing for macro views)
        for mi, name in enumerate(METRICS):
            vals = hist[:, mi] * l_factors[mi]
            # Aggressive smoothing for macro views to show trends, not barcodes
            if not use_minutes and zoom < 40.0 and len(vals) > 4:
                # PCr (mi=0) needs more smoothing as it's a fast-twitch metric
                w_factor = 40.0 if mi == 0 else 20.0
                w = int(w_factor / (zoom**0.5 + 0.1))
                w = max(1, min(w, len(vals)//2))
                if w > 1:
                    vals = np.convolve(vals, np.ones(w)/w, mode='same')
            
            ax.plot(x, vals, color=COLS[name], lw=MAIN_LW if use_minutes else SUB_LW, alpha=0.9 if use_minutes else SUB_ALPHA)
            
            # Add dots at the edges of segments in Graph 1 (use_minutes is True for Graph 1)
            if use_minutes:
                ax.scatter([x[0], x[-1]], [vals[0], vals[-1]], color=COLS[name], s=30, edgecolors='white', zorder=25)
        
        current_x = x[-1]
    return current_x

def plot_discrete_series(ax, segments, info_list, group_by_series=False):
    """Plot bio metrics with dots and recovery lines using non-linear time stretching."""
    series_g_counts = [7, 4, 7, 4]
    series_boundaries = []
    curr = 0
    for c in series_g_counts:
        series_boundaries.append(curr + c - 1)
        curr += c

    # Graph 3 (group_by_series=False) shows intra-game details, Graph 4 (group_by_series=True) shows series trends
    GAME_STRETCH = 100.0 if not group_by_series else 10.0
    REST_STRETCH = 5.0  # Inter-series
    INTRA_REST_STRETCH = 2.0 # Stretch the rest between games within a series
    
    # 1. Build warped timelines
    real_ts = []
    warp_ts = []
    curr_w_t = 0.0
    t_start_all = segments[info_list[0]['start']]['t'][0]
    real_ts.append(t_start_all)
    warp_ts.append(0.0)

    for i, g in enumerate(info_list):
        if i > 0:
            gap_start = segments[info_list[i-1]['end']-1]['t'][-1]
            gap_end = segments[g['start']]['t'][0]
            dt = gap_end - gap_start
            is_inter = (i-1) in series_boundaries
            stretch = REST_STRETCH if (group_by_series and is_inter) else (INTRA_REST_STRETCH if group_by_series else 1.0)
            curr_w_t += dt * stretch
            real_ts.append(gap_end)
            warp_ts.append(curr_w_t)
            
        g_start = segments[g['start']]['t'][0]
        g_end   = segments[g['end']-1]['t'][-1]
        dt = g_end - g_start
        curr_w_t += dt * GAME_STRETCH
        real_ts.append(g_end)
        warp_ts.append(curr_w_t)

    def warp(t): return np.interp(t, real_ts, warp_ts)

    # 2. Plotting
    if not group_by_series:
        # Graph 3 Logic: Quarterly Trend within Games
        for i, g in enumerate(info_list):
            w0, w1 = warp(segments[g['start']]['t'][0]), warp(segments[g['end']-1]['t'][-1])
            ax.axvspan(w0, w1, color='#ffffff', alpha=0.1, lw=0)
            
            # Sub-divide game into 4 quarters
            g_segs = segments[g['start']:g['end']]
            if not g_segs: continue
            
            # 1. Segment-Boundary Trend Points
            seg_pts = []
            for seg in g_segs:
                wt0 = warp(seg['t'][0])
                wt1 = warp(seg['t'][-1])
                seg_pts.append((wt0, seg['hist'][0, :5]))
                seg_pts.append((wt1, seg['hist'][-1, :5]))
            
            # Remove duplicate edge points (wt1 of seg N is wt0 of seg N+1)
            unique_pts = []
            if seg_pts:
                unique_pts.append(seg_pts[0])
                for p in seg_pts[1:]:
                    if p[0] > unique_pts[-1][0] + 1e-3:
                        unique_pts.append(p)
            
            for mi, name in enumerate(METRICS):
                tx = [p[0] for p in unique_pts]
                ty = [p[1][mi]*SCALE_FACTORS[mi] for p in unique_pts]
                ax.plot(tx, ty, color=COLS[name], ls='--', lw=1.2, alpha=0.5)
                # Consistent marker size restored
                ax.scatter(tx, ty, color=COLS[name], s=20, edgecolors='white', alpha=0.7, zorder=10)

            # Recovery lines
            if i < len(info_list)-1:
                for seg in segments[info_list[i]['end'] : info_list[i+1]['start']]:
                    wt = warp(np.asarray(seg['t']))
                    for mi, name in enumerate(METRICS):
                        ax.plot(wt, seg['hist'][:, mi]*SCALE_FACTORS[mi], color=COLS[name], lw=2.5, alpha=0.9)
            
    else:
        # Graph 4 Logic: Game-by-game Trend within Series
        curr = 0
        for s_idx, count in enumerate(series_g_counts):
            series_games = info_list[curr : curr + count]
            t0, t1 = warp(segments[series_games[0]['start']]['t'][0]), warp(segments[series_games[-1]['end']-1]['t'][-1])
            ax.axvspan(t0, t1, color='#ffffff', alpha=0.15, lw=1.5, edgecolor='#555577')
            
            game_pts = []
            # 1. Start-of-Series Certainty Point
            v_start_s = segments[series_games[0]['start']]['hist'][0, :5]
            game_pts.append((t0, v_start_s))
            
            # 2. Game Averages
            for g in series_games:
                gw0, gw1 = warp(segments[g['start']]['t'][0]), warp(segments[g['end']-1]['t'][-1])
                g_segs = segments[g['start']:g['end']]
                v_avg = np.mean(np.vstack([s['hist'] for s in g_segs]), axis=0)[:5]
                game_pts.append(((gw0+gw1)/2.0, v_avg))
            
            # 3. End-of-Series Certainty Point
            v_end_s = segments[series_games[-1]['end']-1]['hist'][-1, :5]
            game_pts.append((t1, v_end_s))

            # Re-order metrics to put Lactate (Green) on top of others if they overlap at 1.0
            PLOT_ORDER = ['PCr', 'Gly', 'M', 'CNS', 'Lac']
            for name in PLOT_ORDER:
                mi = METRICS.index(name)
                tx = [p[0] for p in game_pts]
                ty = [p[1][mi]*SCALE_FACTORS[mi] for p in game_pts]
                ax.plot(tx, ty, color=COLS[name], ls='--', lw=2, alpha=0.8)
                
                # Markers
                ax.scatter([tx[0], tx[-1]], [ty[0], ty[-1]], color=COLS[name], s=60, edgecolors='white', zorder=15)
                if len(tx) > 2:
                    ax.scatter(tx[1:-1], ty[1:-1], color=COLS[name], s=25, edgecolors='white', alpha=0.9, zorder=12)
            
            if s_idx < len(series_g_counts)-1:
                next_g_first = info_list[curr + count]
                for seg in segments[series_games[-1]['end'] : next_g_first['start']]:
                    wt = warp(np.asarray(seg['t']))
                    for mi, name in enumerate(METRICS):
                        ax.plot(wt, seg['hist'][:, mi]*SCALE_FACTORS[mi], color=COLS[name], lw=3, alpha=0.9)
            curr += count

    ax.set_xlim(warp_ts[0], warp_ts[-1])

def add_legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=COLS['PCr'],  lw=2.5, label='PCr'),
        Line2D([0],[0], color=COLS['Gly'],  lw=2.5, label='Glycogen'),
        Line2D([0],[0], color=COLS['Lac'],  lw=2.5, label='Lactate'),
        Line2D([0],[0], color=COLS['M'],    lw=2.5, label='Muscle'),
        Line2D([0],[0], color=COLS['CNS'],  lw=2.5, label='CNS'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.4, 
              facecolor='#000000', edgecolor='#555577', ncol=5)

# ─── Build Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(22, 68), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.grid(alpha=0.04, color='white')
    ax.spines[['top','right','left','bottom']].set_color('#333355')
    ax.tick_params(colors='#aaaacc', labelsize=9)

title_kw = dict(fontsize=22, fontweight='bold', color='white', pad=22,
                fontfamily='DejaVu Sans')
sub_kw   = dict(fontsize=11, color='#aaaacc', pad=6, fontfamily='DejaVu Sans')

# ── Panel 1: 2018 Finals Game 1 (Intra-Game) ──────────────────────────────────
ax = axes[0]
finals_games = [g for g in po_info if g['round'] == 'F']
if finals_games:
    g1 = finals_games[0]
    g1_segs = segs_po[g1['start']:g1['end']]
    plot_segments(ax, g1_segs, zoom=1.0, use_minutes=True)
    ax.set_xlim(0, (g1_segs[-1]['t'][-1] - g1_segs[0]['t'][0]) * 1440.0)
    
    all_v = np.vstack([s['hist'][:,:5] for s in g1_segs])
    ymin, ymax = np.min(all_v), np.max(all_v)
    ax.set_ylim(max(0, ymin*0.8), min(1.1, ymax*1.1))

ax.set_ylabel('% of Metric Max', color='#aaaacc')
ax.set_title('GRAPH 1: 2018 NBA Finals - Game 1 Intra-Game Dynamics', **title_kw)
add_legend(ax)

# ── Panel 2: Recovery Between Game 1 and Game 2 ───────────────────────────────
ax = axes[1]
if len(finals_games) >= 2:
    g1_end = finals_games[0]['end']
    g2_start = finals_games[1]['start']
    recov_segs = segs_po[g1_end : g2_start]
    
    if recov_segs:
        t0_r = recov_segs[0]['t'][0]
        for seg in recov_segs:
            t_w = (np.asarray(seg['t']) - t0_r) * 1440.0 / 60.0 # Hours
            h = seg['hist']
            for mi, name in enumerate(METRICS):
                ax.plot(t_w, h[:, mi]*SCALE_FACTORS[mi], color=COLS[name], lw=3)
    
    # Add dots at absolute start and end of rest period
    v_start = recov_segs[0]['hist'][0, :5]
    v_end   = recov_segs[-1]['hist'][-1, :5]
    t_start = 0.0
    t_end   = (recov_segs[-1]['t'][-1] - t0_r) * 1440.0 / 60.0 # Hours
    for mi, name in enumerate(METRICS):
        ax.scatter([t_start, t_end], [v_start[mi]*SCALE_FACTORS[mi], v_end[mi]*SCALE_FACTORS[mi]], 
                   color=COLS[name], s=70, edgecolors='white', zorder=10)
            
    all_v = np.vstack([s['hist'][:,:5] for s in recov_segs])
    ymin, ymax = np.min(all_v), np.max(all_v)
    ax.set_ylim(max(0, ymin*0.9), min(1.1, ymax*1.05))
    ax.set_xlim(0, t_end)

ax.set_xlabel('Hours Post-Game 1', color='#aaaacc')
ax.set_ylabel('% of Metric Max', color='#aaaacc')
ax.set_title('GRAPH 2: Recovery Time Between 2018 Finals Game 1 and Game 2', **title_kw)
add_legend(ax)

# ── Panel 3: 2018 Finals (Series Overview - Discrete) ─────────────────────────
ax = axes[2]
if finals_games:
    plot_discrete_series(ax, segs_po, finals_games)
    
    # Extract only recovery/start/end points for scaling
    all_v = []
    for g in finals_games:
        all_v.append(segs_po[g['start']]['hist'][0, :5])
        all_v.append(segs_po[g['end']-1]['hist'][-1, :5])
    for i in range(len(finals_games)-1):
        g_end = finals_games[i]['end']
        g_next = finals_games[i+1]['start']
        for s in segs_po[g_end:g_next]: all_v.append(s['hist'][:,:5])
    
    all_v = np.vstack(all_v)
    ymin, ymax = np.min(all_v), np.max(all_v)
    ax.set_ylim(max(0, ymin*0.9), min(1.1, ymax*1.05))
ax.set_xticks([])
ax.set_xlabel('Relative Warped Time (Games Stretched 10x)', color='#aaaacc')
ax.set_ylabel('% of Metric Max', color='#aaaacc')
ax.set_title(f"GRAPH 3: 2018 NBA Finals - Game Start/End Fatigue (Warped Time)\n(Note: Games are horizontally stretched 50x to show net impact clearer.)", 
             color='white', fontsize=14, fontweight='bold', pad=20)
add_legend(ax)

# ── Panel 4: 2018 Playoffs (Discrete Playoff-Wide - Series Shading) ──────────
ax = axes[3]
if po_info:
    plot_discrete_series(ax, segs_po, po_info, group_by_series=True)
    
    # Scale based on series points and inter-series gaps
    all_v = []
    for g in po_info:
        all_v.append(segs_po[g['start']]['hist'][0, :5])
        all_v.append(segs_po[g['end']-1]['hist'][-1, :5])
    
    all_v = np.vstack(all_v)
    ymin, ymax = np.min(all_v), np.max(all_v)
    ax.set_ylim(max(0, ymin*0.9), min(1.1, ymax*1.05))
ax.set_xticks([])
ax.set_xlabel('Relative Warped Time (Games 10x, Intra-Rest 2x, Inter-Rest 5x)', color='#aaaacc')
ax.set_ylabel('% of Metric Max', color='#aaaacc')
ax.set_title('GRAPH 4: Entire 2018 Playoffs - Series-Centric Fatigue (Warped Time)\n(Note: Games stretched 10x, Intra-Series rest 2x, and Inter-Series rest 5x.)', **title_kw)
add_legend(ax)

# ── Panel 5: Seasonal Fatigue (RS -> PO) ───────────────────────────────────
ax = axes[4]
ax.set_facecolor(BG)

# RS block (length 2), Gap block (length 2 for visibility), PO block (length 2)
t_start_rs = segs_rs[0]['t'][0]
t_end_rs   = segs_rs[-1]['t'][-1]
t_start_po = segs_po[0]['t'][0]
t_end_po   = segs_po[-1]['t'][-1]

real_t5 = [t_start_rs, t_end_rs, t_start_po, t_end_po]
warp_t5 = [0.0, 2.0, 4.0, 6.0] 
def warp5(t): return np.interp(t, real_t5, warp_t5)

# RS Gray Box
ax.axvspan(0, 2, color='#ffffff', alpha=0.1, label='Regular Season')
# PO Gray Box
ax.axvspan(4, 6, color='#ffffff', alpha=0.1, label='Post-Season')

# RS Trends (5 Quintiles)
rs_pts = []
v_start_rs = segs_rs[0]['hist'][0, :5]
v_end_rs   = segs_rs[-1]['hist'][-1, :5]
rs_pts.append((0.0, v_start_rs)) # Start-of-RS Point
num_q = 5
for i in range(num_q):
    chunk = segs_rs[i*(len(segs_rs)//num_q) : (i+1)*(len(segs_rs)//num_q)]
    if not chunk: continue
    t_avg = (warp5(chunk[0]['t'][0]) + warp5(chunk[-1]['t'][-1])) / 2.0
    v_avg = np.mean(np.vstack([s['hist'] for s in chunk]), axis=0)[:5]
    rs_pts.append((t_avg, v_avg))
rs_pts.append((2.0, v_end_rs)) # End-of-RS Point

# PO Trends (Series)
po_pts = []
v_start_po = segs_po[0]['hist'][0, :5]
v_end_po   = segs_po[-1]['hist'][-1, :5]
po_pts.append((4.0, v_start_po)) # Start-of-PO Point
curr = 0
for count in [7, 4, 7, 4]: # Games per series
    series_games = po_info[curr : curr + count]
    if not series_games: break
    s_segs = segs_po[series_games[0]['start'] : series_games[-1]['end']]
    t_avg = (warp5(s_segs[0]['t'][0]) + warp5(s_segs[-1]['t'][-1])) / 2.0
    v_avg = np.mean(np.vstack([s['hist'] for s in s_segs]), axis=0)[:5]
    po_pts.append((t_avg, v_avg))
    curr += count
po_pts.append((6.0, v_end_po)) # End-of-PO Point

for mi, name in enumerate(METRICS):
    # RS Dash
    ax.plot([p[0] for p in rs_pts], [p[1][mi]*SCALE_FACTORS[mi] for p in rs_pts], color=COLS[name], ls='--', lw=2, alpha=0.7)
    # PO Dash
    ax.plot([p[0] for p in po_pts], [p[1][mi]*SCALE_FACTORS[mi] for p in po_pts], color=COLS[name], ls='--', lw=2, alpha=0.7)
    
    # Gap Recovery (Real Data)
    for seg in segs_gap:
        wt = warp5(np.asarray(seg['t']))
        ax.plot(wt, seg['hist'][:, mi]*SCALE_FACTORS[mi], color=COLS[name], lw=3)

# Markers
for pts, size in [(rs_pts, 60), (po_pts, 100)]:
    for p in pts:
        for mi, name in enumerate(METRICS):
            ax.scatter(p[0], p[1][mi]*SCALE_FACTORS[mi], color=COLS[name], s=size, edgecolors='white', zorder=10)

ax.set_xticks([0, 2, 3, 4, 6])
ax.set_xticklabels(['Start RS', 'End RS', 'Pre-PO Rest', 'Start PO', 'End PO'], color='#aaaacc')
ax.set_ylim(0, 1.1)
ax.set_ylabel('% of Metric Max', color='#aaaacc')
ax.set_title('GRAPH 5: Seasonal Trends (RS Quintiles & PO Series)', **title_kw)
add_legend(ax)

# ─── Save Fatigue Engine Mechanics ───────────────────────────────────────────
plt.tight_layout(pad=14.0)
out_path = os.path.join(graphs_dir, 'fatigue_engine_mechanics.png')
plt.savefig(out_path, dpi=300, facecolor=BG)
plt.close()
print(f"Saved 5-panel bio-fatigue visualization -> {out_path}")

# ─── New: Career TPI Profile ──────────────────────────────────────────────────
def plot_career_profile():
    df = pd.read_csv(results_path).dropna(subset=['Total_TPI'])
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)
    
    # Normalize for color mapping
    norm = plt.Normalize(df['Total_TPI'].min(), df['Total_TPI'].max())
    colors = plt.cm.plasma(norm(df['Total_TPI']))
    
    bars = ax.bar(df['Year'].astype(str), df['Total_TPI'], color=colors, alpha=0.6, edgecolor='white', linewidth=0.5, label='Total TPI')
    
    # Secondary Y-Axis for TPI per Game (Line)
    ax2 = ax.twinx()
    ax2.plot(df['Year'].astype(str), df['TPI_per_G'], color='#ffffff', marker='o', lw=2.5, markersize=8, label='TPI/G', zorder=10)
    ax2.set_ylabel('TPI per Game', color='#ffffff', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ffffff')
    
    # Highlights
    peak_idx = df['Total_TPI'].idxmax()
    peak_year = str(df.loc[peak_idx, 'Year'])
    ax.annotate('TOTAL TPI PEAK', xy=(peak_year, df.loc[peak_idx, 'Total_TPI']), 
                xytext=(0, 20), textcoords='offset points', ha='center',
                arrowprops=dict(arrowstyle='->', color='white'), color='white', fontweight='bold')

    ax.set_title('LeBron James: Career True Playoff Impact (TPI) Profile', fontsize=18, color='white', pad=20)
    ax.set_ylabel('Total TPI', color='#aaaacc', fontsize=12)
    ax.tick_params(colors='#aaaacc')
    ax.grid(axis='y', alpha=0.1, color='white')
    
    # Combined Legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left', framealpha=0.5, facecolor=BG)
    
    out_path = os.path.join(graphs_dir, 'lebron_career_tpi_profile.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved Career TPI Profile -> {out_path}")

# ─── New: Component Drivers ───────────────────────────────────────────────────
def plot_component_drivers():
    df = pd.read_csv(results_path).dropna(subset=['Total_TPI'])
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True, facecolor=BG)
    
    titles = ['Production Score', 'Resistance Score', 'Production Multiplier (Bio-Fatigue)']
    cols   = ['Prod_Score', 'Res_Score', 'Prod_Multiplier']
    colors = ['#00e5ff', '#ff9100', '#ff4d4d']
    
    for i, ax in enumerate(axes):
        ax.set_facecolor(BG)
        ax.plot(df['Year'].astype(str), df[cols[i]], color=colors[i], marker='o', lw=2, markersize=8)
        ax.set_title(titles[i], color='white', fontsize=14)
        ax.grid(alpha=0.1, color='white')
        ax.tick_params(colors='#aaaacc')
        
    plt.xticks(rotation=45)
    plt.suptitle('TPI Component Drivers: LeBron James (2006–2025)', color='white', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(graphs_dir, 'tpi_component_drivers.png')
    plt.savefig(out_path, dpi=300, facecolor=BG)
    plt.close()
    print(f"Saved TPI Component Drivers -> {out_path}")

if __name__ == "__main__":
    # Generate all plots
    plot_career_profile()
    plot_component_drivers()
    # (Fatigue Mechanics already generated by global script execution)
