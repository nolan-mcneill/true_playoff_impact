import pandas as pd
import numpy as np
from scipy.integrate import odeint, trapezoid

# --- 1. UTILS ---
def get_total_playoff_games(schedule_str):
    if pd.isna(schedule_str): return 0
    return len(str(schedule_str).split(','))

# --- 2. PRODUCTION LOGIC ---
# k=1.5: nonlinear TS% correction — rewards/penalizes efficiency increasingly
# as ts_ratio moves further from 1.0 (league average). Volume is intentionally
# excluded here since BPM already captures it.
TS_EXP = 1.5

def calculate_adj_bpm(bpm, ts_player, ts_league):
    ts_ratio = ts_player / ts_league
    return bpm * pow(ts_ratio, TS_EXP)

# --- 3. TEAM CAPABILITY & RESISTANCE ---
# SRS only: schedule-adjusted, single consistent unit (pts/game margin).
# Net Rating dropped to avoid mixing per-100-possession and per-game units.
def calculate_team_capability(c_srs, p_srs):
    if p_srs == 0:
        return c_srs
    current = c_srs
    legacy = p_srs
    return (current * 0.444) + (legacy * 0.556)

# BPM_SCALE: converts player-level BPM (pts/100 above replacement) to
# team-level SRS-equivalent impact. ~0.45 per Engelmann et al.
BPM_SCALE = 0.45

def calculate_resistance_final(row, rs_pace_lkp, po_pace_lkp, team_srs_df, opp_srs_df, df_teammate_inj, df_opp_inj):
    year = int(row['Year'])
    # Pace factors: normalize each SRS value to a 100-possession baseline.
    rs_f  = 100.0 / rs_pace_lkp.get(year,     100.0)
    po_f1 = 100.0 / po_pace_lkp.get(year - 1, 100.0)

    # Base Help SRS for the year
    team_data = team_srs_df[team_srs_df['Year'] == year]
    if not team_data.empty:
        t_row = team_data.iloc[0]
        raw_help_srs = calculate_team_capability(
            float(t_row['Reg Season SRS']) * rs_f,
            float(t_row['Prev Post Season SRS']) * po_f1
        )
    else:
        raw_help_srs = 0.0

    opp_data = opp_srs_df[opp_srs_df['Year'] == year]
    series_res_scores = []
    k = 0.35  # Power factor for resistance decay/growth

    if not opp_data.empty:
        for _, o_row in opp_data.iterrows():
            series_num = int(o_row['Round'])
            series_games = float(o_row['Games'])
            if series_games == 0: continue
            
            raw_opp_srs = calculate_team_capability(
                float(o_row['Reg Season SRS']) * rs_f,
                float(o_row['Prev Post Season SRS']) * po_f1
            )
            
            # Opponent injury penalty for this series
            opp_inj_series = df_opp_inj[(df_opp_inj['Year'] == year) & (df_opp_inj['Series'] == series_num)]
            opp_penalty_bpm = 0.0
            for _, inj in opp_inj_series.iterrows():
                if pd.isna(inj['BPM']) or pd.isna(inj['Games_Played']) or pd.isna(inj['Series_Games']): continue
                s_games = float(inj['Series_Games'])
                if s_games == 0: continue
                games_missed = s_games - float(inj['Games_Played'])
                if games_missed > 0:
                    opp_penalty_bpm += float(inj['BPM']) * (games_missed / s_games)
            
            adj_opp_srs = raw_opp_srs - (opp_penalty_bpm * BPM_SCALE)
            
            # Teammate injury penalty for this series
            help_inj_series = df_teammate_inj[(df_teammate_inj['Year'] == year) & (df_teammate_inj['Series'] == series_num)]
            help_penalty_bpm = 0.0
            for _, inj in help_inj_series.iterrows():
                if pd.isna(inj['BPM']) or pd.isna(inj['Games_Played']) or pd.isna(inj['Series_Games']): continue
                s_games = float(inj['Series_Games'])
                if s_games == 0: continue
                games_missed = s_games - float(inj['Games_Played'])
                if games_missed > 0:
                    help_penalty_bpm += float(inj['BPM']) * (games_missed / s_games)
                    
            adj_help_srs = raw_help_srs - (help_penalty_bpm * BPM_SCALE)
            
            # Calculate gap and resistance for THIS series
            g = adj_opp_srs - adj_help_srs
            if g >= 0:
                res = pow(g + 1, k)
            else:
                res = 1 / pow(abs(g) + 1, k)
                
            series_res_scores.append({'res': res, 'games': series_games})
            
        if not series_res_scores:
            return 0.0001
            
        # Sort series by resistance score descending
        series_res_scores.sort(key=lambda x: x['res'], reverse=True)
        
        base_weights = [0.50, 0.25, 0.15, 0.10]
        num_series = len(series_res_scores)
        applied_weights = base_weights[:num_series]
        weight_sum = sum(applied_weights)
        normalized_weights = [w / weight_sum for w in applied_weights]
        
        numerator = sum(s['res'] * normalized_weights[i] * s['games'] for i, s in enumerate(series_res_scores))
        denominator = sum(normalized_weights[i] * s['games'] for i, s in enumerate(series_res_scores))
        
        final_res = numerator / denominator if denominator > 0 else 0.0001
        return max(0.0001, final_res)
    else:
        return 0.0001

# --- 4. FATIGUE ENGINE (ODE) ---




# --- 5. MAIN EXECUTION ---
def run_tpi_analysis(df_path, teammate_inj_path, opp_inj_path, team_srs_path, opp_srs_path):
    df = pd.read_csv(df_path)
    df_teammate_inj = pd.read_csv(teammate_inj_path)
    df_opp_inj = pd.read_csv(opp_inj_path)
    df_team_srs = pd.read_csv(team_srs_path)
    df_opp_srs = pd.read_csv(opp_srs_path)

    # Build pace lookup dicts from CSV (keyed by integer year)
    rs_pace_lkp = dict(zip(df['Year'].astype(int), df['RS_Pace']))
    po_pace_lkp = dict(zip(df['Year'].astype(int), df['PO_Pace']))

    # Load the new micro-fatigue results
    micro_fatigue_path = os.path.join(base_dir, 'data', 'fatigue_metric', 'lebron_career_fatigue_results.csv')
    df_fatigue = pd.read_csv(micro_fatigue_path)
    fatigue_lkp = dict(zip(df_fatigue['Year'].astype(int), df_fatigue['Fatigue_Avg']))

    results = []

    for _, row in df.iterrows():
        if pd.isna(row['BPM']):
            results.append([np.nan]*6)
            continue

        prod = calculate_adj_bpm(row['BPM'], row['TS_Player'], row['TS_League'])
        res = calculate_resistance_final(
            row, rs_pace_lkp, po_pace_lkp, df_team_srs, df_opp_srs,
            df_teammate_inj, df_opp_inj
        )
        
        year = int(row['Year'])
        # Use micro-fatigue value from our PBP reconstruction
        fatigue_avg = fatigue_lkp.get(year, 0.5) # Default 0.5 if missing
        
        gp = get_total_playoff_games(row['Schedule_Days'])
        
        tpi_per_g = (prod * res) * fatigue_avg
        total_tpi = tpi_per_g * gp
        
        results.append([total_tpi, tpi_per_g, prod, res, fatigue_avg, gp])
        
    cols = ['Total_TPI', 'TPI_per_G', 'Prod_Score', 'Res_Score', 'Fatigue_Avg', 'GP']
    res_df = pd.DataFrame(results, columns=cols)
    final = pd.concat([df['Year'], res_df], axis=1)
    
    # Sort for career view
    return final.sort_values('Year')

# --- RUN AND PRINT ---
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

final_output = run_tpi_analysis(
    os.path.join(base_dir, 'data', 'prod_score', 'lebron_data', 'lebron_tpi_metrics_final.csv'),
    os.path.join(base_dir, 'data', 'res_score', 'playoff_injury_data', 'lebron_teammate_injury_data.csv'),
    os.path.join(base_dir, 'data', 'res_score', 'playoff_injury_data', 'lebron_opponent_injury_data.csv'),
    os.path.join(base_dir, 'data', 'res_score', 'team_capability_data', 'lebron_team_srs_data.csv'),
    os.path.join(base_dir, 'data', 'res_score', 'team_capability_data', 'lebron_opponent_srs_data.csv'),
)

results_dir = os.path.join(base_dir, 'data', 'tpi_results')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, 'lebron_tpi_results.csv')
output_cols = ['Year', 'Total_TPI', 'TPI_per_G', 'Prod_Score', 'Res_Score', 'Fatigue_Avg', 'GP']
final_output[output_cols].to_csv(results_path, index=False)

print(f"\n--- LEBRON JAMES TPI CAREER ANALYSIS SAVED TO {results_path} ---")
print(final_output[output_cols].to_string(index=False))