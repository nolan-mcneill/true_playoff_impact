import pandas as pd
import numpy as np
from scipy.integrate import odeint, trapezoid

# --- 1. INJURY PENALTY LOGIC ---
def get_total_playoff_games(schedule_str):
    if pd.isna(schedule_str): return 0
    return len(str(schedule_str).split(','))

# Series weights by number of rounds played and current round number.
# Higher rounds (Finals=4) are weighted more heavily.
SERIES_WEIGHTS = {
    4: {4: 0.50, 3: 0.25, 2: 0.15, 1: 0.10},
    3: {3: 0.50, 2: 0.30, 1: 0.20},
    2: {2: 0.60, 1: 0.40},
    1: {1: 1.00},
}

def _series_penalty(inj, num_series):
    """Compute a single injury's weighted penalty.
    Formula: BPM × (games_missed / series_games) × series_weight
    games_missed is derived as Series_Games - Games_Played.
    Returns 0 if no games were missed or data is invalid.
    """
    if pd.isna(inj['BPM']) or pd.isna(inj['Games_Played']) or pd.isna(inj['Series_Games']):
        return 0
    series_games = float(inj['Series_Games'])
    if series_games == 0:
        return 0
    games_missed = series_games - float(inj['Games_Played'])
    if games_missed <= 0:
        return 0  # Player was fully healthy in this series
    series_num = int(inj['Series'])
    weight = SERIES_WEIGHTS.get(num_series, {}).get(series_num, 0)
    if weight == 0 and num_series > 0:
        weight = 1.0 / num_series  # Fallback for data/round alignment
    return float(inj['BPM']) * (games_missed / series_games) * weight

def calculate_injury_penalties(df_main, df_teammate_inj, df_opp_inj):
    opp_penalties = {}
    help_penalties = {}

    for _, row in df_main.iterrows():
        year = row['Year']
        total_games = get_total_playoff_games(row['Schedule_Days'])

        if total_games >= 16: num_series = 4
        elif total_games >= 12: num_series = 3
        elif total_games >= 8: num_series = 2
        elif total_games > 0: num_series = 1
        else: num_series = 0

        # Opponent penalty — series-weighted (50/25/15/10)
        year_opp = df_opp_inj[df_opp_inj['Year'] == year]
        opp_penalties[year] = sum(
            _series_penalty(inj, num_series) for _, inj in year_opp.iterrows()
        )

        # Teammate (help) penalty — now uses identical series-weighted logic
        year_help = df_teammate_inj[df_teammate_inj['Year'] == year]
        help_penalties[year] = sum(
            _series_penalty(inj, num_series) for _, inj in year_help.iterrows()
        )

    return opp_penalties, help_penalties

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
    current = c_srs
    legacy = p_srs
    return (current * 0.444) + (legacy * 0.556)

# BPM_SCALE: converts player-level BPM (pts/100 above replacement) to
# team-level SRS-equivalent impact. ~0.45 per Engelmann et al.
BPM_SCALE = 0.45

def calculate_resistance_final(row, rs_pace_lkp, po_pace_lkp, team_srs_df, opp_srs_df, opp_penalty=0, help_penalty=0):
    year = int(row['Year'])
    # Pace factors: normalize each SRS value to a 100-possession baseline.
    rs_f  = 100.0 / rs_pace_lkp.get(year,     100.0)
    po_f1 = 100.0 / po_pace_lkp.get(year - 1, 100.0)

    # Calculate base_help using LeBron's team data
    team_data = team_srs_df[team_srs_df['Year'] == year]
    if not team_data.empty:
        t_row = team_data.iloc[0]
        base_help = calculate_team_capability(
            float(t_row['Reg Season SRS']) * rs_f,
            float(t_row['Prev Post Season SRS']) * po_f1
        )
    else:
        base_help = 0.0

    # Calculate base_opp dynamically from opponent data
    opp_data = opp_srs_df[opp_srs_df['Year'] == year]
    if not opp_data.empty:
        opp_scores = []
        for _, o_row in opp_data.iterrows():
            score = calculate_team_capability(
                float(o_row['Reg Season SRS']) * rs_f,
                float(o_row['Prev Post Season SRS']) * po_f1
            )
            opp_scores.append({'score': score, 'games': float(o_row['Games'])})
        
        # Rank opponents by score descending
        opp_scores.sort(key=lambda x: x['score'], reverse=True)
        
        base_weights = [0.50, 0.25, 0.15, 0.10]
        num_opps = len(opp_scores)
        applied_weights = base_weights[:num_opps]
        weight_sum = sum(applied_weights)
        normalized_weights = [w / weight_sum for w in applied_weights]
        
        numerator = sum(opp['score'] * normalized_weights[i] * opp['games'] for i, opp in enumerate(opp_scores))
        denominator = sum(normalized_weights[i] * opp['games'] for i, opp in enumerate(opp_scores))
        
        base_opp = numerator / denominator if denominator > 0 else 0.0
    else:
        base_opp = 0.0

    # Scale BPM-derived penalties into pace-adjusted SRS units
    g = (base_opp - opp_penalty * BPM_SCALE) - (base_help - help_penalty * BPM_SCALE)
    k = 0.35  # Power factor for resistance decay/growth

    if g >= 0:
        res = pow(g + 1, k)
    else:
        res = 1 / pow(abs(g) + 1, k)
    return max(0.0001, res)

# --- 4. FATIGUE ENGINE (ODE) ---
R_BASE, ETA, LMBDA, BETA, ZETA, PHI = 0.33, 0.15, 0.12, 0.02, 0.05, 1.05

def fatigue_derivative(Af, t, r_eff):
    recovery_rate = r_eff * (1 + ETA * Af) * np.exp(-LMBDA * Af)
    return -Af * recovery_rate

def get_avg_fatigue(row):
    days = [float(x) for x in str(row['Schedule_Days']).split(',')]
    shifts = [float(x) for x in str(row['TZ_Shifts']).split(',')]
    current_af = (pow(row['RS_MPG'] / 36, 1.4) * (row['RS_USG'] / 25)) * (row['Games_Played'] / 82)
    total_area = 0

    for i in range(len(days)):
        work_base = pow(row['MPG'] / 36, 1.4) * (row['USG'] / 25)
        impulse = (work_base * PHI) * np.exp(BETA * current_af)
        current_af += impulse
        
        if i < len(days) - 1:
            t_span = np.linspace(days[i], days[i+1], 50)
            r_eff = R_BASE * pow(1 - ZETA, abs(shifts[i]))
            decay_path = odeint(fatigue_derivative, current_af, t_span, args=(r_eff,))
            total_area += trapezoid(decay_path.flatten(), t_span)
            current_af = decay_path[-1][0]
            
    return total_area / len(days)

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

    opp_pens, help_pens = calculate_injury_penalties(df, df_teammate_inj, df_opp_inj)
    results = []

    for _, row in df.iterrows():
        if pd.isna(row['BPM']):
            results.append([np.nan]*6)
            continue

        prod = calculate_adj_bpm(row['BPM'], row['TS_Player'], row['TS_League'])
        res = calculate_resistance_final(
            row, rs_pace_lkp, po_pace_lkp, df_team_srs, df_opp_srs,
            opp_pens.get(row['Year'], 0), help_pens.get(row['Year'], 0)
        )
        fatigue_avg = get_avg_fatigue(row)
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