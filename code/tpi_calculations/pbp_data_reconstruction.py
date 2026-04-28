import os
import pandas as pd
import subprocess
import time

# LeBron James Player ID
LEBRON_ID = 2544

def download_and_extract(year):
    filename = f"nbastats_po_{year}.tar.xz"
    csv_filename = f"nbastats_po_{year}.csv"
    url = f"https://github.com/shufinskiy/nba_data/raw/main/datasets/{filename}"
    
    if not os.path.exists(csv_filename):
        try:
            subprocess.run(["curl.exe", "-L", url, "-o", filename], check=True)
            subprocess.run(["tar.exe", "-xf", filename], check=True)
            os.remove(filename)
        except Exception as e:
            return None
    return csv_filename

def to_seconds(clock_str):
    try:
        m, s = map(int, clock_str.split(':'))
        return m * 60 + s
    except:
        return 0

def process_year_pbp(csv_file, year):
    if not os.path.exists(csv_file):
        return []
    
    df = pd.read_csv(csv_file, low_memory=False)
    lebron_game_ids = df[df['PLAYER1_ID'] == LEBRON_ID]['GAME_ID'].unique()
    results = []
    
    for game_id in lebron_game_ids:
        g_df = df[df['GAME_ID'] == game_id]
        l_events = g_df[g_df['PLAYER1_ID'] == LEBRON_ID]
        if l_events.empty: continue
        team_id = l_events.iloc[0]['PLAYER1_TEAM_ID']
        
        for period in sorted(g_df['PERIOD'].unique()):
            p_df = g_df[g_df['PERIOD'] == period]
            l_fga = len(p_df[(p_df['PLAYER1_ID'] == LEBRON_ID) & (p_df['EVENTMSGTYPE'].isin([1, 2]))])
            l_fta = len(p_df[(p_df['PLAYER1_ID'] == LEBRON_ID) & (p_df['EVENTMSGTYPE'] == 3)])
            l_tov = len(p_df[(p_df['PLAYER1_ID'] == LEBRON_ID) & (p_df['EVENTMSGTYPE'] == 5)])
            
            t_fga = len(p_df[(p_df['PLAYER1_TEAM_ID'] == team_id) & (p_df['EVENTMSGTYPE'].isin([1, 2]))])
            t_fta = len(p_df[(p_df['PLAYER1_TEAM_ID'] == team_id) & (p_df['EVENTMSGTYPE'] == 3)])
            t_tov = len(p_df[(p_df['PLAYER1_TEAM_ID'] == team_id) & (p_df['EVENTMSGTYPE'] == 5)])
            
            timeouts = p_df[p_df['EVENTMSGTYPE'] == 9]['PCTIMESTRING'].apply(to_seconds).unique().tolist()
            q_length = 720 if period <= 4 else 300
            timeouts_elapsed = [q_length - t for t in timeouts]
            
            results.append({
                'Year': year + 1,
                'Game_ID': game_id,
                'Period': period,
                'L_FGA': l_fga, 'L_FTA': l_fta, 'L_TOV': l_tov,
                'T_FGA': t_fga, 'T_FTA': t_fta, 'T_TOV': t_tov,
                'Timeouts': ",".join(map(str, timeouts_elapsed))
            })
    return results

def main():
    years = list(range(2005, 2025))
    all_results = []
    output_csv = "data/fatigue_metric/lebron_pbp_usg_components.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    for year in years:
        csv_file = download_and_extract(year)
        if csv_file:
            all_results.extend(process_year_pbp(csv_file, year))
            if os.path.exists(csv_file): os.remove(csv_file)
        
        if all_results:
            pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"PBP Reconstruction Complete: {output_csv}")

if __name__ == "__main__":
    main()
