# True Playoff Impact (TPI) Model

## Overview
True Playoff Impact (TPI) is an advanced, era-adjusted metric designed to rigorously quantify a player's true individual impact and "carry jobs" during the NBA Playoffs. Moving beyond traditional box score stats, TPI evaluates a player's performance contextually by incorporating efficiency, opponent strength, team capability, injury context, and cumulative physical fatigue.

The fundamental equation is:
**`Total TPI = (Production Score × Resistance Score × Average Fatigue) × Games Played`**

This model evaluates impact on a per-game basis (`TPI_per_G`) and a cumulative run basis (`Total_TPI`).

---

## The Core Components

### 1. Production Score
The Production Score (`Prod_Score`) evaluates a player's on-court output, heavily rewarding efficient scoring while relying on Box Plus/Minus (BPM) as the foundational metric.

*   **Base Metric:** Playoff Box Plus/Minus (BPM).
*   **Efficiency Adjustment:** A non-linear True Shooting percentage (TS%) multiplier is applied to reward high-efficiency scoring relative to the era's league average. 
*   **Formula:** `Adjusted BPM = BPM × (TS_Player / TS_League)^1.5`

### 2. Resistance Score
The Resistance Score (`Res_Score`) quantifies the difficulty of the playoff environment. It compares the "Final Boss" weighted strength of the opponents against the strength of the player's own supporting cast, adjusting for injuries and pace.

*   **Team Capability (SRS):** Evaluates team and opponent strength using a blend of Current Regular Season SRS (44.4%) and Previous Post-Season SRS (55.6%). Values are pace-adjusted to a 100-possession baseline.
*   **"Final Boss" Opponent Weighting:** per-series Resistance Scores are ranked from toughest to easiest. The toughest series receives the highest base weight (50%), ensuring deep runs against elite competition are properly rewarded.
*   **Injury Penalties:** Injuries are evaluated on a per-series basis. The injured player's regular-season BPM is converted into SRS equivalents and scaled by the proportion of games missed, dynamically lowering the team's strength.

### 3. Fatigue Engine (Micro-PBP Reconstruction)
The Fatigue Engine (`Fatigue_Avg`) models the compounding physical toll of a playoff run. Unlike standard models, TPI uses **minute-by-minute career play-by-play (PBP) data** to reconstruct exact exertion levels.

*   **Micro-Modeling:** A piecewise Ordinary Differential Equation (ODE) system simulates the fatigue path for every individual game, using exact ON/OFF substitution intervals.
*   **PBP USG%:** Usage rate is calculated programmatically for every quarter of every game (FGA + 0.44*FTA + TOV) to determine the exact intensity of every playing minute.
*   **On-Court Focus:** The final `Fatigue_Avg` is calculated strictly during **on-court minutes**, ensuring the "exertion credit" accurately reflects the performance context.
*   **Macro Recovery:** Models exponential decay of fatigue between games based on days of rest and cumulative career wear-and-tear.

---

## Career Benchmarks (LeBron James 2006-2025)

The TPI model identifies the 2018 run as the statistically greatest "carry job" in NBA history, driven by an unprecedented combination of volume, opponent resistance, and physical exhaustion.

| Year | Total TPI | TPI per G | Fatigue_Avg | Historical Context |
| :--- | :--- | :--- | :--- | :--- |
| **2018** | **280.5** | **12.75** | **0.603** | **The Apex Carry Job** |
| **2016** | 211.9 | 10.09 | 0.483 | 3-1 Comeback / Peak Efficiency |
| **2015** | 189.6 | 9.48 | 0.633 | Maximum Physical Exertion |
| **2014** | 168.4 | 8.86 | 0.505 | Elite Production (Final Miami Year) |
| **2012** | 139.5 | 6.06 | 0.579 | High Volume First Title |

---

## Technical Details

The model is built in Python, utilizing `pandas` for PBP data processing and `scipy.integrate.odeint` for continuous ODE fatigue modeling.

### Core Architecture
*   `/code/tpi_calculations/tpi_v2.py`: The core calculation engine.
*   `/code/tpi_calculations/tpi_career_fatigue.py`: Minute-by-minute career PBP fatigue simulation.
*   `/data/fatigue_metric/`: Raw PBP components and minute-interval datasets.
*   `/data/tpi_results/`: Final computed TPI datasets.