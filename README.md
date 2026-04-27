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
*   **"Final Boss" Opponent Weighting:** Instead of a simple average, the Resistance Score is calculated natively for each individual series (incorporating pace, team SRS, and injuries for that specific series). These per-series Resistance Scores are then ranked from toughest to easiest. The toughest series (the "Final Boss") receives the highest base weight (e.g., 50%), the second toughest 25%, the third 15%, and the easiest 10%. This ensures beating a juggernaut in the 2nd round is properly rewarded.
*   **Injury Penalties:** Injuries are evaluated on a per-series basis. The injured player's regular-season BPM is converted into SRS equivalents and scaled by the proportion of games missed in that specific series. This penalty directly lowers the opponent's (or LeBron's team's) SRS for that series *before* the Resistance Score is calculated, dynamically adjusting the "moment-to-moment" strength of the team.
*   **Resistance Calculation:** For each series, a Resistance Gap (`g`) is calculated: `Adjusted Opponent SRS - Adjusted Team Help SRS`. A power curve (`k = 0.35`) is applied to this gap. The final overall Resistance Score is the weighted average of these per-series Resistance Scores, with weights scaled by the number of games played in each series.

### 3. Fatigue Engine
The Fatigue Engine (`Fatigue_Avg`) models the compounding physical toll of a playoff run. A deep playoff run as a high-usage player requires significantly more exertion than a short stint.

*   **Workload:** Calculated per-game using Minutes Per Game (MPG) and Usage Rate (USG%), scaled non-linearly.
*   **Rest & Recovery:** Models the days of rest between games using a continuous Ordinary Differential Equation (ODE).
*   **Timezone Tax:** Penalizes recovery based on the number of timezone shifts between games.
*   **Mechanism:** Fatigue accumulates as "impulses" after each game and decays exponentially during rest days. The engine calculates the integral (area under the curve) of this continuous fatigue path to determine the `Fatigue_Avg`.

---

## Technical Details

The model is built in Python, heavily relying on `pandas` for data structuring and `scipy.integrate.odeint` for the continuous fatigue engine modeling.

### Core Architecture
*   `/code/tpi_calculations/tpi_v2.py`: The core calculation engine housing the TPI formulas.
*   `/data/prod_score/`: Player production data (BPM, TS%, usage).
*   `/data/res_score/`: Environmental data (Team SRS, opponent SRS, schedule/rest days, timezone shifts, injury tracking).
*   `/data/tpi_results/`: Output CSVs containing the final computed Total TPI and Per-Game TPI across different years.