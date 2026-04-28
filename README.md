# True Playoff Impact (TPI) Model

## Overview
True Playoff Impact (TPI) is an advanced, era-adjusted metric designed to rigorously quantify a player's true individual impact and "carry jobs" during the NBA Playoffs. Moving beyond traditional box score stats, TPI evaluates a player's performance contextually by incorporating efficiency, opponent strength, team capability, injury context, and cumulative physical fatigue.

The fundamental equation is:
**`Total TPI = (Production Score × Resistance Score × Average Fatigue) × Games Played`**

---

## The Core Components

### 1. Production Score
The Production Score (`Prod_Score`) evaluates a player's on-court output, heavily rewarding efficient scoring while relying on Box Plus/Minus (BPM) as the foundational metric.
*   **Efficiency Adjustment:** A non-linear True Shooting percentage (TS%) multiplier is applied (`^1.5`) to reward high-efficiency scoring relative to the era's league average.

### 2. Resistance Score
The Resistance Score (`Res_Score`) quantifies the difficulty of the playoff environment. It compares the "Final Boss" weighted strength of the opponents against the strength of the player's own supporting cast.
*   **Team Capability (SRS):** A blend of Current Regular Season SRS (44%) and Previous Post-Season SRS (56%), pace-adjusted to 100 possessions.
*   **"Final Boss" Weighting:** Series are weighted (50%/25%/15%/10%) based on difficulty, ensuring deep runs against elite competition are prioritized.
*   **Injury Penalties:** Supporting cast and opponent injuries are evaluated on a series-by-series basis using missing-BPM equivalents.

### 3. High-Fidelity Fatigue Engine (3-Scale ODE)
The TPI Fatigue Engine is a biologically-grounded simulation that models metabolic strain and recovery using exact play-by-play (PBP) substitution patterns.

#### The Triple-Scale Recovery Model:
*   **Scale 1: Adrenaline-Inhibited Play (`BETA_PLAYING`):** While on-court, systemic recovery is inhibited by high catecholamine levels and muscle pressure. Recovery is near-zero, ensuring fatigue strictly accumulates during play.
*   **Scale 2: Metabolic Clearing (`BETA_TIMEOUT`):** During timeouts and bench intervals, the body rapidly clears lactate and restores Phosphocreatine (PCr) on a **minutes-scale**.
*   **Scale 3: Systemic Remodeling (`BETA_REST_DAY`):** Between games, the body performs tissue repair, neural restoration, and glycogen replenishment on a **days-scale**.

#### Temporal Accuracy:
*   **Actual-Date Integration:** The model uses the exact historical calendar dates for both the Regular Season and Postseason, correctly capturing the erratic rest intervals (back-to-backs, travel days) that drive cumulative wear-and-tear.

---

## Visualization Suite: The Snake Zoom
To represent the vast difference between 40 minutes of high-intensity play and 3 days of rest, TPI uses a **"Snake Zoom"** visualization architecture.

*   **Non-Linear Time Magnification:** Game segments are magnified by **30x** on the X-axis while rest days remain at 1x. This allows metabolic "zig-zags" (exertion vs. timeout recovery) to be visible alongside long-term decay.
*   **5-Level Fidelity:**
    *   **Level 1 (Intra-Game):** 1-minute resolution ODE view of single games.
    *   **Level 2 (Series):** Snake-zoomed view of a single series.
    *   **Level 3 (Postseason):** Snake-zoomed view of the full playoff run.
    *   **Level 4 (Regular Season):** High-level view of the 82-game campaign using actual dates.
    *   **Level 5 (Overall Season):** A 15x-magnified bridge showing the transition from regular season baseline to playoff apex.

---

## Career Benchmarks (LeBron James 2006-2025)

The TPI model identifies the 2018 run as the statistically greatest "carry job" in NBA history.

| Year | Total TPI | TPI per G | Fatigue_Avg | Historical Context |
| :--- | :--- | :--- | :--- | :--- |
| **2018** | **280.5** | **12.75** | **0.603** | **The Apex Carry Job** |
| **2016** | 211.9 | 10.09 | 0.483 | 3-1 Comeback / Peak Efficiency |
| **2015** | 189.6 | 9.48 | 0.633 | Maximum Physical Exertion |

---

## Technical Architecture
*   `code/tpi_calculations/tpi_career_fatigue.py`: Core simulation engine (ODE + Temporal logic).
*   `code/tpi_calculations/plot_tpi.py`: The 5-level high-fidelity visualization suite.
*   `data/tpi_results/lebron_tpi_results.csv`: Final computed TPI datasets.