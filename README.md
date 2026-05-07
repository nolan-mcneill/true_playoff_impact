# True Playoff Impact (TPI) Model

## Overview

True Playoff Impact (TPI) is an advanced, era-adjusted metric designed to rigorously quantify a player's true individual impact during the NBA Playoffs. Moving beyond traditional box score stats, TPI evaluates performance contextually by incorporating efficiency, opponent strength, team capability, injury context, and a physiologically rigorous biological fatigue simulation.

The fundamental equation is:

> **`Total TPI = (Production Score × Resistance Score × Production Multiplier) × Games Played`**

Where `Production Multiplier` is the output of a 6-state biological ODE system that simulates the real cumulative metabolic cost of a playoff run, computed from exact play-by-play data.

---

## The Three Core Components

### 1. Production Score (`Prod_Score`)

Evaluates raw on-court output using BPM as the base, adjusted for shooting efficiency relative to the era's league average.

- **Base Metric:** Box Plus/Minus (BPM) from Basketball Reference.
- **Efficiency Multiplier:** A non-linear TS% correction `(ts_player / ts_league)^1.5` is applied.

### 2. Resistance Score (`Res_Score`)

Quantifies the difficulty of the playoff environment by measuring the gap between opponent quality and teammate support.

- **Team Capability (SRS-based):** Blends Current Regular Season SRS (44.4%) with Previous Postseason SRS (55.6%).
- **"Final Boss" Series Weighting:** Series are weighted 50% / 25% / 15% / 10% (hardest to easiest) to prioritize elite opponents.
- **Injury Context:** Missing-BPM penalties are applied for both teammate and opponent injuries.
- **Resistance Formula:** `gap = adj_opp_srs - adj_help_srs`. Resistance is calculated as `(gap+1)^0.35` for positive gaps.

### 3. Bio-Fatigue Engine: 6-State ODE System (`Production Multiplier`)

The TPI Fatigue Engine is a biologically-grounded differential equation system that models six distinct physiological state variables.

#### The Six State Variables

| Variable | Biological Meaning | Range |
| :--- | :--- | :--- |
| **PCr** | Phosphocreatine (Fast-resynthesis capacity) | 0–1 |
| **Gly** | Glycogen stores (Aerobic fuel reserve) | 0–1 |
| **Lac** | Lactate clearance (Normalized efficiency) | 0–1 |
| **M** | Muscle integrity (Contractile capacity) | 0–1 |
| **CNS** | Central Nervous System readiness | 0–1 |
| **Phi (Φ)** | EPOC debt accumulator | 0–∞ |

#### The Production Multiplier (P)

The output scalar `P` combines all metrics:
```
P = PCr^0.5 × Lac^0.4 × Gly^0.3 × M^0.4 × CNS^0.8
```

#### Centralized Model (`bio_model.py`)

The model is centralized in a shared module to ensure consistency across all simulations and visualizations. It includes:
- **Active Play**: Intensity is calculated as `(Movement * 0.5) + (Contact * 0.5)`.
- **Bench/Timeout Rest**: Rapid partial recovery.
- **Inter-Game Recovery**: Long-term metabolic and neural repair using consistent minute-based integration.

---

## Data Pipeline

1. **PBP Reconstruction** (`data_reconstruction.py`): Downloads and parses raw PBP data to extract usage and intensity components.
2. **Fatigue Engine** (`fatigue_engine.py`): Runs the full bio-ODE simulation from Regular Season through Postseason.
3. **TPI Main** (`tpi_main.py`): Combines Production, Resistance, and Fatigue into the final TPI scores.
4. **Visualization** (`visualization.py`): Generates high-fidelity 5-panel charts of the fatigue dynamics.

---

## Visualization Suite: The Multi-Scale Fatigue View

The TPI visualization engine produces a 5-panel dashboard analyzing fatigue at every level of resolution:

- **Graph 1: Intra-Game Dynamics**: Minute-by-minute metabolic shifts within a single game (e.g., 2018 Finals G1).
- **Graph 2: Recovery Windows**: Hourly recovery curves between sequential games.
- **Graph 3: Series Progression**: Game-to-game fatigue trends within a specific series.
- **Graph 4: Playoff Gauntlet**: Series-by-series trends across the entire postseason, emphasizing the cumulative cost of deep runs.
- **Graph 5: Seasonal Trends**: The full journey from Regular Season quintiles to the Postseason peak.

---

## Career Results (LeBron James 2006–2025)

| Year | Total TPI | TPI per G | Prod_Score | Res_Score | Prod_Multiplier | GP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **2016** | **154.28** | 7.35 | 14.85 | 1.431 | 0.346 | 21 |
| **2018** | 116.13 | 5.28 | 15.08 | 1.674 | 0.209 | 22 |
| **2014** | 115.15 | 6.06 | 15.77 | 1.112 | 0.345 | 19 |
| **2017** | 90.94 | 5.35 | 14.70 | 1.235 | 0.295 | 17 |
| **2012** | 85.12 | 3.70 | 12.71 | 0.842 | 0.346 | 23 |
| **2023** | 70.73 | 4.16 | 6.65 | 1.540 | 0.406 | 17 |
| **2013** | 69.17 | 3.01 | 10.90 | 0.757 | 0.364 | 23 |
| **2015** | 68.32 | 3.42 | 8.92 | 1.321 | 0.290 | 20 |
| **2006** | 62.11 | 4.78 | 9.88 | 1.357 | 0.356 | 13 |
| **2020** | 60.52 | 2.88 | 14.09 | 0.564 | 0.362 | 21 |

---

## Technical Architecture

```
true_playoff_impact/
├── code/
│   └── tpi_calculations/
│       ├── bio_model.py                # Centralized bio-ODE logic & constants
│       ├── pbp_data_reconstruction.py  # PBP downloader & parser
│       ├── tpi_career_fatigue.py       # Full-career simulation engine
│       ├── tpi_v2.py                   # Final TPI assembly
│       └── plot_tpi.py                 # Multi-panel visualization suite
├── data/
│   ├── fatigue_metric/                 # Simulation inputs and intermediate outputs
│   ├── prod_score/                     # Player efficiency and BPM data
│   ├── res_score/                      # Team/Opponent SRS and Injury data
│   └── tpi_results/                    # Final CSV results and PNG graphs
└── README.md
```