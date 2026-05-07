import numpy as np

# ─── Bio-ODE model constants ──────────────────────────────────────────────────
# Recovery rates (per minute)
k_rP = 0.40      # PCr recovery (faster half-life ~60s)
k_rL = 0.05      # Lactate removal (half-life ~15-20 mins)
k_rG = 0.002     # Glycogen resynthesis
p_rM = 0.0001    # Muscle repair (very slow, days)
k_rC = 0.00005   # CNS recovery (extremely slow)

# Decay rates (per minute per intensity unit)
k_dP = 1.5       # PCr depletion (fast)
k_dL = 0.20      # Lactate accumulation
k_dG = 0.005     # Glycogen depletion
k_dM = 0.04      # Muscle damage (Adjusted for better visibility in macro view)
k_dC = 0.008     # CNS fatigue (Adjusted for better visibility in macro view)

# Other model constants
k_EPOC = 0.001   # Glycogen depletion during EPOC
b_P = 2.5        # PCr inhibition constant
b_L = 1.5        # (Reserved)
lam = 1.0        # (Reserved)

def bio_model_ode(state, t, I_total, I_col):
    """Core ODE for on-court play."""
    PCr, Gly, Lac, M, CNS, Phi = state
    # EPOC-like multiplier (Phi)
    dPhi = (0.15 * I_total) - (0.05 * Phi)
    
    # Differential equations
    dPCr = k_rP * (1.0 - PCr) * np.exp(-b_P * I_total) - k_dP * (I_total**2) * PCr
    dGly = k_rG * (1.0 - Gly) - (k_dG * (I_total**1.5) * Gly) - (k_EPOC * Phi * Gly)
    dLac = k_rL * (1.0 - Lac) - (k_dL * (I_total**2) * Lac)
    dM   = p_rM * (1.0 - M) - (k_dM * (I_col**2) * M)
    dCNS = k_rC * (1.0 - CNS) * np.sqrt(max(1e-3, M)) - (k_dC * I_total * (1.5 - Lac) * CNS)
    
    return [dPCr, dGly, dLac, dM, dCNS, dPhi]

def bio_model_off(state, t):
    """ODE for bench/timeout periods (short rest)."""
    PCr, Gly, Lac, M, CNS, Phi = state
    dPhi = -0.01 * Phi
    dPCr = k_rP * (1.0 - PCr)
    dGly = k_rG * (1.0 - Gly) - (k_EPOC * Phi * Gly)
    dLac = k_rL * (1.0 - Lac)
    dM   = p_rM * (1.0 - M)
    dCNS = k_rC * (1.0 - CNS) * np.sqrt(max(1e-3, M))
    return [dPCr, dGly, dLac, dM, dCNS, dPhi]

def bio_model_rest(state, t):
    """ODE for long rest (days/hours)."""
    PCr, Gly, Lac, M, CNS, Phi = state
    dPhi = -0.05 * Phi
    dPCr = k_rP * (1.0 - PCr)
    dGly = k_rG * (1.0 - Gly) - (k_EPOC * Phi * Gly)
    dLac = k_rL * (1.0 - Lac)
    dM   = p_rM * (1.0 - M)
    dCNS = k_rC * (1.0 - CNS) * np.sqrt(max(1e-3, M))
    return [dPCr, dGly, dLac, dM, dCNS, dPhi]

def get_P(state):
    """Calculate the production multiplier based on current bio-state."""
    PCr, Gly, Lac, M, CNS, Phi = state
    # Weights optimized for playoff impact
    return (max(0, PCr)**0.5) * (max(0, Lac)**0.4) * (max(0, Gly)**0.3) * (max(0, M)**0.4) * (max(0, CNS)**0.8)
