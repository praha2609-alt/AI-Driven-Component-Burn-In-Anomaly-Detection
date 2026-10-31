"""
The same simulation as burnin_simulation.py, written as a function so the
dashboard (app.py) can re-run it with different settings.
"""
import numpy as np
import pandas as pd

HOURS = np.arange(0, 169, 4)   # reading every 4 h for 168 h


def simulate(n=1000, defect_rate=0.05, noise=0.02, seed=42,
             weibull_scale=60.0, weibull_shape=0.5, hours=HOURS,
             esr0_mean=0.10, c0_mean=470.0, leak0_mean=20.0, spread=0.07,
             k_healthy=1e-4, a_healthy=5e-5):
    """Return a long table: one row per capacitor per reading."""
    rng = np.random.default_rng(seed)

    # Healthy behaviour for every unit
    esr0 = rng.normal(esr0_mean, spread * esr0_mean, n)
    c0 = rng.normal(c0_mean, spread * c0_mean, n)
    leak0 = rng.normal(leak0_mean, spread * leak0_mean, n)
    k = rng.normal(k_healthy, 0.3 * k_healthy, n).clip(min=0)
    a = rng.normal(a_healthy, 0.3 * a_healthy, n).clip(min=0)

    # Which units are weak, what kind of defect, and when it starts (Weibull)
    n_def = max(1, int(round(n * defect_rate)))
    is_defect = np.zeros(n, dtype=bool)
    is_defect[rng.choice(n, n_def, replace=False)] = True
    defect_type = np.where(is_defect, rng.integers(0, 3, n), -1)  # 0=fast ESR, 1=leakage, 2=step
    onset = np.where(is_defect, (weibull_scale * rng.weibull(weibull_shape, n)).clip(max=hours.max()), np.inf)
    severity = rng.uniform(3, 10, n)

    # Readings (units x time)
    t = hours[None, :].astype(float)
    after = np.clip(t - onset[:, None], 0, None)
    esr = esr0[:, None] * (1 + k[:, None] * t)
    cap = c0[:, None] * (1 - a[:, None] * t)
    leak = leak0[:, None] * np.ones_like(t)

    m0 = (defect_type == 0)[:, None]
    m1 = (defect_type == 1)[:, None]
    m2 = (defect_type == 2)[:, None]
    sev = severity[:, None]
    esr += m0 * esr0[:, None] * k[:, None] * (sev - 1) * after
    leak += m1 * leak0[:, None] * 0.05 * sev * after
    step = (after > 0).astype(float)
    esr += m2 * esr0[:, None] * 0.3 * sev * step
    cap -= m2 * c0[:, None] * 0.02 * sev * step

    # Measurement noise + occasional glitchy readings
    def noisy(x):
        x = x * (1 + rng.normal(0, noise, x.shape))
        glitch = rng.random(x.shape) < 0.005
        return np.where(glitch, x * rng.uniform(0.7, 1.3, x.shape), x)

    esr, cap, leak = noisy(esr), noisy(cap), noisy(leak)

    return pd.DataFrame({
        "unit": np.repeat(np.arange(n), len(hours)),
        "hour": np.tile(hours, n),
        "esr_ohm": esr.ravel(),
        "capacitance_uF": cap.ravel(),
        "leakage_uA": leak.ravel(),
        "is_defect": np.repeat(is_defect, len(hours)),
        "defect_type": np.repeat(defect_type, len(hours)),
    })
