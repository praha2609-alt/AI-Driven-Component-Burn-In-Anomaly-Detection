"""
Step 1: Simulate burn-in data for aluminium electrolytic capacitors.
Output: burnin_data.csv (one row per unit per reading) and a sanity-check plot.

All numbers marked [REPLACE] are placeholders. Replace them with values from
a real datasheet (Panasonic / Nichicon / Rubycon) and cite it in your paper.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)          # fixed seed -> reproducible results

# ---------- Settings you can change ----------
N = 1000                  # number of virtual capacitors
DEFECT_RATE = 0.05        # 5% are weak on purpose (an assumption, state it)
HOURS = np.arange(0, 169, 4)   # reading every 4 h for 168 h
NOISE = 0.02              # 2% measurement noise (raise it to make the test harder)

ESR0_MEAN = 0.10          # ohm   [REPLACE with datasheet value]
C0_MEAN = 470.0           # uF    [REPLACE]
LEAK0_MEAN = 20.0         # uA    [REPLACE]
SPREAD = 0.07             # 7% unit-to-unit variation in starting values

K_HEALTHY = 1e-4          # ESR rise per hour for healthy units [calibrate from aging data]
A_HEALTHY = 5e-5          # capacitance fall per hour for healthy units
WEIBULL_SHAPE = 0.5       # beta < 1 = infant-mortality behaviour
WEIBULL_SCALE = 60.0      # hours; smaller = defects appear earlier

# ---------- Healthy behaviour for every unit ----------
esr0 = rng.normal(ESR0_MEAN, SPREAD * ESR0_MEAN, N)
c0 = rng.normal(C0_MEAN, SPREAD * C0_MEAN, N)
leak0 = rng.normal(LEAK0_MEAN, SPREAD * LEAK0_MEAN, N)
k = rng.normal(K_HEALTHY, 0.3 * K_HEALTHY, N).clip(min=0)
a = rng.normal(A_HEALTHY, 0.3 * A_HEALTHY, N).clip(min=0)

# ---------- Choose which units are defective ----------
is_defect = np.zeros(N, dtype=bool)
is_defect[rng.choice(N, int(N * DEFECT_RATE), replace=False)] = True
defect_type = np.where(is_defect, rng.integers(0, 3, N), -1)  # 0=fast ESR, 1=leakage, 2=step jump
onset = np.where(is_defect, (WEIBULL_SCALE * rng.weibull(WEIBULL_SHAPE, N)).clip(max=168), np.inf)
severity = rng.uniform(3, 10, N)         # mild (3x) to severe (10x)

# ---------- Build the readings (units x time) ----------
t = HOURS[None, :]
after = np.clip(t - onset[:, None], 0, None)     # hours since defect started (0 for healthy)

esr = esr0[:, None] * (1 + k[:, None] * t)
cap = c0[:, None] * (1 - a[:, None] * t)
leak = leak0[:, None] * np.ones_like(t, dtype=float)

m0 = (defect_type == 0)[:, None]   # fast ESR drift
m1 = (defect_type == 1)[:, None]   # leakage rise
m2 = (defect_type == 2)[:, None]   # sudden step jump
sev = severity[:, None]

esr += m0 * esr0[:, None] * k[:, None] * (sev - 1) * after
leak += m1 * leak0[:, None] * 0.05 * sev * after
step = (after > 0).astype(float)
esr += m2 * esr0[:, None] * 0.3 * sev * step
cap -= m2 * c0[:, None] * 0.02 * sev * step

# ---------- Measurement noise + occasional glitchy readings ----------
def noisy(x):
    x = x * (1 + rng.normal(0, NOISE, x.shape))
    glitch = rng.random(x.shape) < 0.005          # 0.5% bad readings
    return np.where(glitch, x * rng.uniform(0.7, 1.3, x.shape), x)

esr, cap, leak = noisy(esr), noisy(cap), noisy(leak)

# ---------- Save as a table ----------
df = pd.DataFrame({
    "unit": np.repeat(np.arange(N), len(HOURS)),
    "hour": np.tile(HOURS, N),
    "esr_ohm": esr.ravel(),
    "capacitance_uF": cap.ravel(),
    "leakage_uA": leak.ravel(),
    "is_defect": np.repeat(is_defect, len(HOURS)),      # ground truth (only for scoring)
    "defect_type": np.repeat(defect_type, len(HOURS)),
})
df.to_csv("burnin_data.csv", index=False)
print(f"Saved burnin_data.csv: {N} units, {is_defect.sum()} defective, {len(df)} rows")

# ---------- Sanity-check plot ----------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for axi, arr, name in zip(ax, [esr, cap, leak], ["ESR (ohm)", "Capacitance (uF)", "Leakage (uA)"]):
    for i in np.where(~is_defect)[0][:40]:
        axi.plot(HOURS, arr[i], color="grey", alpha=0.3)
    for i in np.where(is_defect)[0][:10]:
        axi.plot(HOURS, arr[i], color="red", alpha=0.8)
    axi.set_title(name)
    axi.set_xlabel("Burn-in hours")
plt.suptitle("Grey = healthy units, Red = defective units")
plt.tight_layout()
plt.savefig("sanity_check.png", dpi=120)
print("Saved sanity_check.png")
