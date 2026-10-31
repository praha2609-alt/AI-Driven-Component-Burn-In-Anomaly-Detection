"""
File 2: Turn the raw readings (burnin_data.csv) into a few numbers per capacitor.
The detectors never see the raw table, only these features.
Every feature is measured RELATIVE to the unit's own starting value, because
healthy units start at slightly different values.
"""
import numpy as np
import pandas as pd

PARAMS = ["esr_ohm", "capacitance_uF", "leakage_uA"]


def load_data(path="burnin_data.csv"):
    return pd.read_csv(path)


def build_features(df, cutoff_hour=168):
    """Features using ONLY readings up to cutoff_hour (this is how we test 'how early')."""
    d = df[df.hour <= cutoff_hour]
    hours = np.sort(d.hour.unique())
    t = hours - hours.mean()
    feats = {}
    for p in PARAMS:
        m = d.pivot(index="unit", columns="hour", values=p).sort_index()[hours].to_numpy()
        base = m[:, :2].mean(axis=1, keepdims=True)      # unit's starting value
        rel = m / base - 1.0                              # change relative to start
        name = p.split("_")[0]
        feats[f"{name}_mean"] = rel.mean(axis=1)
        feats[f"{name}_std"] = rel.std(axis=1)
        feats[f"{name}_slope"] = (rel * t).sum(axis=1) / (t ** 2).sum()
        feats[f"{name}_max"] = rel.max(axis=1)
    return pd.DataFrame(feats, index=sorted(d.unit.unique()))


def get_labels(df):
    """Answer key (1 = defective). Used ONLY for scoring, never given to a detector."""
    return df.groupby("unit").is_defect.first().sort_index().astype(int).to_numpy()
