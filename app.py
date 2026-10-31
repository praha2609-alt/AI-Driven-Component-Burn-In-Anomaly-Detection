"""
Burn-in anomaly detection dashboard.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

from simulator import simulate
from features import build_features, get_labels
from detectors import run_all

st.set_page_config(page_title="Burn-in Anomaly Detection", page_icon="🔧", layout="wide")

CUTOFFS = [12, 24, 36, 48, 72, 96, 120, 168]
METHODS = ["PAT (baseline)", "Isolation Forest", "One-Class SVM"]
PARAMS = {"esr_ohm": "ESR (ohm)", "capacitance_uF": "Capacitance (µF)", "leakage_uA": "Leakage (µA)"}
HOURS = np.arange(0, 337, 4)   # simulate up to 336 h so suspects can get extra burn-in
STAGE1 = ["PAT", "Isolation Forest", "One-Class SVM"]
STAGE2 = ["Isolation Forest", "One-Class SVM"]

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {background: rgba(120,144,156,0.12); padding: 14px 16px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached work
@st.cache_data(show_spinner="Simulating capacitors...")
def get_data(n, rate, noise, seed, scale, beta):
    return simulate(n=n, defect_rate=rate, noise=noise, seed=seed,
                    weibull_scale=scale, weibull_shape=beta, hours=HOURS)


@st.cache_data(show_spinner=False)
def run_at(df, cutoff, contamination, pat_k):
    return run_all(build_features(df, cutoff), contamination, pat_k)


@st.cache_data(show_spinner=False)
def run_on_units(df, cutoff, units_mask, contamination, pat_k):
    """Run the detectors using ONLY the selected units (e.g. the accepted ones) as the batch."""
    X = build_features(df, cutoff)
    return run_all(X[units_mask], contamination, pat_k)


@st.cache_data(show_spinner="Running the detectors at every checkpoint...")
def sweep(df, contamination, pat_k):
    y = get_labels(df)
    rows = []
    for c in CUTOFFS:
        for name, (score, flag) in run_at(df, c, contamination, pat_k).items():
            tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
            rows.append({
                "hours": c, "method": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                "ROC-AUC": roc_auc_score(y, score),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔧 Settings")
with st.sidebar.form("settings"):
    st.subheader("Simulated batch")
    n = st.slider("Number of capacitors", 200, 2000, 1000, 100)
    rate = st.slider("Weak units (%)", 1.0, 10.0, 5.0, 0.5) / 100
    noise = st.slider("Measurement noise (%)", 0.5, 10.0, 2.0, 0.5) / 100
    scale = st.slider("Defect onset scale (hours)", 20, 200, 60, 10,
                      help="Smaller = defects start earlier (Weibull scale).")
    beta = st.slider("Weibull shape (β)", 0.3, 1.5, 0.5, 0.1,
                     help="β < 1 means infant-mortality behaviour.")
    seed = st.number_input("Random seed", 0, 9999, 42)

    st.subheader("Detectors")
    contamination = st.slider("Expected weak fraction for ML models (%)", 1.0, 15.0, 5.0, 0.5) / 100
    pat_k = st.slider("PAT limit (robust sigma)", 3.0, 10.0, 6.0, 0.5)

    st.subheader("Cost assumptions")
    c_escape = st.number_input("Cost of a shipped weak unit", 1.0, 10000.0, 100.0)
    c_over = st.number_input("Cost of a scrapped good unit", 0.1, 1000.0, 1.0)
    c_hour = st.number_input("Chamber cost per unit-hour", 0.0, 10.0, 0.02, format="%.3f")

    st.subheader("Two-stage recheck")
    first = st.selectbox("Stage 1 (first screen, 168 h)", STAGE1)
    recheck_a = st.selectbox("Stage 2A: recheck the FLAGGED units with", STAGE2,
                             help="Removes false alarms. Only units Stage 1 flagged get extra burn-in.")
    recheck_b = st.selectbox("Stage 2B: recheck the ACCEPTED units with", STAGE2, index=1,
                             help="Looks for weak units Stage 1 missed. Uses only the accepted units as the reference batch.")
    extra = st.slider("Extra burn-in for suspects (hours)", 24, 168, 168, 24)
    audit_pct = st.slider("Stage 2B: % of accepted units rechecked (most borderline)", 0, 100, 10, 5,
                          help="Extra burn-in for the accepted units that looked closest to being flagged.")
    audit_cont = st.slider("Expected weak fraction among accepted units (%)", 0.5, 10.0, 2.0, 0.5,
                           help="Used only for the audit. Fewer weak units should remain after stage 1.") / 100
    st.form_submit_button("▶ Run / update", width="stretch")

# ------------------------------------------------------------------ compute
df = get_data(n, rate, noise, int(seed), scale, beta)
y = get_labels(df)
res = sweep(df, contamination, pat_k)
res["cost"] = res.FN * c_escape + res.FP * c_over + n * res.hours * c_hour

# ------------------------------------------------------------------ header
st.title("Early Anomaly Detection in Burn-In Screening")
st.caption("Aluminium electrolytic capacitors · simulation-based comparison of PAT, "
           "Isolation Forest and One-Class SVM")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Capacitors", f"{n:,}")
k2.metric("Weak units (hidden)", int(y.sum()))
k3.metric("Measurement noise", f"{noise*100:.1f}%")
k4.metric("Burn-in length", "168 h")

tab_data, tab_res, tab_early, tab_cost, tab_two, tab_unit, tab_about = st.tabs(
    ["📈 Data", "🏁 Results", "⏱️ Early detection", "💰 Cost", "🔁 Two-stage recheck",
     "🔍 Unit inspector", "ℹ️ About"])

# ------------------------------------------------------------------ Data
with tab_data:
    st.subheader("What the simulated readings look like")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    healthy = df[~df.is_defect].unit.unique()[:40]
    weak = df[df.is_defect].unit.unique()[:10]
    for ax, (p, label) in zip(axes, PARAMS.items()):
        m = df[df.hour <= 168].pivot(index="unit", columns="hour", values=p)
        for u in healthy:
            ax.plot(m.columns, m.loc[u], color="#90a4ae", alpha=0.35)
        for u in weak:
            ax.plot(m.columns, m.loc[u], color="#d32f2f", alpha=0.85)
        ax.set_title(label); ax.set_xlabel("Burn-in hours"); ax.grid(alpha=0.2)
    fig.suptitle("Grey = healthy units, Red = weak units", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Raw data (first 200 rows)"):
        st.dataframe(df.head(200), width="stretch")
    st.download_button("⬇ Download full dataset (CSV)", df.to_csv(index=False),
                       "burnin_data.csv", "text/csv")

# ------------------------------------------------------------------ Results
with tab_res:
    st.subheader("How many weak units does each method catch?")
    cutoff = st.select_slider("Hours of burn-in data used", options=CUTOFFS, value=168)
    sel = res[res.hours == cutoff].set_index("method").loc[METHODS]
    cols = st.columns(3)
    for col, name in zip(cols, METHODS):
        r = sel.loc[name]
        col.metric(name, f"{int(r.TP)} / {int(r.TP + r.FN)} caught",
                   f"{int(r.FP)} good units wrongly rejected", delta_color="off")
    table = sel[["TP", "FN", "FP", "TN", "precision", "recall", "F1", "ROC-AUC"]].rename(columns={
        "TP": "Caught", "FN": "Missed", "FP": "False alarms", "TN": "Good kept"})
    st.dataframe(table.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                     "F1": "{:.2f}", "ROC-AUC": "{:.2f}"}),
                 width="stretch")
    st.caption("Accuracy is not shown on purpose: with few weak units, a detector that "
               "calls everything good would score ~95% and catch nothing.")

# ------------------------------------------------------------------ Early detection
with tab_early:
    st.subheader("How early are weak units caught?")
    st.caption("Each point uses only the readings up to that many hours.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weak units caught (recall %)**")
        st.line_chart((res.pivot(index="hours", columns="method", values="recall") * 100)[METHODS])
    with c2:
        st.markdown("**Good units wrongly rejected (false alarms)**")
        st.line_chart(res.pivot(index="hours", columns="method", values="FP")[METHODS])
    st.markdown("**F1 score (balance of precision and recall, higher is better)**")
    st.line_chart(res.pivot(index="hours", columns="method", values="F1")[METHODS])
    st.caption("F1 = 2 x precision x recall / (precision + recall). It is high only when the method catches most "
               "weak units AND most of its rejections are really weak.")
    st.markdown("**Separation quality (ROC-AUC, higher is better)**")
    st.line_chart(res.pivot(index="hours", columns="method", values="ROC-AUC")[METHODS])

# ------------------------------------------------------------------ Cost
with tab_cost:
    st.subheader("Cost trade-off: escapes vs overkill vs chamber time")
    st.line_chart(res.pivot(index="hours", columns="method", values="cost")[METHODS])
    best = res.loc[res.groupby("method").cost.idxmin()].set_index("method").loc[METHODS]
    st.markdown("**Cheapest burn-in length for each method**")
    st.dataframe(best[["hours", "cost", "TP", "FN", "FP", "F1"]].rename(columns={
        "hours": "Best hours", "cost": "Total cost", "TP": "Caught", "FN": "Missed (escapes)",
        "FP": "False alarms (overkill)"}).style.format({"Total cost": "{:.0f}", "F1": "{:.2f}"}),
        width="stretch")
    st.caption("Costs are arbitrary units set in the sidebar. They are assumptions, not real factory figures.")

# ------------------------------------------------------------------ Two-stage recheck
with tab_two:
    st.subheader("Two-stage screening: Stage 1, then Stage 2A and Stage 2B")
    st.caption("Stage 1 screens the whole batch at 168 h. **Stage 2A** rechecks the FLAGGED units (removes false alarms): "
               "a unit stays rejected only if Stage 2A flags it too. **Stage 2B** rechecks the most borderline ACCEPTED units "
               "(finds missed weak units), using only the accepted units as the reference batch. "
               "Each part has its own detector choice.")
    t1 = run_at(df, 168, contamination, pat_k)
    t2 = run_at(df, 168 + extra, contamination, pat_k)
    v1 = np.vstack([t1[m][1] for m in METHODS]).sum(axis=0)
    v2 = np.vstack([t2[m][1] for m in METHODS]).sum(axis=0)

    def pick(rule, out, votes):
        """Turn a rule name into a True/False flag for every unit."""
        if rule == "PAT":
            return out["PAT (baseline)"][1]
        if rule == "Isolation Forest":
            return out["Isolation Forest"][1]
        if rule == "One-Class SVM":
            return out["One-Class SVM"][1]
        if rule == "Any of 3":
            return votes >= 1
        if rule == "2 of 3 agree":
            return votes >= 2
        return votes >= 3            # "All 3 agree"

    def tally(flag, unit_hours):
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        fn = int(((~flag) & (y == 1)).sum())
        return {"Caught": tp, "Missed": fn, "False alarms": fp,
                "Precision": tp / (tp + fp) if tp + fp else 0.0,
                "Recall": tp / (tp + fn) if tp + fn else 0.0,
                "F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                "Chamber unit-hours": int(unit_hours),
                "Total cost": fn * c_escape + fp * c_over + unit_hours * c_hour}

    stage1 = pick(first, t1, v1)
    stage2 = pick(recheck_a, t2, v2)
    final = stage1 & stage2
    accepted = ~stage1
    base_hours = n * 168 + int(stage1.sum()) * extra

    # Audit: the detectors are re-run on the ACCEPTED units only (rejected and recheck-flagged
    # units are left out completely), using their longer-run data.
    audit_flag = np.zeros(n, dtype=bool)
    ta = va = None
    if accepted.sum() >= 20:
        ta = run_on_units(df, 168 + extra, accepted, audit_cont, pat_k)
        va = np.vstack([ta[m][1] for m in METHODS]).sum(axis=0)
        audit_flag[accepted] = pick(recheck_b, ta, va)

    def base_hours_for(base):
        return base_hours

    def audit(pct, aflag=None, base=None):
        """Stage 2B: recheck the pct% most borderline ACCEPTED units (highest Isolation Forest score at 168 h).
        aflag = per-unit audit flags (defaults to the chosen 2B detector); base = rejections after Stage 2A."""
        aflag = audit_flag if aflag is None else aflag
        base = final if base is None else base
        k = int(round(pct / 100 * accepted.sum()))
        border = np.zeros(n, dtype=bool)
        if k > 0:
            sc = np.where(accepted, t1["Isolation Forest"][0], -np.inf)
            border[np.argsort(sc)[-k:]] = True
        newly = border & aflag
        return base | newly, border, newly, base_hours_for(base) + int(border.sum()) * extra

    flag_a, border_a, newly_a, hours_a = audit(audit_pct)
    flag_all, _, _, hours_all = audit(100)

    two_name = f"Stage 1 {first} → 2A {recheck_a}"
    audit_name = f"+ 2B {recheck_b} on {audit_pct}% of accepted"
    table = pd.DataFrame({
        "PAT alone (168 h)": tally(t1["PAT (baseline)"][1], n * 168),
        "Isolation Forest alone (168 h)": tally(t1["Isolation Forest"][1], n * 168),
        "One-Class SVM alone (168 h)": tally(t1["One-Class SVM"][1], n * 168),
        "2 of 3 vote (168 h)": tally(v1 >= 2, n * 168),
        two_name: tally(final, base_hours),
        audit_name: tally(flag_a, hours_a),
        "+ audit ALL accepted": tally(flag_all, hours_all),
    }).T

    def card(col, name):
        r = table.loc[name]
        col.metric(name, f"{int(r['Caught'])} / {int(r['Caught'] + r['Missed'])} caught",
                   f"{int(r['False alarms'])} good units wrongly rejected · F1 {r['F1']:.2f}", delta_color="off")

    for col, name in zip(st.columns(3), ["PAT alone (168 h)", "Isolation Forest alone (168 h)",
                                         "One-Class SVM alone (168 h)"]):
        card(col, name)
    for col, name in zip(st.columns(2), [two_name, audit_name]):
        card(col, name)

    fmt = {"Caught": "{:.0f}", "Missed": "{:.0f}", "False alarms": "{:.0f}",
           "Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}",
           "Chamber unit-hours": "{:,.0f}", "Total cost": "{:,.0f}"}
    st.dataframe(table.style.format(fmt), width="stretch")

    new_weak = int((newly_a & (y == 1)).sum())
    st.info(f"Stage 1 ({first}) flagged {int(stage1.sum())} units. Only those got {extra} extra "
            f"hours of burn-in, and {int(final.sum())} were rejected after Stage 2A ({recheck_a}). "
            f"Stage 2B ({recheck_b}) looked only at accepted units: it rechecked {int(border_a.sum())} borderline ones and rejected "
            f"{int(newly_a.sum())} more ({new_weak} of them truly weak).")
    unit_cost = extra * c_hour
    st.caption(f"Break-even: rechecking one accepted unit costs about {unit_cost:.2f} in chamber time, "
               f"and catching a weak unit saves {c_escape:.0f}. So an audit pays off only if more than "
               f"about {unit_cost / c_escape * 100:.1f}% of the audited units are truly weak.")

    with st.expander("How much should we audit? (cost vs % of accepted units rechecked)"):
        rows = []
        for pct in [0, 10, 20, 30, 50, 100]:
            f, b, _, h = audit(pct)
            r = tally(f, h)
            rows.append({"Audit % of accepted": pct, "Units rechecked": int(b.sum()),
                         "Caught": r["Caught"], "False alarms": r["False alarms"], "F1": r["F1"],
                         "Chamber unit-hours": r["Chamber unit-hours"], "Total cost": r["Total cost"]})
        sweep_df = pd.DataFrame(rows).set_index("Audit % of accepted")
        st.dataframe(sweep_df.style.format({"F1": "{:.2f}", "Chamber unit-hours": "{:,.0f}", "Total cost": "{:,.0f}"}),
                     width="stretch")
        st.line_chart(sweep_df["Total cost"])
        st.caption("If the line rises as the audit grows, extra rechecking costs more than it saves "
                   "under the current cost assumptions. Raise the cost of a shipped weak unit in the "
                   "sidebar to see when auditing starts to pay off.")

    with st.expander("Compare every Stage 2A + Stage 2B choice (for your Stage 1 and audit %)"):
        rows = []
        for a in STAGE2:
            fin_a = stage1 & pick(a, t2, v2)
            for b in STAGE2:
                af = np.zeros(n, dtype=bool)
                if ta is not None:
                    af[accepted] = pick(b, ta, va)
                f, bd, _, h = audit(audit_pct, af, fin_a)
                r = tally(f, h)
                rows.append({"Stage 2A (flagged)": a, "Stage 2B (accepted)": b, "Caught": r["Caught"],
                             "Missed": r["Missed"], "False alarms": r["False alarms"],
                             "Precision": r["Precision"], "F1": r["F1"], "Total cost": r["Total cost"]})
        grid = pd.DataFrame(rows).sort_values("Total cost").reset_index(drop=True)
        st.dataframe(grid.style.format({"Precision": "{:.2f}", "F1": "{:.2f}", "Total cost": "{:,.0f}"}), width="stretch")
        st.caption(f"Stage 1 stays {first}. Sorted from cheapest to most expensive under the sidebar cost assumptions.")

    with st.expander("Compare every Stage 1 + Stage 2A combination (no Stage 2B)"):
        rows = []
        for a in STAGE1:
            s1 = pick(a, t1, v1)
            for b in STAGE2:
                r = tally(s1 & pick(b, t2, v2), n * 168 + int(s1.sum()) * extra)
                rows.append({"Stage 1": a, "Stage 2": b, "Caught": r["Caught"],
                             "Missed": r["Missed"], "False alarms": r["False alarms"],
                             "Precision": r["Precision"], "F1": r["F1"], "Total cost": r["Total cost"]})
        combos = pd.DataFrame(rows).sort_values("Total cost").reset_index(drop=True)
        st.dataframe(combos.style.format({"Precision": "{:.2f}", "F1": "{:.2f}", "Total cost": "{:,.0f}"}),
                     width="stretch")
        st.caption("Sorted from cheapest to most expensive under the sidebar cost assumptions.")

    st.caption("Stage 2A (flagged units) can only remove false alarms. Stage 2B (accepted units) can "
               "recover missed weak units, but costs extra chamber time. Stage 2 compares each unit "
               "with the whole batch's longer-run data, which a real factory would need reference "
               "data for. Costs are assumptions from the sidebar.")

# ------------------------------------------------------------------ Unit inspector
with tab_unit:
    st.subheader("Inspect a single capacitor")

    # ---- choose a unit: type a number, or jump to an interesting example
    # (the examples use the hidden answer key, so they only exist in a simulation)
    groups = {
        "Type a capacitor number": None,
        "Weak unit, caught": np.where(flag_a & (y == 1))[0],
        "Weak unit, missed (escaped)": np.where(~flag_a & (y == 1))[0],
        "Good unit, wrongly rejected": np.where(flag_a & (y == 0))[0],
        "Good unit, flagged at Stage 1 but cleared by Stage 2A": np.where(stage1 & ~stage2 & (y == 0))[0],
        "Weak unit, found only by Stage 2B (accepted units)": np.where(newly_a & (y == 1))[0],
    }
    g1, g2 = st.columns(2)
    group = g1.selectbox("Pick an example (or type a number)", list(groups), key="unit_group")
    members = groups[group]
    if members is None:
        unit = int(g2.number_input("Capacitor number", 0, n - 1, 0, key="unit_manual"))
    elif len(members) == 0:
        g2.info("No units in this group for the current settings.")
        unit = 0
    else:
        unit = int(g2.selectbox(f"Unit ({len(members)} in this group)", [int(m) for m in members],
                                key="unit_pick"))

    # ---- the two-stage journey of this unit
    st.markdown("#### Two-stage screening result")
    st.caption(f"Stage 1 = {first} at 168 h on every unit. Stage 2A = {recheck_a} at {168 + extra} h on units Stage 1 "
               f"flagged. Stage 2B = {recheck_b} at {168 + extra} h on the {audit_pct}% most borderline accepted units.")

    s1_flag, s2_flag = bool(stage1[unit]), bool(stage2[unit])
    if s1_flag and s2_flag:
        path = "Flagged at Stage 1 → still flagged at Stage 2A"
        verdict, kind = "REJECT", "error"
    elif s1_flag:
        path = "Flagged at Stage 1 → cleared by Stage 2A"
        verdict, kind = "ACCEPT (false alarm avoided by Stage 2A)", "success"
    elif border_a[unit] and audit_flag[unit]:
        path = "Accepted at Stage 1 → picked for Stage 2B → flagged by Stage 2B"
        verdict, kind = "REJECT (caught by Stage 2B)", "error"
    elif border_a[unit]:
        path = "Accepted at Stage 1 → picked for Stage 2B → passed Stage 2B"
        verdict, kind = "ACCEPT", "success"
    else:
        path = "Accepted at Stage 1 → not borderline, not rechecked by Stage 2B"
        verdict, kind = "ACCEPT", "success"
    {"error": st.error, "success": st.success}[kind](f"Capacitor {unit}: **{verdict}**  \n{path}")

    # Only the detectors that belong to the chosen Stage 1 / 2A / 2B combination are shown.
    def methods_for(rule):
        return {"PAT": ["PAT (baseline)"], "Isolation Forest": ["Isolation Forest"],
                "One-Class SVM": ["One-Class SVM"]}.get(rule, METHODS)

    def says(rule, out, votes, idx=None):
        i = unit if idx is None else idx
        ms = methods_for(rule)
        if len(ms) == 1:
            return f"{ms[0]} flags it" if out[ms[0]][1][i] else f"{ms[0]} does not flag it"
        return f"{int(votes[i])} of 3 methods flag it"

    used = [m for m in METHODS if m in methods_for(first) or m in methods_for(recheck_a)
            or m in methods_for(recheck_b)]
    acc_idx = int(np.cumsum(accepted)[unit] - 1)          # this unit's position among the accepted units
    in_2b = bool((not s1_flag) and border_a[unit] and ta is not None)

    p1, p2, p3 = st.columns(3)
    p1.metric(f"Stage 1 ({first})", "Flagged" if s1_flag else "Passed", says(first, t1, v1), delta_color="off")
    if s1_flag:
        p2.metric(f"Stage 2A ({recheck_a})", "Flagged" if s2_flag else "Cleared",
                  f"{says(recheck_a, t2, v2)} at {168 + extra} h", delta_color="off")
    else:
        p2.metric(f"Stage 2A ({recheck_a})", "Not needed", "Stage 1 accepted it", delta_color="off")
    if s1_flag:
        p3.metric(f"Stage 2B ({recheck_b})", "Not applicable", "unit was flagged, so 2A handles it", delta_color="off")
    elif in_2b:
        p3.metric(f"Stage 2B ({recheck_b})", "Flagged" if audit_flag[unit] else "Passed",
                  says(recheck_b, ta, va, acc_idx), delta_color="off")
    else:
        p3.metric(f"Stage 2B ({recheck_b})", "Not rechecked", "not borderline enough", delta_color="off")

    rows = []
    for m in methods_for(first):
        rows.append({"Stage": "Stage 1", "Detector": m, "Burn-in data": "168 h",
                     "Result": "Flagged" if t1[m][1][unit] else "Passed",
                     "Score": f"{t1[m][0][unit]:.2f}"})
    for m in methods_for(recheck_a):
        if s1_flag:
            res, sc = ("Flagged" if t2[m][1][unit] else "Cleared"), f"{t2[m][0][unit]:.2f}"
        else:
            res, sc = "Not run", "-"
        rows.append({"Stage": "Stage 2A", "Detector": m, "Burn-in data": f"{168 + extra} h",
                     "Result": res, "Score": sc})
    for m in methods_for(recheck_b):
        if in_2b:
            res, sc = ("Flagged" if ta[m][1][acc_idx] else "Passed"), f"{ta[m][0][acc_idx]:.2f}"
        elif s1_flag:
            res, sc = "Not applicable", "-"
        else:
            res, sc = "Not run", "-"
        rows.append({"Stage": "Stage 2B", "Detector": m, "Burn-in data": f"{168 + extra} h",
                     "Result": res, "Score": sc})
    detail = pd.DataFrame(rows)
    st.table(detail.style.hide(axis="index")
             .set_properties(**{"text-align": "center"})
             .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]))
    st.caption("Each stage lists only the detector(s) chosen for it. Higher score = more suspicious. "
               "Stage 2A runs only on units Stage 1 flagged; Stage 2B runs only on the borderline accepted units, "
               "scored against the accepted units alone, so its scores are not comparable with 2A's.")

    # ---- borderline accepted units (the Stage 2B candidates) and why
    st.markdown("#### Borderline accepted capacitors (Stage 2B candidates)")
    X168 = build_features(df, 168)
    zf = (X168 - X168.median()) / ((X168.quantile(0.75) - X168.quantile(0.25)) / 1.349).replace(0, np.nan)
    p_names = {"esr": "ESR", "capacitance": "Capacitance", "leakage": "Leakage"}
    s_names = {"mean": "average change", "std": "scatter", "slope": "drift rate", "max": "peak change"}

    def reasons(u, k=2):
        z = zf.iloc[u].dropna()
        top = z.reindex(z.abs().sort_values(ascending=False).index)[:k]
        out_ = []
        for f, v in top.items():
            p_, st_ = f.split("_")
            out_.append(f"{p_names[p_]} {s_names[st_]} ({v:+.1f} sigma)")
        return ", ".join(out_)

    if_score, if_flag = t1["Isolation Forest"]
    flag_line = float(np.min(if_score[if_flag])) if if_flag.any() else float("nan")
    acc_ids = np.where(accepted)[0]
    order = acc_ids[np.argsort(-if_score[acc_ids])]
    rank = {int(u): i + 1 for i, u in enumerate(order)}
    k_b = int(border_a.sum())
    cut_score = float(np.min(if_score[border_a])) if k_b else float("nan")

    st.caption(f"After Stage 1, {int(accepted.sum())} units are accepted. They are ranked by their Isolation Forest "
               f"anomaly score at 168 h (higher = looked stranger). The top {audit_pct}% ({k_b} units) are the "
               f"borderline ones: they went unflagged, but only just. Only these get extra burn-in and Stage 2B.")

    if s1_flag:
        st.info(f"Capacitor {unit} was flagged at Stage 1, so it is not an accepted unit and is not a Stage 2B candidate. "
                f"Stage 2A handles it.")
    else:
        gap = flag_line - if_score[unit]
        why = reasons(unit)
        if border_a[unit]:
            st.warning(f"**Capacitor {unit} is borderline.** It ranks #{rank[unit]} of {int(accepted.sum())} accepted "
                       f"units (the top {k_b} are borderline). Its anomaly score is {if_score[unit]:.3f}, "
                       f"{'just ' + format(gap, '.3f') + ' below' if gap >= 0 else format(-gap, '.3f') + ' above'} "
                       f"the Isolation Forest flag line ({flag_line:.3f}). "
                       f"Why it looks odd: {why}. So it gets extra burn-in and a Stage 2B check with {recheck_b}.")
        else:
            st.success(f"**Capacitor {unit} is not borderline.** It ranks #{rank[unit]} of {int(accepted.sum())} accepted "
                       f"units, outside the top {k_b}. Its score is {if_score[unit]:.3f}, below the borderline cut-off "
                       f"of {cut_score:.3f}, so it is shipped without a Stage 2B check. Its most unusual features "
                       f"are: {why}.")

    show_key = st.checkbox("Show the answer key column (simulation only)", key="border_key")
    brows = []
    for u in order[:k_b]:
        u = int(u)
        r_ = {"Capacitor": u, "Rank": rank[u], "Anomaly score": round(float(if_score[u]), 3),
              "Score minus flag line": round(float(if_score[u] - flag_line), 3),
              "Why it looks odd": reasons(u),
              "Stage 2B result": "Rejected" if audit_flag[u] else "Passed"}
        if show_key:
            r_["Actually weak?"] = "Yes" if y[u] else "No"
        brows.append(r_)
    if brows:
        st.dataframe(pd.DataFrame(brows), hide_index=True, width="stretch", height=300)
        st.caption("Score minus flag line: negative = still below Isolation Forest's flag line; positive = above it "
                   "(possible when Stage 1 uses a different detector). 'Sigma' = how many robust standard "
                   "deviations the unit's feature sits from the batch median. Stage 2B scores the unit against the "
                   "accepted units only.")
    else:
        st.info("No borderline units: set the Stage 2B percentage above 0.")

    # ---- ML analysis over time: re-run the chosen detectors at every checkpoint
    end_h = 168 + extra
    checkpoints = sorted(set(CUTOFFS) | {end_h})
    track = {}
    for m in used:
        sc_l, fl_l, th_l = [], [], []
        for c in checkpoints:
            score, flag = run_at(df, c, contamination, pat_k)[m]
            sc_l.append(float(score[unit])); fl_l.append(bool(flag[unit]))
            th_l.append(float(np.min(score[flag])) if flag.any() else np.nan)
        first_hit = next((c for c, f in zip(checkpoints, fl_l) if f), None)
        track[m] = {"score": sc_l, "flag": fl_l, "thr": th_l, "first": first_hit}

    st.markdown("#### What the ML detectors see in the readings")
    lines = []
    for m in used:
        fh = track[m]["first"]
        lines.append(f"- **{m}** " + (f"first flags capacitor {unit} at **{fh} h**." if fh is not None
                                      else f"never flags capacitor {unit} up to {end_h} h."))
    st.markdown("\n".join(lines))

    colors = {"PAT (baseline)": "#6a1b9a", "Isolation Forest": "#ef6c00", "One-Class SVM": "#00838f"}
    d2 = df[df.hour <= end_h]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (p, label) in zip(axes, PARAMS.items()):
        m_ = d2.pivot(index="unit", columns="hour", values=p)
        hrs = m_.columns.values
        ax.fill_between(hrs, m_.quantile(0.05).values, m_.quantile(0.95).values, color="#cfd8dc", label="Batch 5–95%")
        ax.plot(hrs, m_.median().values, color="#78909c", lw=1, label="Batch median")
        ax.plot(hrs, m_.loc[unit].values, color="#d32f2f", lw=2, label=f"Unit {unit}")
        ax.axvline(168, color="#1565c0", ls="--", lw=1)
        ax.axvline(end_h, color="#2e7d32", ls="--", lw=1)
        for m in used:
            fh = track[m]["first"]
            if fh is not None:
                ax.axvline(fh, color=colors[m], ls=":", lw=2, label=f"{m} first flags it ({fh} h)")
        ax.set_title(label); ax.set_xlabel("Burn-in hours"); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="upper left")
    ymax = axes[0].get_ylim()[1]
    axes[0].text(168, ymax, " Stage 1", color="#1565c0", fontsize=8, va="top")
    if extra > 0:
        axes[0].text(end_h, ymax, " Stage 2", color="#2e7d32", fontsize=8, va="top", ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(used), figsize=(5 * len(used) if len(used) > 1 else 7, 3.2), squeeze=False)
    for ax, m in zip(axes[0], used):
        t = track[m]
        ax.plot(checkpoints, t["thr"], color="#9e9e9e", ls="--", lw=1.5, label="Flag line (lowest flagged score in batch)")
        ax.plot(checkpoints, t["score"], color=colors[m], lw=2, marker="o", ms=3, label=f"Unit {unit} score")
        hit = [(c, sc_) for c, sc_, f in zip(checkpoints, t["score"], t["flag"]) if f]
        if hit:
            ax.scatter(*zip(*hit), color="#d32f2f", zorder=5, s=36, label="Flagged")
        ax.axvline(168, color="#1565c0", ls="--", lw=1)
        ax.set_title(f"{m}: anomaly score over time"); ax.set_xlabel("Hours of data used")
        ax.grid(alpha=0.2); ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Each checkpoint re-runs the detector on the whole batch using only the data up to that hour. "
               "A unit is flagged once its score rises above the batch's flag line. The earlier the red dots start, "
               "the earlier the ML method spots this unit.")

    # ---- the original single-checkpoint view
    st.markdown("#### Early-detection view (one checkpoint)")
    st.caption(f"What your Stage 1 rule ({first}) would say if burn-in were stopped at this hour.")
    c1, c2 = st.columns([1, 2])
    cut_u = c1.select_slider("Hours of data available", options=CUTOFFS, value=72, key="cut_u")
    out = run_at(df, cut_u, contamination, pat_k)
    votes_u = np.vstack([out[m][1] for m in METHODS]).sum(axis=0)
    early = pick(first, out, votes_u)
    c2.markdown(f"**Batch result at {cut_u} h ({first}):** ⛔ Flagged {int(early.sum())} · "
                f"✅ Passed {n - int(early.sum())}")
    if early[unit]:
        st.error(f"Capacitor {unit} at {cut_u} h: **Flagged** by Stage 1 ({first})")
    else:
        st.success(f"Capacitor {unit} at {cut_u} h: **Passed** Stage 1 ({first})")

    ms_first = methods_for(first)
    for col, m in zip(st.columns(len(ms_first)), ms_first):
        score, flag = out[m]
        col.metric(m, "Flagged" if flag[unit] else "OK", f"score {score[unit]:.2f}", delta_color="off")

    with st.expander("Reveal the answer key (only possible in a simulation)"):
        truth = bool(y[unit])
        kinds = {0: "fast ESR rise", 1: "leakage rise", 2: "sudden step jump"}
        dt = int(df[df.unit == unit].defect_type.iloc[0])
        st.write(f"Capacitor {unit} is **{'weak' if truth else 'healthy'}**"
                 + (f" (defect type: {kinds[dt]})" if truth else "") + ".")

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown("""
**What this is.** A software simulation of burn-in screening. Real burn-in data is confidential,
so the batch is generated from formulas with realistic noise, unit-to-unit variation and
Weibull (β < 1) infant-mortality defects.

**Methods compared.**
- **PAT**: industry-style outlier limits (robust mean ± k robust sigma).
- **Isolation Forest** and **One-Class SVM**: unsupervised ML that learn what normal looks like.

**Limits.** The data is synthetic. Defect rate, noise, Weibull settings and costs are assumptions.
Results show how methods compare on this testbed and need validation on real burn-in data.
""")