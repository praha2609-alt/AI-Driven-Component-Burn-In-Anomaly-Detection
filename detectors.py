"""
File 3: The three methods we compare.
Each returns (score, flag): higher score = more suspicious, flag = True if rejected.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

PAT_K = 6.0          # standard PAT limit: robust mean +/- 6 robust sigma
CONTAMINATION = 0.05  # expected fraction of weak units given to the ML models (an assumption)


def pat_detector(X, k=PAT_K):
    """Part Average Testing (industry baseline): flag a unit if ANY feature lies
    outside median +/- k * robust sigma of the batch."""
    med = X.median()
    sigma = (X.quantile(0.75) - X.quantile(0.25)) / 1.349   # robust sigma from IQR
    z = (X - med).abs() / sigma
    score = z.max(axis=1).to_numpy()
    return score, score > k


def isolation_forest_detector(X, contamination=CONTAMINATION, seed=42):
    Xs = RobustScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=seed).fit(Xs)
    return -model.score_samples(Xs), model.predict(Xs) == -1


def ocsvm_detector(X, contamination=CONTAMINATION):
    # Tuned on simulated batches: Robust scaling + small gamma (0.01) gave the best F1/AUC.
    # (Robust scaling with gamma="scale" collapses to near-random, so the pair must go together.)
    Xs = RobustScaler().fit_transform(X)
    model = OneClassSVM(kernel="rbf", gamma=0.01, nu=contamination).fit(Xs)
    return -model.decision_function(Xs), model.predict(Xs) == -1


def run_all(X, contamination=CONTAMINATION, pat_k=PAT_K):
    return {
        "PAT (baseline)": pat_detector(X, pat_k),
        "Isolation Forest": isolation_forest_detector(X, contamination),
        "One-Class SVM": ocsvm_detector(X, contamination),
    }