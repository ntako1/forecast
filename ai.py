#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTO EARTHQUAKE FORECASTER — AI-Enhanced Edition (SAFE VERSION)
---------------------------------------------------------------
Συνδυάζει:
 - Εκτίμηση σεισμικής συχνότητας (λ) & προσομοίωση Number_of_events (Poisson)
 - Ενεργειακή πρόβλεψη μεγεθών (Predicted_Magnitude)
 - AI-based injury calibration (MLP Neural Network)
 - Εξαγωγή σε Excel με δύο σενάρια (mild / extreme)

Παράγει:
 - forecast_ai.xlsx με δύο φύλλα (mild / extreme)
 - forecast_metrics_log.csv με βασικές μετρικές
"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# ============================================================
# ΡΥΘΜΙΣΕΙΣ
# ============================================================
START_YEAR = 2026
END_YEAR = 2040
FIXED_SEED = 20251024

SCENARIOS = {
    "mild": {"EXTREME_MULT":0.5, "ENERGY_COEFF": 0.5},
    "extreme": {"EXTREME_MULT": 1.0, "ENERGY_COEFF": 1.0},
}
# ============================================================


# ------------------------------------------------------------
# Poisson generator
# ------------------------------------------------------------
def poisson_knuth(lam: float, rng: random.Random) -> int:
    if lam <= 0.0 or not math.isfinite(lam):
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


# ------------------------------------------------------------
# Εκτίμηση λ (μέσος σεισμοί ≥5.2 ανά έτος)
# ------------------------------------------------------------
def estimate_lambda_from_csv(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")

    if "Number_of_events" in df.columns:
        df["Number_of_events"] = pd.to_numeric(df["Number_of_events"], errors="coerce")
        d = df.dropna(subset=["Year", "Number_of_events"]).copy()
        agg = d.groupby("Year", as_index=False)["Number_of_events"].sum()
        years = int(agg["Year"].nunique())
        total = float(agg["Number_of_events"].sum())
        if years == 0:
            raise ValueError("No valid years in CSV.")
        return total / years

    # Fallback: με βάση Magnitude >= 5.2
    mags = pd.to_numeric(df.get("Magnitude"), errors="coerce")
    d = df.dropna(subset=["Year", "Magnitude"]).copy()
    d["ge52"] = (mags >= 5.2).astype(int)
    agg = d.groupby("Year", as_index=False)["ge52"].max()
    p = float(agg["ge52"].mean())
    return -math.log(1.0 - p) if p < 1.0 else float("inf")


# ------------------------------------------------------------
# Ενεργειακά προσαρμοσμένη πρόβλεψη Magnitude
# ------------------------------------------------------------
def energy_adjusted_forecast(df: pd.DataFrame,
                             forecast_years: list[int],
                             threshold: float,
                             k: float,
                             seed: int = FIXED_SEED) -> pd.DataFrame:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Magnitude"] = pd.to_numeric(df.get("Magnitude"), errors="coerce")
    df["Number_of_events"] = pd.to_numeric(df.get("Number_of_events"), errors="coerce").fillna(0)

    hist = df[(df["Year"] < START_YEAR) & (df["Magnitude"] >= threshold)].dropna(subset=["Magnitude"])
    if hist.empty:
        raise ValueError(" Δεν υπάρχουν ιστορικά σεισμοί με Magnitude ≥ 5.2.")

    mean_mag = hist["Magnitude"].mean()
    std_mag = hist["Magnitude"].std(ddof=1)

    rng = np.random.default_rng(seed)
    preds = []
    for _, row in df[df["Year"].isin(forecast_years)].iterrows():
        n_events = float(row["Number_of_events"])
        if n_events <= 0:
            preds.append(np.nan)
            continue
        adjusted_mean = mean_mag + k * math.log10(1 + n_events)
        predicted_mag = rng.normal(adjusted_mean, std_mag)
        preds.append(predicted_mag)

    df_future_mask = df["Year"].isin(forecast_years)
    df.loc[df_future_mask, "Predicted_Magnitude"] = np.round(preds, 3)
    df["Magnitude"] = df["Magnitude"].fillna(df["Predicted_Magnitude"])

    # Υπολογισμός σεισμικής ενέργειας
    df["log_Energy"] = np.where(df["Magnitude"].notna(), 1.5 * df["Magnitude"] + 4.8, np.nan)
    df["Energy_Joules"] = np.where(df["log_Energy"].notna(), 10 ** df["log_Energy"], np.nan)
    return df


# ------------------------------------------------------------
# AI Injury Calibration (MLP + διαφορετικό multiplier ανά σενάριο)
# ------------------------------------------------------------
def ai_calibration(df: pd.DataFrame, scenario_name: str = "mild") -> pd.DataFrame:
    if "No. Injured" not in df.columns:
        print(" Δεν υπάρχει στήλη 'No. Injured' — παραλείπεται AI calibration.")
        return df

    hist = df.dropna(subset=["Magnitude", "No. Injured"]).copy()
    if len(hist) < 10:
        print(" Όχι αρκετά δεδομένα για AI calibration.")
        return df

    # Εκπαίδευση μικρού νευρωνικού δικτύου
    X = hist[["Magnitude"]]
    y = np.log1p(hist["No. Injured"])

    model = MLPRegressor(
        hidden_layer_sizes=(8, 4),
        activation="relu",
        solver="adam",
        max_iter=1000,
        learning_rate_init=0.01,
        random_state=42
    )
    model.fit(X, y)
    print(f" AI calibration (MLPRegressor) εκπαιδεύτηκε για σενάριο: {scenario_name}")

    # Πρόβλεψη
    df["AI_Predicted_Injuries"] = np.nan
    valid_mask = df["Magnitude"].notna()
    if valid_mask.any():
        X_valid = df.loc[valid_mask, ["Magnitude"]]
        df.loc[valid_mask, "AI_Predicted_Injuries"] = np.expm1(model.predict(X_valid))

    # Διαφορετικό multiplier ανά σενάριο
    if scenario_name.lower() == "extreme":
        INJURY_MULT = 1.7
    else:
        INJURY_MULT = 1.3

    df["AI_Predicted_Injuries"] = df["AI_Predicted_Injuries"] * INJURY_MULT

    # Αρνητικές τιμές σε 0
    df["AI_Predicted_Injuries"] = df["AI_Predicted_Injuries"].clip(lower=0)
    return df


# ------------------------------------------------------------
# Κύρια εκτέλεση
# ------------------------------------------------------------
def main():
    csv_candidates = [f for f in Path(".").glob("*.csv") if "metrics" not in f.name.lower()]
    if not csv_candidates:
        print(" Δεν βρέθηκε CSV αρχείο (χωρίς metrics). Βάλε το αρχικό dataset π.χ. Greece.csv.")
        return

    csv_path = str(csv_candidates[0])
    base_name = Path(csv_path).stem
    print(f" Είσοδος: {csv_path}")

    lam = estimate_lambda_from_csv(csv_path)
    print(f" Εκτιμώμενη λ (μέσος σεισμοί/έτος): {lam:.3f}")

    forecast_years = list(range(START_YEAR, END_YEAR + 1))
    raw = pd.read_csv(csv_path)
    output_sheets = {}

    for scenario_name, params in SCENARIOS.items():
        EXTREME_MULT = params["EXTREME_MULT"]
        ENERGY_COEFF = params["ENERGY_COEFF"]

        rng = random.Random(FIXED_SEED)
        counts = [poisson_knuth(lam * EXTREME_MULT, rng) for _ in forecast_years]

        df_future = pd.DataFrame({
            "Year": forecast_years,
            "Number_of_events": counts,
            "Scenario": scenario_name
        })

        df_all = pd.concat([raw, df_future], ignore_index=True)
        df_all = energy_adjusted_forecast(df_all, forecast_years=forecast_years,
                                          threshold=5.2, k=ENERGY_COEFF)
        df_all = ai_calibration(df_all, scenario_name=scenario_name)

        output_sheets[scenario_name] = df_all
        print(f"\n===== Σενάριο: {scenario_name.upper()} =====")
        print(df_all[df_all["Year"].isin(forecast_years)][["Year", "Number_of_events", "Predicted_Magnitude"]].head())

    # Εξαγωγή σε Excel
    out_xlsx = f"{base_name}_forecast_ai.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        for name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"\n Αποθηκεύτηκε: {out_xlsx}")
    print("Sheets: mild, extreme — περιλαμβάνει AI_Predicted_Injuries.")

    # Μετρικές
    metrics_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lambda_est": lam,
        "forecast_start": START_YEAR,
        "forecast_end": END_YEAR,
        "mild_events": sum(output_sheets["mild"]["Number_of_events"]),
        "extreme_events": sum(output_sheets["extreme"]["Number_of_events"]),
    }

    metrics_path = "forecast_metrics_log.csv"
    try:
        old = pd.read_csv(metrics_path)
        updated = pd.concat([old, pd.DataFrame([metrics_log])], ignore_index=True)
    except FileNotFoundError:
        updated = pd.DataFrame([metrics_log])

    updated.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n Οι μετρικές αποθηκεύτηκαν στο '{metrics_path}'")


# ------------------------------------------------------------
# Εκτέλεση
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
