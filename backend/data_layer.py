"""
==============================================================================
DATA LAYER - Phase 1: The Core Engine
==============================================================================
Enterprise-grade data ingestion, validation, phase-aware feature engineering,
Gaussian-noise data augmentation, and batch-level merge for the industrial AI
optimization prototype.

Author : Core Engine Team
Version: 1.0.0
==============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------─
# CONSTANTS
# ----------------------------------------------------------------------------─
DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")

PROCESS_FILE: str = "_h_batch_process_data.xlsx"
PRODUCTION_FILE: str = "_h_batch_production_data.xlsx"

# Ordered manufacturing phases
PHASES: List[str] = [
    "Preparation", "Granulation", "Drying", "Milling",
    "Blending", "Compression", "Coating", "Quality_Testing",
]

# Sensor columns for feature engineering
SENSOR_COLS: List[str] = [
    "Temperature_C", "Pressure_Bar", "Humidity_Percent",
    "Motor_Speed_RPM", "Compression_Force_kN", "Flow_Rate_LPM",
    "Power_Consumption_kW", "Vibration_mm_s",
]

# Physical sensor limits for anomaly detection  (min, max)
SENSOR_LIMITS: Dict[str, Tuple[float, float]] = {
    "Temperature_C":        (-10.0, 200.0),
    "Pressure_Bar":         (0.0,   50.0),
    "Humidity_Percent":     (0.0,  100.0),
    "Motor_Speed_RPM":      (0.0, 5000.0),
    "Compression_Force_kN": (0.0,  100.0),
    "Flow_Rate_LPM":        (0.0,  500.0),
    "Power_Consumption_kW": (0.0,  200.0),
    "Vibration_mm_s":       (0.0,   50.0),
}

# Gaussian noise standard-deviation fraction for data augmentation
NOISE_FRACTION: float = 0.08  # 8 % of each feature's magnitude


# ----------------------------------------------------------------------------─
# 1. INGESTION & VALIDATION
# ----------------------------------------------------------------------------─
def load_process_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the time-series process telemetry data.

    Parameters
    ----------
    filepath : str, optional
        Absolute path to the XLSX file.  Falls back to DATA_DIR default.

    Returns
    -------
    pd.DataFrame
        Validated process telemetry DataFrame.
    """
    path = filepath or os.path.join(DATA_DIR, PROCESS_FILE)
    print(f"[DATA LAYER] Loading process data from: {path}")
    df = pd.read_excel(path)

    # -- Schema check ----------------------------------------------------
    required = {"Batch_ID", "Time_Minutes", "Phase"} | set(SENSOR_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Process data missing columns: {missing}")

    # -- Null handling --------------------------------------------------─
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  [WARNING] Filling {null_counts.sum()} null values via forward-fill + zero")
        df = df.ffill().fillna(0)

    # -- Anomaly detection: IQR + physical limits ------------------------
    anomalies_total = 0
    for col in SENSOR_COLS:
        lo, hi = SENSOR_LIMITS[col]
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = mask.sum()
        if n_bad:
            print(f"  [WARNING] {col}: {n_bad} readings outside [{lo}, {hi}] - clipping")
            df[col] = df[col].clip(lo, hi)
            anomalies_total += n_bad

    # IQR-based outlier flagging (soft - log only, no removal)
    for col in SENSOR_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outlier:
            print(f"  [INFO] {col}: {n_outlier} IQR outliers detected (kept)")

    print(f"  [OK] Process data validated - {len(df)} rows, {df['Batch_ID'].nunique()} batch(es)")
    return df


def load_production_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and validate batch-summary production data.

    Parameters
    ----------
    filepath : str, optional
        Absolute path to the XLSX file.

    Returns
    -------
    pd.DataFrame
        Validated production summary DataFrame.
    """
    path = filepath or os.path.join(DATA_DIR, PRODUCTION_FILE)
    print(f"[DATA LAYER] Loading production data from: {path}")
    df = pd.read_excel(path)

    required = {
        "Batch_ID", "Granulation_Time", "Binder_Amount", "Drying_Temp",
        "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc",
        "Moisture_Content", "Tablet_Weight", "Hardness", "Friability",
        "Disintegration_Time", "Dissolution_Rate", "Content_Uniformity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Production data missing columns: {missing}")

    # Null handling
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  [WARNING] Filling {null_counts.sum()} null values with column medians")
        df = df.fillna(df.median(numeric_only=True))

    # Duplicate Batch_ID check
    dupes = df[df["Batch_ID"].duplicated()]
    if len(dupes):
        print(f"  [WARNING] Dropping {len(dupes)} duplicate Batch_ID entries")
        df = df.drop_duplicates(subset=["Batch_ID"], keep="first")

    print(f"  [OK] Production data validated - {len(df)} batches")
    return df


# ----------------------------------------------------------------------------─
# 2. PHASE-AWARE FEATURE ENGINEERING
# ----------------------------------------------------------------------------─
def _thermal_ramp_rate(phase_df: pd.DataFrame) -> float:
    """Calculate Thermal Ramp Rate (°C/min) for a phase segment.

    Uses the first-order finite difference of Temperature_C over Time_Minutes.
    Returns the mean absolute ramp rate across the phase window.
    """
    if len(phase_df) < 2:
        return 0.0
    dt = np.diff(phase_df["Time_Minutes"].values).astype(float)
    d_temp = np.diff(phase_df["Temperature_C"].values)
    # Avoid division by zero for duplicate timestamps
    valid = dt > 0
    if not valid.any():
        return 0.0
    rates = d_temp[valid] / dt[valid]
    return float(np.mean(np.abs(rates)))


def _auc_trapz(phase_df: pd.DataFrame, col: str) -> float:
    """Calculate Area Under the Curve using trapezoidal integration.

    Parameters
    ----------
    phase_df : pd.DataFrame
        Slice of telemetry within one phase.
    col : str
        Column name (e.g. 'Power_Consumption_kW').

    Returns
    -------
    float
        AUC value  (unit·minutes).
    """
    if len(phase_df) < 2:
        return 0.0
    # np.trapezoid is the NumPy 2.x name; fall back to np.trapz for older versions
    _integrate = getattr(np, "trapezoid", np.trapz)
    return float(_integrate(
        phase_df[col].values,
        phase_df["Time_Minutes"].values,
    ))


def _time_above_threshold(phase_df: pd.DataFrame, col: str, threshold: float) -> float:
    """Calculate minutes spent with sensor value above a threshold."""
    if len(phase_df) < 2:
        return 0.0
    dt = np.diff(phase_df["Time_Minutes"].values).astype(float)
    above = phase_df[col].values[:-1] > threshold
    return float(np.sum(dt[above]))

def _peak_to_mean_ratio(phase_df: pd.DataFrame, col: str) -> float:
    """Calculate ratio of maximum value to mean value."""
    if len(phase_df) == 0:
        return 1.0
    mean_val = phase_df[col].mean()
    if mean_val == 0:
        return 1.0
    return float(phase_df[col].max() / mean_val)

def _relative_time_of_max(phase_df: pd.DataFrame, col: str, batch_start: float, batch_end: float) -> float:
    """Calculate when the max value occurred as a fraction of total batch duration."""
    if len(phase_df) == 0 or batch_end <= batch_start:
        return 0.0
    max_idx = phase_df[col].idxmax()
    if pd.isna(max_idx):
        return 0.0
    max_time = phase_df.loc[max_idx, "Time_Minutes"]
    return float((max_time - batch_start) / (batch_end - batch_start))

def extract_phase_features(process_df: pd.DataFrame) -> pd.DataFrame:
    """Extract advanced features per phase per batch from telemetry data.

    For each phase, computes:
      - Statistical aggregates: mean, std, min, max of every sensor column
      - Thermal Ramp Rate  (°C / min)
      - Power AUC  (kW·min)
      - Vibration AUC  (mm/s·min)

    Parameters
    ----------
    process_df : pd.DataFrame
        Raw process telemetry (should contain 'Batch_ID', 'Phase', etc.).

    Returns
    -------
    pd.DataFrame
        One row per Batch_ID with all phase-level features as columns.
        Column naming convention: ``{Phase}_{Sensor}_{Stat}``
    """
    records: List[Dict[str, object]] = []

    for batch_id, batch_df in process_df.groupby("Batch_ID"):
        row: Dict[str, object] = {"Batch_ID": batch_id}
        batch_start: float = float(batch_df["Time_Minutes"].min())
        batch_end: float = float(batch_df["Time_Minutes"].max())

        for phase in PHASES:
            pf = batch_df[batch_df["Phase"] == phase].sort_values("Time_Minutes")

            # -- Statistical aggregates ----------------------------------
            for sensor in SENSOR_COLS:
                prefix = f"{phase}_{sensor}"
                if len(pf) > 0:
                    row[f"{prefix}_mean"] = pf[sensor].mean()
                    row[f"{prefix}_std"]  = pf[sensor].std(ddof=0)
                    row[f"{prefix}_min"]  = pf[sensor].min()
                    row[f"{prefix}_max"]  = pf[sensor].max()
                else:
                    row[f"{prefix}_mean"] = 0.0
                    row[f"{prefix}_std"]  = 0.0
                    row[f"{prefix}_min"]  = 0.0
                    row[f"{prefix}_max"]  = 0.0

            # -- Advanced features --------------------------------------─
            row[f"{phase}_Thermal_Ramp_Rate"] = _thermal_ramp_rate(pf)
            row[f"{phase}_Power_AUC"]         = _auc_trapz(pf, "Power_Consumption_kW")
            row[f"{phase}_Vibration_AUC"]     = _auc_trapz(pf, "Vibration_mm_s")

            # -- Temporal Dynamics --------------------------------------─
            row[f"{phase}_Power_Peak_Mean_Ratio"] = _peak_to_mean_ratio(pf, "Power_Consumption_kW")
            row[f"{phase}_Max_Vibration_Relative_Time"] = _relative_time_of_max(pf, "Vibration_mm_s", batch_start, batch_end)
            if phase == "Granulation":
                row[f"{phase}_High_Humidity_Time_Mins"] = _time_above_threshold(pf, "Humidity_Percent", 50.0)

        records.append(row)

    features_df = pd.DataFrame(records)
    print(f"[DATA LAYER] Extracted {features_df.shape[1] - 1} features across "
          f"{len(PHASES)} phases for {len(features_df)} batch(es)")
    return features_df


# ----------------------------------------------------------------------------─
# 3. DATA AUGMENTATION  (SMOTE-Style Bounded Interpolation)
# ----------------------------------------------------------------------------─
def augment_telemetry_features(
    baseline_row: pd.Series,
    production_df: pd.DataFrame,
    noise_fraction: float = NOISE_FRACTION,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthesize varied telemetry features using SMOTE-style interpolation.

    Why this is necessary:
      The process telemetry file contains data for only 1 batch (T001).
      Duplicating identical features across 60 batches would cause
      **feature collapse** - the surrogate model cannot learn if all
      input features are constant while outputs vary.

    Strategy (SMOTE-style bounded augmentation):
      1. Use T001's extracted feature vector as the baseline.
      2. Create initial diverse "seed" rows via production-setting-correlated
         perturbations so we have real anchors to interpolate between.
      3. For each batch, pick two random seed rows and create a convex
         combination:  synthetic = row_a + alpha * (row_b - row_a)
         where alpha ∈ [0, 1].
      4. Clip every feature to the observed [min, max] across all seeds,
         using the 10th–90th percentile as the primary safe zone with the
         absolute min/max as hard bounds. This ensures physically plausible
         sensor values.

    Parameters
    ----------
    baseline_row : pd.Series
        Feature vector for the reference batch (T001).
    production_df : pd.DataFrame
        Production summary data for all batches.
    noise_fraction : float
        Standard deviation of initial perturbation for seed creation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Augmented feature matrix - one row per batch in production_df.
    """
    rng = np.random.default_rng(seed)
    all_batch_ids = production_df["Batch_ID"].values
    n_batches = len(all_batch_ids)

    # Separate Batch_ID from numeric features
    feature_cols = [c for c in baseline_row.index if c != "Batch_ID"]
    baseline_values = baseline_row[feature_cols].values.astype(float)
    n_features = len(feature_cols)

    # -- STEP 1: Create diverse "seed" rows via correlated perturbation --
    n_seeds = min(n_batches, 10)  # Use up to 10 anchor points
    seed_matrix = np.tile(baseline_values, (n_seeds, 1))

    # Add bounded Gaussian noise to seeds, correlated with production settings
    for s in range(n_seeds):
        noise = rng.normal(0.0, noise_fraction, size=n_features)
        seed_matrix[s, :] = baseline_values * (1.0 + noise)

    # Correlate seeds with production settings for physical plausibility
    if "Drying_Temp" in production_df.columns:
        drying_temp_vals = production_df["Drying_Temp"].values[:n_seeds].astype(float)
        drying_temp_median = np.median(production_df["Drying_Temp"].values.astype(float))
        drying_temp_ratio = drying_temp_vals / (drying_temp_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Drying_") and "Temperature" in col:
                seed_matrix[:len(drying_temp_ratio), j] *= drying_temp_ratio

    if "Machine_Speed" in production_df.columns:
        speed_vals = production_df["Machine_Speed"].values[:n_seeds].astype(float)
        speed_median = np.median(production_df["Machine_Speed"].values.astype(float))
        speed_ratio = speed_vals / (speed_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Compression_") and ("Motor" in col or "Vibration" in col):
                seed_matrix[:len(speed_ratio), j] *= speed_ratio

    if "Compression_Force" in production_df.columns:
        force_vals = production_df["Compression_Force"].values[:n_seeds].astype(float)
        force_median = np.median(production_df["Compression_Force"].values.astype(float))
        force_ratio = force_vals / (force_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Compression_") and "Force" in col:
                seed_matrix[:len(force_ratio), j] *= force_ratio

    # Ensure non-negative seeds
    seed_matrix = np.maximum(seed_matrix, 0.0)

    # -- STEP 2: Compute bounding ranges from seeds ----------------------
    feat_min = np.minimum(seed_matrix.min(axis=0), baseline_values * 0.7)
    feat_max = np.maximum(seed_matrix.max(axis=0), baseline_values * 1.3)
    # 10th–90th percentile safe zone (softer bounds for interpolation)
    if n_seeds >= 3:
        feat_p10 = np.percentile(seed_matrix, 10, axis=0)
        feat_p90 = np.percentile(seed_matrix, 90, axis=0)
    else:
        feat_p10 = feat_min
        feat_p90 = feat_max

    # -- STEP 3: SMOTE-style interpolation for all batches --------------─
    augmented_matrix = np.zeros((n_batches, n_features))

    for i in range(n_batches):
        # Pick two random seeds and interpolate
        idx_a, idx_b = rng.choice(n_seeds, size=2, replace=False)
        alpha = rng.uniform(0.0, 1.0)
        synthetic = seed_matrix[idx_a] + alpha * (seed_matrix[idx_b] - seed_matrix[idx_a])

        # Add tiny jitter to break ties (within 2% of feature magnitude)
        jitter = rng.normal(0.0, 0.02, size=n_features) * np.abs(baseline_values + 1e-8)
        synthetic += jitter

        # Clip to hard physical bounds [feat_min, feat_max]
        synthetic = np.clip(synthetic, feat_min, feat_max)

        augmented_matrix[i, :] = synthetic

    # -- Production-setting correlation pass (for ALL batches) ----------─
    if "Drying_Temp" in production_df.columns:
        drying_temp_all = production_df["Drying_Temp"].values.astype(float)
        drying_temp_median = np.median(drying_temp_all)
        drying_temp_ratio = drying_temp_all / (drying_temp_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Drying_") and "Temperature" in col:
                augmented_matrix[:, j] *= drying_temp_ratio

    if "Machine_Speed" in production_df.columns:
        speed_all = production_df["Machine_Speed"].values.astype(float)
        speed_median = np.median(speed_all)
        speed_ratio = speed_all / (speed_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Compression_") and ("Motor" in col or "Vibration" in col):
                augmented_matrix[:, j] *= speed_ratio

    if "Compression_Force" in production_df.columns:
        force_all = production_df["Compression_Force"].values.astype(float)
        force_median = np.median(force_all)
        force_ratio = force_all / (force_median + 1e-8)
        for j, col in enumerate(feature_cols):
            if col.startswith("Compression_") and "Force" in col:
                augmented_matrix[:, j] *= force_ratio

    # -- Row 0 = T001 keeps its exact baseline (no noise) --------------─
    t001_idx = np.where(all_batch_ids == "T001")[0]
    if len(t001_idx) > 0:
        augmented_matrix[t001_idx[0], :] = baseline_values

    # -- Final bounding: non-negative + hard max ------------------------─
    augmented_matrix = np.maximum(augmented_matrix, 0.0)
    augmented_matrix = np.clip(augmented_matrix, feat_min, feat_max)

    augmented_df = pd.DataFrame(augmented_matrix, columns=feature_cols)
    augmented_df.insert(0, "Batch_ID", all_batch_ids)

    print(f"[DATA LAYER] SMOTE-style augmented features for {n_batches} batches "
          f"(bounded to observed min/max, {n_seeds} seed anchors)")

    # -- Verify variance (feature collapse check) ------------------------
    numeric_cols = augmented_df.select_dtypes(include=[np.number]).columns
    zero_var = (augmented_df[numeric_cols].std() == 0).sum()
    if zero_var > 0:
        print(f"  [WARNING] WARNING: {zero_var} features still have zero variance!")
    else:
        print(f"  [OK] All {len(numeric_cols)} numeric features have non-zero variance")

    return augmented_df


# ----------------------------------------------------------------------------─
# 4. MERGE - Build final training dataset
# ----------------------------------------------------------------------------─
def build_training_dataset(
    process_df: Optional[pd.DataFrame] = None,
    production_df: Optional[pd.DataFrame] = None,
    noise_fraction: float = NOISE_FRACTION,
) -> pd.DataFrame:
    """End-to-end pipeline: ingest -> validate -> engineer -> augment -> merge.

    Parameters
    ----------
    process_df : pd.DataFrame, optional
        Pre-loaded process telemetry.  If None, loads from disk.
    production_df : pd.DataFrame, optional
        Pre-loaded production summary.  If None, loads from disk.
    noise_fraction : float
        Passed through to augmentation.

    Returns
    -------
    pd.DataFrame
        Merged training dataset with production features, quality targets,
        and augmented phase-level telemetry features.
    """
    # -- Load ------------------------------------------------------------
    if process_df is None:
        process_df = load_process_data()
    if production_df is None:
        production_df = load_production_data()

    # -- Extract phase-level features from T001 --------------------------
    phase_features = extract_phase_features(process_df)
    baseline_row = phase_features.iloc[0]  # T001's feature vector

    # -- Augment: produce varied rows for all 60 batches ----------------─
    augmented_features = augment_telemetry_features(
        baseline_row, production_df, noise_fraction=noise_fraction
    )

    # -- Merge on Batch_ID ----------------------------------------------─
    merged = production_df.merge(augmented_features, on="Batch_ID", how="left")

    # -- Final quality check --------------------------------------------─
    null_after_merge = merged.isnull().sum().sum()
    if null_after_merge:
        print(f"  [WARNING] {null_after_merge} nulls after merge - filling with 0")
        merged = merged.fillna(0)

    print(f"\n{'=' * 60}")
    print(f"[DATA LAYER] Training dataset ready:")
    print(f"  Rows    : {merged.shape[0]}")
    print(f"  Columns : {merged.shape[1]}")
    print(f"  Features: {merged.shape[1] - 1} (excl. Batch_ID)")
    print(f"{'=' * 60}\n")

    return merged


# ----------------------------------------------------------------------------─
# CLI ENTRY POINT
# ----------------------------------------------------------------------------─
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        DATA LAYER - Phase 1: Core Engine                ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    dataset = build_training_dataset()

    # -- Show sample output ----------------------------------------------
    print("Sample (first 5 rows, first 10 cols):")
    print(dataset.iloc[:5, :10].to_string(index=False))

    # -- Save for downstream consumption --------------------------------─
    output_path = os.path.join(DATA_DIR, "training_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"\n[OK] Saved training dataset to: {output_path}")

    # -- Variance report (proof that augmentation prevents collapse) ----─
    numeric = dataset.select_dtypes(include=[np.number])
    zero_var_cols = numeric.columns[numeric.std() == 0].tolist()
    if zero_var_cols:
        print(f"\n[WARNING] Zero-variance columns: {zero_var_cols}")
    else:
        print(f"\n[OK] No zero-variance columns - feature collapse prevented!")
