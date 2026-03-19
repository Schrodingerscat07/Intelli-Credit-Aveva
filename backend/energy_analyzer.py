"""
==============================================================================
ENERGY ANALYZER — Power Consumption Pattern Analysis
==============================================================================
Compares per-phase energy consumption patterns against golden signature
baselines to detect anomalies, predict asset health issues, and provide
process reliability insights.

Author : Core Engine Team
Version: 2.0.0
==============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Manufacturing phases (must match data_layer.py)
PHASES = [
    "Preparation", "Granulation", "Drying", "Milling",
    "Blending", "Compression", "Coating", "Quality_Testing",
]

# Metrics to analyze per phase
ENERGY_METRICS = ["Power_Consumption_kW", "Vibration_mm_s", "Temperature_C"]

# Deviation thresholds
THRESHOLD_WARNING = 0.15   # 15% deviation → WARNING
THRESHOLD_CRITICAL = 0.30  # 30% deviation → CRITICAL

# Phase-to-asset mapping for maintenance recommendations
PHASE_ASSET_MAP = {
    "Preparation":     "Mixing Unit / Pre-Processing Station",
    "Granulation":     "Granulator Motor / Binder Pump",
    "Drying":          "Drying Chamber / Heating Element",
    "Milling":         "Milling Machine / Blade Assembly",
    "Blending":        "Blender Motor / Agitator Bearings",
    "Compression":     "Compression Press / Force Actuator",
    "Coating":         "Coating Drum / Spray Nozzle System",
    "Quality_Testing": "QC Instruments / Sensor Calibration",
}


class EnergyPatternAnalyzer:
    """Analyzes energy consumption patterns for asset and process reliability."""

    def analyze_patterns(
        self,
        current_telemetry: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare current batch energy patterns against golden baseline.

        Parameters
        ----------
        current_telemetry : Dict
            Current batch telemetry features.
        baseline : Dict
            Golden Signature baseline from Qdrant.

        Returns
        -------
        Dict with anomalies list, asset health score, and recommendations.
        """
        anomalies: List[Dict[str, Any]] = []
        phase_scores: List[float] = []

        for phase in PHASES:
            phase_anomalies = self._analyze_phase(
                phase, current_telemetry, baseline
            )
            anomalies.extend(phase_anomalies)

            # Phase health score = 1.0 - max deviation in this phase
            if phase_anomalies:
                max_dev = max(abs(a["deviation_pct"]) for a in phase_anomalies)
                phase_scores.append(max(0.0, 1.0 - max_dev))
            else:
                phase_scores.append(1.0)

        # Overall asset health score (0-100)
        asset_health_score = round(
            (sum(phase_scores) / max(len(phase_scores), 1)) * 100, 1
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies)

        return {
            "anomalies":          anomalies,
            "anomaly_count":      len(anomalies),
            "asset_health_score": asset_health_score,
            "phase_scores":       {
                phase: round(score * 100, 1)
                for phase, score in zip(PHASES, phase_scores)
            },
            "recommendations":    recommendations,
            "severity_summary":   {
                "NORMAL":   sum(1 for a in anomalies if a["severity"] == "NORMAL"),
                "WARNING":  sum(1 for a in anomalies if a["severity"] == "WARNING"),
                "CRITICAL": sum(1 for a in anomalies if a["severity"] == "CRITICAL"),
            },
        }

    def _analyze_phase(
        self,
        phase: str,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze energy patterns for a single manufacturing phase."""
        anomalies: List[Dict[str, Any]] = []

        # Check Power AUC for this phase
        power_key = f"ctx_{phase}_Power_AUC"
        self._check_metric(
            anomalies, phase, "Power_AUC", "kW·min",
            current, baseline, power_key,
        )

        # Check Vibration AUC
        vib_key = f"ctx_{phase}_Vibration_AUC"
        self._check_metric(
            anomalies, phase, "Vibration_AUC", "mm/s·min",
            current, baseline, vib_key,
        )

        # Check Thermal Ramp Rate
        thermal_key = f"ctx_{phase}_Thermal_Ramp_Rate"
        self._check_metric(
            anomalies, phase, "Thermal_Ramp_Rate", "°C/min",
            current, baseline, thermal_key,
        )

        # Check mean power consumption
        power_mean_key = f"ctx_{phase}_Power_Consumption_kW_mean"
        self._check_metric(
            anomalies, phase, "Power_Mean", "kW",
            current, baseline, power_mean_key,
        )

        # Check mean vibration
        vib_mean_key = f"ctx_{phase}_Vibration_mm_s_mean"
        self._check_metric(
            anomalies, phase, "Vibration_Mean", "mm/s",
            current, baseline, vib_mean_key,
        )

        return anomalies

    def _check_metric(
        self,
        anomalies: List[Dict[str, Any]],
        phase: str,
        metric_name: str,
        unit: str,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
        key: str,
    ) -> None:
        """Check a single metric for deviation from baseline."""
        # Try to find the value in current telemetry or baseline
        curr_val = current.get(key, None)
        base_val = baseline.get(key, None)

        if curr_val is None or base_val is None:
            return

        try:
            curr_val = float(curr_val)
            base_val = float(base_val)
        except (TypeError, ValueError):
            return

        if abs(base_val) < 1e-8:
            return

        deviation = (curr_val - base_val) / abs(base_val)
        abs_dev = abs(deviation)

        if abs_dev >= THRESHOLD_WARNING:
            severity = "CRITICAL" if abs_dev >= THRESHOLD_CRITICAL else "WARNING"
            direction = "higher" if deviation > 0 else "lower"

            anomalies.append({
                "phase":          phase,
                "metric":         metric_name,
                "unit":           unit,
                "current_value":  round(curr_val, 3),
                "baseline_value": round(base_val, 3),
                "deviation_pct":  round(deviation * 100, 1),
                "direction":      direction,
                "severity":       severity,
                "asset":          PHASE_ASSET_MAP.get(phase, "Unknown"),
            })

    def _generate_recommendations(
        self, anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations from detected anomalies."""
        recommendations: List[str] = []
        seen_phases = set()

        critical = [a for a in anomalies if a["severity"] == "CRITICAL"]
        warnings = [a for a in anomalies if a["severity"] == "WARNING"]

        for a in critical:
            phase = a["phase"]
            if phase not in seen_phases:
                seen_phases.add(phase)
                if "Power" in a["metric"] and a["direction"] == "higher":
                    recommendations.append(
                        f"URGENT: {phase} phase shows {abs(a['deviation_pct']):.0f}% "
                        f"higher power consumption — inspect {a['asset']} for "
                        f"degradation or calibration drift"
                    )
                elif "Vibration" in a["metric"] and a["direction"] == "higher":
                    recommendations.append(
                        f"URGENT: {phase} phase vibration {abs(a['deviation_pct']):.0f}% "
                        f"above normal — schedule predictive maintenance for {a['asset']}"
                    )
                elif "Thermal" in a["metric"]:
                    recommendations.append(
                        f"URGENT: {phase} thermal ramp rate deviated by "
                        f"{abs(a['deviation_pct']):.0f}% — check heating element "
                        f"efficiency in {a['asset']}"
                    )

        for a in warnings:
            phase = a["phase"]
            if phase not in seen_phases:
                seen_phases.add(phase)
                recommendations.append(
                    f"MONITOR: {phase} phase {a['metric']} is "
                    f"{abs(a['deviation_pct']):.0f}% {a['direction']} than baseline "
                    f"— track {a['asset']} over next 5 batches"
                )

        if not recommendations:
            recommendations.append(
                "All energy patterns within normal operating range. "
                "No maintenance actions required."
            )

        return recommendations
