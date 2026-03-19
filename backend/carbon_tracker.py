"""
==============================================================================
CARBON TRACKER — Real Carbon Emission Tracking
==============================================================================
Calculates per-batch and cumulative CO₂ emissions from power consumption data.
Uses India grid emission factor as default.

Author : Core Engine Team
Version: 2.0.0
==============================================================================
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# EMISSION FACTORS (kgCO₂ per kWh)
# ─────────────────────────────────────────────────────────────────────────────
EMISSION_FACTORS: Dict[str, float] = {
    "india_grid":       0.82,   # India weighted avg (CEA 2023)
    "eu_grid":          0.23,   # EU-27 average
    "us_grid":          0.39,   # US EPA eGRID average
    "renewable":        0.02,   # Near-zero for wind/solar
    "natural_gas":      0.45,   # Combined cycle gas turbine
}

# Default regulatory benchmarks
DEFAULT_REGULATORY = {
    "max_carbon_per_batch_kg":    25.0,    # kgCO₂ cap per batch
    "max_power_per_batch_kwh":    50.0,    # kWh cap per batch
    "annual_carbon_budget_tons":  500.0,   # Annual organisational target
}

# Default batch duration estimate (hours) for pharma manufacturing
DEFAULT_BATCH_DURATION_HOURS = 1.5


def calculate_carbon(
    power_kw: float,
    duration_hours: float = DEFAULT_BATCH_DURATION_HOURS,
    emission_factor: float = EMISSION_FACTORS["india_grid"],
) -> Dict[str, float]:
    """Calculate carbon emissions for a single batch.

    Parameters
    ----------
    power_kw : float
        Average power consumption during the batch (kW).
    duration_hours : float
        Batch duration in hours.
    emission_factor : float
        kgCO₂ per kWh.

    Returns
    -------
    Dict[str, float]
        Detailed carbon metrics.
    """
    energy_kwh = power_kw * duration_hours
    carbon_kg = energy_kwh * emission_factor

    return {
        "power_kw":          round(power_kw, 2),
        "duration_hours":    round(duration_hours, 2),
        "energy_kwh":        round(energy_kwh, 2),
        "emission_factor":   emission_factor,
        "carbon_kg":         round(carbon_kg, 3),
        "carbon_intensity":  round(carbon_kg / max(energy_kwh, 1e-8), 4),
    }


class CarbonTracker:
    """Tracks cumulative and per-batch carbon emissions across the system."""

    def __init__(
        self,
        emission_factor: float = EMISSION_FACTORS["india_grid"],
        regulatory_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        self.emission_factor = emission_factor
        self.regulatory = regulatory_limits or DEFAULT_REGULATORY.copy()
        self._batch_records: List[Dict[str, Any]] = []
        self._cumulative_carbon_kg: float = 0.0
        self._cumulative_energy_kwh: float = 0.0

    def track_batch(
        self,
        batch_id: str,
        power_kw: float,
        duration_hours: float = DEFAULT_BATCH_DURATION_HOURS,
    ) -> Dict[str, Any]:
        """Track carbon for a completed batch.

        Returns
        -------
        Dict with carbon metrics + regulatory compliance status.
        """
        metrics = calculate_carbon(power_kw, duration_hours, self.emission_factor)

        self._cumulative_carbon_kg += metrics["carbon_kg"]
        self._cumulative_energy_kwh += metrics["energy_kwh"]

        # Regulatory compliance check
        exceeds_batch_carbon = metrics["carbon_kg"] > self.regulatory["max_carbon_per_batch_kg"]
        exceeds_batch_power = metrics["energy_kwh"] > self.regulatory["max_power_per_batch_kwh"]

        record = {
            "batch_id":              batch_id,
            **metrics,
            "cumulative_carbon_kg":  round(self._cumulative_carbon_kg, 3),
            "cumulative_energy_kwh": round(self._cumulative_energy_kwh, 2),
            "exceeds_carbon_limit":  exceeds_batch_carbon,
            "exceeds_power_limit":   exceeds_batch_power,
            "carbon_limit_kg":       self.regulatory["max_carbon_per_batch_kg"],
            "power_limit_kwh":       self.regulatory["max_power_per_batch_kwh"],
        }
        self._batch_records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Get cumulative carbon tracking summary."""
        n = len(self._batch_records)
        return {
            "total_batches":         n,
            "cumulative_carbon_kg":  round(self._cumulative_carbon_kg, 3),
            "cumulative_energy_kwh": round(self._cumulative_energy_kwh, 2),
            "avg_carbon_per_batch":  round(self._cumulative_carbon_kg / max(n, 1), 3),
            "avg_energy_per_batch":  round(self._cumulative_energy_kwh / max(n, 1), 2),
            "emission_factor":       self.emission_factor,
            "regulatory_limits":     self.regulatory,
            "batches_exceeding_carbon": sum(
                1 for r in self._batch_records if r["exceeds_carbon_limit"]
            ),
        }

    def update_regulatory(self, new_limits: Dict[str, float]) -> None:
        """Update regulatory limits."""
        self.regulatory.update(new_limits)

    def update_emission_factor(self, factor_name: str) -> bool:
        """Switch emission factor by name."""
        if factor_name in EMISSION_FACTORS:
            self.emission_factor = EMISSION_FACTORS[factor_name]
            return True
        return False
