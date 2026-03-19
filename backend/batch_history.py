"""
==============================================================================
BATCH HISTORY — Persistent Batch Performance Tracking
==============================================================================
Stores all batch execution records in a JSON file for historical metric
tracking, performance trend analysis, and continuous improvement evidence.

Author : Core Engine Team
Version: 2.0.0
==============================================================================
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")
HISTORY_FILE = os.path.join(DATA_DIR, "batch_history.json")


class BatchHistoryStore:
    """Persistent store for batch execution history using a JSON file."""

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = filepath or HISTORY_FILE
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing history from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._records = []

    def _save(self) -> None:
        """Persist history to disk."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, default=str)

    def add_batch(
        self,
        batch_id: str,
        proposed_settings: Dict[str, Any],
        simulated_outcome: Dict[str, Any],
        quality_delta: float,
        qdrant_updated: bool,
        human_approved: bool,
        human_feedback: str = "",
        carbon_metrics: Optional[Dict[str, Any]] = None,
        energy_anomalies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Add a completed batch record to history.

        Returns
        -------
        Dict[str, Any]
            The stored record.
        """
        record = {
            "batch_id":           batch_id,
            "timestamp":          time.time(),
            "timestamp_iso":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "proposed_settings":  proposed_settings,
            "simulated_outcome":  simulated_outcome,
            "quality_delta":      round(quality_delta, 6),
            "qdrant_updated":     qdrant_updated,
            "human_approved":     human_approved,
            "human_feedback":     human_feedback,
            "carbon_metrics":     carbon_metrics or {},
            "energy_anomalies":   energy_anomalies or [],
        }
        self._records.append(record)
        self._save()
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all batch records, newest first."""
        return list(reversed(self._records))

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific batch record by ID."""
        for r in self._records:
            if r["batch_id"] == batch_id:
                return r
        return None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all batches."""
        n = len(self._records)
        if n == 0:
            return {
                "total_batches": 0,
                "avg_quality_delta": 0.0,
                "positive_improvements": 0,
                "qdrant_updates": 0,
                "approval_rate": 0.0,
                "total_carbon_kg": 0.0,
                "avg_carbon_per_batch": 0.0,
            }

        deltas = [r["quality_delta"] for r in self._records]
        positive = sum(1 for d in deltas if d > 0)
        qdrant_ups = sum(1 for r in self._records if r["qdrant_updated"])
        approvals = sum(1 for r in self._records if r["human_approved"])

        total_carbon = sum(
            r.get("carbon_metrics", {}).get("carbon_kg", 0.0)
            for r in self._records
        )

        return {
            "total_batches":         n,
            "avg_quality_delta":     round(sum(deltas) / n, 6),
            "best_quality_delta":    round(max(deltas), 6),
            "worst_quality_delta":   round(min(deltas), 6),
            "positive_improvements": positive,
            "improvement_rate":      round(positive / n * 100, 1),
            "qdrant_updates":        qdrant_ups,
            "approval_rate":         round(approvals / n * 100, 1),
            "total_carbon_kg":       round(total_carbon, 3),
            "avg_carbon_per_batch":  round(total_carbon / n, 3),
            "recent_trend":          deltas[-min(5, n):],
        }
