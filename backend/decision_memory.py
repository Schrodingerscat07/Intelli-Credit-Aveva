"""
==============================================================================
DECISION MEMORY — Operator Decision Persistence & Reuse
==============================================================================
Stores all HITL decisions (approve/reject + feedback) and finds similar past
decisions to warn operators about previously rejected settings.

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
DECISION_FILE = os.path.join(DATA_DIR, "decision_log.json")


class DecisionMemory:
    """Persistent memory of operator HITL decisions for learning and reuse."""

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = filepath or DECISION_FILE
        self._decisions: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing decisions from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._decisions = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._decisions = []

    def _save(self) -> None:
        """Persist decisions to disk."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._decisions, f, indent=2, default=str)

    def log_decision(
        self,
        batch_id: str,
        proposed_settings: Dict[str, float],
        approved: bool,
        feedback: str = "",
        quality_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """Record an operator's HITL decision.

        Parameters
        ----------
        batch_id : str
            The batch that was being reviewed.
        proposed_settings : Dict[str, float]
            The machine settings that were proposed.
        approved : bool
            Whether the operator approved.
        feedback : str
            Operator's textual feedback.
        quality_delta : float
            Resulting quality improvement (0 if rejected).

        Returns
        -------
        Dict[str, Any]
            The stored decision record.
        """
        record = {
            "batch_id":          batch_id,
            "timestamp":         time.time(),
            "timestamp_iso":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "proposed_settings": proposed_settings,
            "approved":          approved,
            "feedback":          feedback,
            "quality_delta":     round(quality_delta, 6),
        }
        self._decisions.append(record)
        self._save()
        return record

    def find_similar_decisions(
        self,
        current_settings: Dict[str, float],
        similarity_threshold: float = 0.15,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find past decisions with similar proposed settings.

        A decision is "similar" if all settings are within
        `similarity_threshold` (fraction) of the current settings.

        Parameters
        ----------
        current_settings : Dict[str, float]
            The new proposed settings to compare against.
        similarity_threshold : float
            Maximum fractional deviation per variable.
        top_k : int
            Maximum similar decisions to return.

        Returns
        -------
        List of similar past decision records, with a `similarity_score`.
        """
        similar: List[Dict[str, Any]] = []

        for decision in self._decisions:
            past_settings = decision.get("proposed_settings", {})
            if not past_settings:
                continue

            # Compute per-variable similarity
            total_dev = 0.0
            n_vars = 0
            all_similar = True

            for var, curr_val in current_settings.items():
                past_val = past_settings.get(var)
                if past_val is None:
                    continue
                try:
                    curr_val = float(curr_val)
                    past_val = float(past_val)
                except (TypeError, ValueError):
                    continue

                if abs(curr_val) < 1e-8:
                    continue

                deviation = abs(curr_val - past_val) / abs(curr_val)
                total_dev += deviation
                n_vars += 1

                if deviation > similarity_threshold:
                    all_similar = False

            if all_similar and n_vars > 0:
                avg_dev = total_dev / n_vars
                similarity_score = round(max(0, 1.0 - avg_dev) * 100, 1)
                similar.append({
                    **decision,
                    "similarity_score": similarity_score,
                })

        # Sort by most similar first
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar[:top_k]

    def get_warnings(
        self, current_settings: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """Generate warnings about previously rejected similar settings.

        Returns
        -------
        List of warning dicts with `message`, `batch_id`, `feedback`.
        """
        similar = self.find_similar_decisions(current_settings)
        warnings: List[Dict[str, str]] = []

        for decision in similar:
            if not decision["approved"]:
                warnings.append({
                    "message": (
                        f"Similar settings were REJECTED for batch "
                        f"{decision['batch_id']} on "
                        f"{decision['timestamp_iso']}."
                    ),
                    "batch_id":          decision["batch_id"],
                    "feedback":          decision.get("feedback", ""),
                    "similarity_score":  decision["similarity_score"],
                })

        return warnings

    def get_all_decisions(self) -> List[Dict[str, Any]]:
        """Return all decision records, newest first."""
        return list(reversed(self._decisions))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about operator decisions."""
        n = len(self._decisions)
        if n == 0:
            return {"total": 0, "approved": 0, "rejected": 0, "approval_rate": 0.0}
        approved = sum(1 for d in self._decisions if d["approved"])
        return {
            "total":          n,
            "approved":       approved,
            "rejected":       n - approved,
            "approval_rate":  round(approved / n * 100, 1),
        }
