"""
==============================================================================
AUDIT LEDGER — V2.0: Hash-Chained Immutable Audit Trail
==============================================================================
Provides a SHA-256 hash-chained immutable log for recording every AI
suggestion, human decision, and carbon outcome. Each record contains a hash
of the previous record, making the chain tamper-evident.

Includes ISO 14064 scope classification for carbon emissions and
auto-generated PDF (or text) compliance reports.

Author : V2.0 Team
Version: 2.0.0
==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("audit_ledger")


class AuditRecord:
    """A single immutable audit record in the hash chain."""

    def __init__(
        self,
        index: int,
        batch_id: str,
        ai_suggestion: Dict[str, Any],
        human_decision: str,
        human_feedback: str,
        carbon_kg: float,
        power_kw: float,
        iso_scope: str = "Scope 2",
        iso_label: str = "Purchased Electricity",
        previous_hash: str = "GENESIS",
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.batch_id = batch_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.ai_suggestion = ai_suggestion
        self.human_decision = human_decision
        self.human_feedback = human_feedback
        self.carbon_kg = carbon_kg
        self.power_kw = power_kw
        self.iso_scope = iso_scope
        self.iso_label = iso_label
        self.previous_hash = previous_hash
        self.extra = extra or {}
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the record contents + previous hash."""
        payload = json.dumps(
            {
                "index": self.index,
                "batch_id": self.batch_id,
                "timestamp": self.timestamp,
                "ai_suggestion": self.ai_suggestion,
                "human_decision": self.human_decision,
                "carbon_kg": self.carbon_kg,
                "power_kw": self.power_kw,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "ai_suggestion": self.ai_suggestion,
            "human_decision": self.human_decision,
            "human_feedback": self.human_feedback,
            "carbon_kg": self.carbon_kg,
            "power_kw": self.power_kw,
            "iso_scope": self.iso_scope,
            "iso_label": self.iso_label,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "extra": self.extra,
        }


class AuditLedger:
    """Hash-chained immutable audit log with ISO 14064 scope support."""

    def __init__(self):
        self._chain: List[AuditRecord] = []

    def append(
        self,
        batch_id: str,
        ai_suggestion: Dict[str, Any],
        human_decision: str,
        human_feedback: str = "",
        carbon_kg: float = 0.0,
        power_kw: float = 0.0,
        iso_scope: str = "Scope 2",
        iso_label: str = "Purchased Electricity",
        extra: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Append a new record to the hash chain."""
        prev_hash = self._chain[-1].hash if self._chain else "GENESIS"
        record = AuditRecord(
            index=len(self._chain),
            batch_id=batch_id,
            ai_suggestion=ai_suggestion,
            human_decision=human_decision,
            human_feedback=human_feedback,
            carbon_kg=carbon_kg,
            power_kw=power_kw,
            iso_scope=iso_scope,
            iso_label=iso_label,
            previous_hash=prev_hash,
            extra=extra,
        )
        self._chain.append(record)
        log.info(
            "Audit record #%d appended (batch=%s, decision=%s, hash=%s...)",
            record.index, batch_id, human_decision, record.hash[:12],
        )
        return record

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the entire hash chain."""
        if not self._chain:
            return {"valid": True, "length": 0}

        for i, record in enumerate(self._chain):
            expected_prev = self._chain[i - 1].hash if i > 0 else "GENESIS"
            if record.previous_hash != expected_prev:
                return {
                    "valid": False,
                    "broken_at": i,
                    "length": len(self._chain),
                    "error": f"Hash chain broken at index {i}",
                }
            recomputed = record._compute_hash()
            if recomputed != record.hash:
                return {
                    "valid": False,
                    "broken_at": i,
                    "length": len(self._chain),
                    "error": f"Record {i} hash mismatch (tampered?)",
                }

        return {"valid": True, "length": len(self._chain)}

    def get_latest(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get the latest N records as dicts."""
        return [r.to_dict() for r in self._chain[-n:]]

    def get_iso_summary(self) -> Dict[str, float]:
        """Get ISO 14064 carbon summary by scope."""
        scope_totals = {"scope_1_kg": 0.0, "scope_2_kg": 0.0, "scope_3_kg": 0.0}
        for record in self._chain:
            if record.iso_scope == "Scope 1":
                scope_totals["scope_1_kg"] += record.carbon_kg
            elif record.iso_scope == "Scope 2":
                scope_totals["scope_2_kg"] += record.carbon_kg
            else:
                scope_totals["scope_3_kg"] += record.carbon_kg
        scope_totals["total_kg"] = sum(scope_totals.values())
        return scope_totals

    def export_audit_pdf(self, output_path: Optional[str] = None) -> str:
        """Export the audit trail as a PDF or text report.

        Uses fpdf2 if available, otherwise falls back to plain text.
        """
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "audit_report.pdf",
            )

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Intelli-Credit AI - Audit Compliance Report", ln=True, align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
            pdf.ln(5)

            # ISO Summary
            iso = self.get_iso_summary()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "ISO 14064 Carbon Summary", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for label, val in iso.items():
                pdf.cell(0, 6, f"  {label}: {val:.4f} kgCO2", ln=True)
            pdf.ln(5)

            # Chain integrity
            integrity = self.verify_chain()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Hash Chain Integrity", ln=True)
            pdf.set_font("Helvetica", "", 10)
            status = "VERIFIED" if integrity["valid"] else f"BROKEN at record {integrity.get('broken_at')}"
            pdf.cell(0, 6, f"  Status: {status} ({integrity['length']} records)", ln=True)
            pdf.ln(5)

            # Records table
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Audit Records", ln=True)
            pdf.set_font("Helvetica", "", 8)

            for record in self._chain:
                d = record.to_dict()
                pdf.cell(0, 5, f"#{d['index']} | Batch: {d['batch_id']} | {d['timestamp'][:19]}", ln=True)
                pdf.cell(0, 5, f"   Decision: {d['human_decision']} | Carbon: {d['carbon_kg']:.3f} kg | {d['iso_scope']}", ln=True)
                pdf.cell(0, 5, f"   Hash: {d['hash'][:32]}...", ln=True)
                if d["human_feedback"]:
                    pdf.cell(0, 5, f"   Feedback: {d['human_feedback'][:80]}", ln=True)
                pdf.ln(2)

            pdf.output(output_path)
            log.info("Audit PDF exported to: %s", output_path)
            return output_path

        except ImportError:
            # Fallback to text report
            txt_path = output_path.replace(".pdf", ".txt")
            lines = [
                "=" * 60,
                "INTELLI-CREDIT AI - AUDIT COMPLIANCE REPORT",
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                "=" * 60,
                "",
                "ISO 14064 Carbon Summary:",
            ]
            iso = self.get_iso_summary()
            for label, val in iso.items():
                lines.append(f"  {label}: {val:.4f} kgCO2")

            lines.append("")
            integrity = self.verify_chain()
            status = "VERIFIED" if integrity["valid"] else f"BROKEN at record {integrity.get('broken_at')}"
            lines.append(f"Hash Chain: {status} ({integrity['length']} records)")
            lines.append("")
            lines.append("-" * 60)

            for record in self._chain:
                d = record.to_dict()
                lines.append(f"#{d['index']} | {d['batch_id']} | {d['timestamp'][:19]}")
                lines.append(f"  Decision: {d['human_decision']} | Carbon: {d['carbon_kg']:.3f} kg | {d['iso_scope']}")
                lines.append(f"  Hash: {d['hash'][:32]}...")
                if d["human_feedback"]:
                    lines.append(f"  Feedback: {d['human_feedback'][:80]}")
                lines.append("")

            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            log.info("Audit text report exported to: %s", txt_path)
            return txt_path
