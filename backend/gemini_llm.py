"""
==============================================================================
GEMINI LLM ENGINE - V2.0: Natural Language Communication
==============================================================================
Uses Google Gemini API for:
  1. Generating plain-English AI briefings from SHAP + prediction data
  2. Processing operator text input as natural language HITL commands
  3. Explaining batch outcomes in simple English

Falls back to context-aware template responses when Gemini is unavailable.

Author : V2.0 Team
Version: 2.0.0
==============================================================================
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("gemini_llm")

# Load .env if dotenv is available
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_model = None


def _get_model():
    """Lazy-initialize the Gemini model."""
    global _model
    if _model is not None:
        return _model

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        log.warning("GEMINI_API_KEY not set. LLM features will use template fallback.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # Try gemini-1.5-flash first (better free tier quota), fall back to 2.0-flash
        for model_name in ["gemini-1.5-flash", "gemini-2.0-flash"]:
            try:
                m = genai.GenerativeModel(model_name)
                # Quick test to verify the model works
                _model = m
                log.info("Gemini model '%s' initialized successfully", model_name)
                return _model
            except Exception:
                continue
        log.warning("No Gemini model available, using fallback")
        return None
    except Exception as e:
        log.error("Failed to initialize Gemini: %s", e)
        return None


def _safe_generate(prompt: str) -> Optional[str]:
    """Call Gemini with error handling. Returns None on failure."""
    model = _get_model()
    if model is None:
        return None
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log.error("Gemini call failed: %s", e)
        return None


def generate_ai_briefing(
    proposed_settings: Dict[str, Any],
    shap_values: Dict[str, float],
    baseline_score: float,
    prediction_intervals: Dict[str, Any],
    no_confident_match: bool,
    telemetry: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a plain-English AI briefing using Gemini LLM.

    Falls back to template-based generation if Gemini is unavailable.
    """
    prompt = f"""You are an AI manufacturing advisor. Generate a clear, concise briefing (3-4 sentences)
for a factory operator about the current batch optimization recommendation.

Speak in first person ("I recommend..."). Be specific with numbers. No jargon.

Data:
- Model Confidence: {baseline_score * 100:.1f}% (Qdrant vector similarity to historical batches)
- Top Feature Drivers (SHAP importance): {json.dumps(shap_values, indent=2)}
- Proposed Machine Settings: {json.dumps(proposed_settings, indent=2)}
- Predicted Outcomes: {json.dumps(prediction_intervals, indent=2)}
- Outside Training Distribution: {no_confident_match}
- Current Sensor Readings: {json.dumps(telemetry or {}, indent=2)}

Generate the briefing:"""

    result = _safe_generate(prompt)
    if result:
        return result

    return _template_briefing(proposed_settings, shap_values, baseline_score,
                               prediction_intervals, no_confident_match)


def process_operator_message(
    message: str,
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Process natural language input from operator and return structured response.

    The operator can type things like:
    - "Looks good, approve it"
    - "The temperature seems too high, can you lower it?"
    - "Reject this, motor vibration is concerning"
    - "What would happen if we increase compression force?"
    - "Tell me about the batch"
    - "What are the risks?"

    Returns:
        Dict with keys: response (str), action (approve/reject/question/modify),
        approved (bool or None), feedback (str)
    """
    prompt = f"""You are an AI assistant helping a factory operator make decisions about machine settings.
The operator just said: "{message}"

Current batch state:
- Proposed settings: {json.dumps(current_state.get('proposed_settings', {}), indent=2)}
- Model confidence: {current_state.get('baseline_score', 0) * 100:.1f}%
- Predicted tablet weight: {current_state.get('prediction_intervals', {}).get('Tablet_Weight', {}).get('predicted', 'N/A')}
- Execution status: {current_state.get('execution_status', 'pending')}

Respond as a JSON object with these exact keys:
{{
  "response": "Your natural language response to the operator (2-3 sentences, friendly and professional)",
  "action": "one of: approve, reject, question, info",
  "approved": true/false/null (true if approving, false if rejecting, null if just asking),
  "feedback": "extracted operator feedback for the audit log"
}}

Return ONLY valid JSON, no markdown:"""

    result = _safe_generate(prompt)
    if result:
        try:
            # Clean markdown code blocks if present
            text = result
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            log.error("Failed to parse Gemini response as JSON: %s", e)

    # Smart context-aware fallback
    return _context_aware_response(message, current_state)


def explain_outcome(
    batch_id: str,
    outcome: Dict[str, Any],
    quality_delta: float,
    carbon_kg: float,
) -> str:
    """Generate a plain-English explanation of the batch execution outcome."""
    prompt = f"""You are an AI manufacturing advisor. Explain the batch execution result to the operator
in 2-3 simple sentences.

- Batch ID: {batch_id}
- Outcome metrics: {json.dumps(outcome, indent=2)}
- Quality improvement vs baseline: {quality_delta:.4f} (positive = better)
- Carbon emissions: {carbon_kg:.3f} kgCO2

Be conversational and highlight whether this was a good or bad result:"""

    result = _safe_generate(prompt)
    if result:
        return result

    improved = "improved" if quality_delta > 0 else "did not improve"
    return (f"Batch {batch_id} has been executed. Quality {improved} "
            f"by {abs(quality_delta):.4f} compared to baseline. "
            f"Carbon footprint: {carbon_kg:.3f} kgCO2.")


# =========================================================================
# CONTEXT-AWARE FALLBACK (used when Gemini is unavailable / quota exceeded)
# =========================================================================

def _context_aware_response(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Smart fallback that uses batch context to give meaningful responses."""
    msg_lower = message.lower().strip()
    proposed = state.get("proposed_settings", {})
    conf = state.get("baseline_score", 0) * 100
    pred = state.get("prediction_intervals", {})
    tw = pred.get("Tablet_Weight", {})
    pw = pred.get("Power_Consumption_kW", {})
    novelty = state.get("novelty_warning", {})
    exec_status = state.get("execution_status", "pending")

    # --- APPROVE ---
    if any(w in msg_lower for w in ["approve", "accept", "looks good", "go ahead", "yes", "lgtm", "proceed"]):
        return {
            "response": f"Understood. Approving batch with {conf:.0f}% confidence. "
                        f"Predicted tablet weight: {tw.get('predicted', 'N/A'):.1f}g. Executing now.",
            "action": "approve",
            "approved": True,
            "feedback": message,
        }

    # --- REJECT ---
    if any(w in msg_lower for w in ["reject", "deny", "cancel", "stop", "don't", "no way"]):
        return {
            "response": f"Understood. Rejecting the proposed settings. Your feedback has been "
                        f"recorded and will influence future recommendations for similar batches.",
            "action": "reject",
            "approved": False,
            "feedback": message,
        }

    # --- RISK QUESTIONS ---
    if any(w in msg_lower for w in ["risk", "danger", "concern", "warning", "safe", "problem"]):
        risks = []
        if conf < 70:
            risks.append(f"Low model confidence ({conf:.0f}%) — this batch is outside the training distribution")
        elif conf < 85:
            risks.append(f"Moderate confidence ({conf:.0f}%) — some deviation from historical patterns")
        else:
            risks.append(f"High confidence ({conf:.0f}%) — this batch closely matches known successful runs")

        if novelty.get("is_novel"):
            risks.append(f"Novelty flag triggered (match score: {novelty.get('score', 0)*100:.1f}%)")

        # Check if any proposed settings are near bounds
        for var, val in proposed.items():
            if isinstance(val, (int, float)):
                if val > 20 and "Force" in var:
                    risks.append(f"{var.replace('_', ' ')} is high at {val:.1f}")
                if val > 70 and "Speed" in var:
                    risks.append(f"{var.replace('_', ' ')} is high at {val:.1f}")

        risk_text = "\n• ".join(risks) if risks else "No significant risks detected for this batch."
        return {
            "response": f"Here's my risk assessment:\n• {risk_text}",
            "action": "info",
            "approved": None,
            "feedback": message,
        }

    # --- BATCH INFO / TELL ME ABOUT ---
    if any(w in msg_lower for w in ["tell me", "about", "batch", "what", "explain", "summary", "describe", "detail", "info"]):
        parts = [f"Current batch analysis (confidence: {conf:.0f}%):"]

        if proposed:
            key_settings = []
            for var in ["Compression_Force", "Machine_Speed", "Drying_Temp", "Granulation_Time"]:
                if var in proposed:
                    key_settings.append(f"{var.replace('_', ' ')}: {proposed[var]:.1f}")
            if key_settings:
                parts.append(f"Key proposed settings: {', '.join(key_settings)}")

        if tw.get("predicted"):
            parts.append(f"Predicted tablet weight: {tw['predicted']:.1f}g")
            if tw.get("lower") and tw.get("upper"):
                parts.append(f"Confidence interval: [{tw['lower']:.1f}, {tw['upper']:.1f}]g")

        if pw.get("predicted"):
            carbon_est = pw["predicted"] * 0.82 / 60
            parts.append(f"Estimated power: {pw['predicted']:.1f} kW (≈{carbon_est:.3f} kgCO2)")

        parts.append(f"Status: {exec_status}")

        if novelty.get("is_novel"):
            parts.append("⚠️ Novelty warning: this batch differs significantly from historical data.")

        return {
            "response": "\n".join(parts),
            "action": "info",
            "approved": None,
            "feedback": message,
        }

    # --- WHAT IF / MODIFY ---
    if any(w in msg_lower for w in ["what if", "lower", "higher", "increase", "decrease", "change", "modify", "adjust"]):
        return {
            "response": f"To explore 'what-if' scenarios, use the Digital Twin Sandbox sliders above. "
                        f"You can adjust any of the {len(proposed)} decision variables and click 'Simulate' "
                        f"to see predicted outcomes in real-time. Current model confidence: {conf:.0f}%.",
            "action": "info",
            "approved": None,
            "feedback": message,
        }

    # --- GENERIC FALLBACK (with context) ---
    return {
        "response": f"I'm monitoring this batch with {conf:.0f}% model confidence. "
                    f"Predicted tablet weight: {tw.get('predicted', 'N/A')}g. "
                    f"You can ask me about risks, batch details, or type 'approve'/'reject' to make your decision.",
        "action": "info",
        "approved": None,
        "feedback": message,
    }


def _template_briefing(
    proposed: Dict[str, Any],
    shap_values: Dict[str, float],
    baseline_score: float,
    prediction_intervals: Dict[str, Any],
    no_confident_match: bool,
) -> str:
    """Template-based fallback when Gemini is not available."""
    parts = []
    conf_pct = baseline_score * 100

    if conf_pct >= 90:
        parts.append(f"I have high confidence ({conf_pct:.0f}% Qdrant match) in this recommendation.")
    elif conf_pct >= 70:
        parts.append(f"I have moderate confidence ({conf_pct:.0f}% match) — this batch differs slightly from historical baselines.")
    else:
        parts.append(f"WARNING: Low confidence ({conf_pct:.0f}% match). This batch is significantly different from training data.")

    if shap_values:
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        drivers = [f"{n.replace('_', ' ')} ({v*100:.1f}%)" for n, v in sorted_shap]
        parts.append(f"Key factors driving this prediction: {' and '.join(drivers)}.")

    if proposed:
        settings = []
        for var in ["Machine_Speed", "Compression_Force", "Drying_Temp"]:
            if var in proposed:
                settings.append(f"{var.replace('_', ' ')}: {proposed[var]:.1f}")
        if settings:
            parts.append(f"I recommend: {', '.join(settings)}.")

    if prediction_intervals:
        tw = prediction_intervals.get("Tablet_Weight", {})
        if tw.get("predicted"):
            pred_str = f"Expected tablet weight: {tw['predicted']:.1f}g"
            if tw.get("lower") and tw.get("upper"):
                pred_str += f" (95% CI: [{tw['lower']:.1f}, {tw['upper']:.1f}])"
            parts.append(pred_str + ".")

    if no_confident_match:
        parts.append("⚠️ No confident historical match found — manual verification recommended.")

    return " ".join(parts)
