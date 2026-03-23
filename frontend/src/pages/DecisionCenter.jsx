import React, { useState, useEffect, useCallback, useRef } from 'react';

const API = 'http://localhost:8000/api';

const card = {
  background: '#fff', borderRadius: '12px', padding: '1.25rem',
  boxShadow: '0 1px 4px rgba(0,0,0,0.06)', border: '1px solid #e8e8e8',
};
const label = { fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' };

const DECISION_VARS = [
  { key: 'Granulation_Time', label: 'Granulation Time', unit: 'min', min: 9, max: 27 },
  { key: 'Binder_Amount', label: 'Binder Amount', unit: 'g', min: 5, max: 15 },
  { key: 'Drying_Temp', label: 'Drying Temp', unit: 'C', min: 40, max: 80 },
  { key: 'Drying_Time', label: 'Drying Time', unit: 'min', min: 20, max: 60 },
  { key: 'Compression_Force', label: 'Compression Force', unit: 'kN', min: 5, max: 25 },
  { key: 'Machine_Speed', label: 'Machine Speed', unit: 'RPM', min: 20, max: 80 },
  { key: 'Lubricant_Conc', label: 'Lubricant Conc', unit: '%', min: 0.3, max: 2.0, step: 0.1 },
];

export default function DecisionCenter() {
  const [state, setState] = useState(null);
  const [briefing, setBriefing] = useState(null);
  const [featureImp, setFeatureImp] = useState({});
  const [shapData, setShapData] = useState({});
  const [shapTarget, setShapTarget] = useState('');
  const [whatIfSettings, setWhatIfSettings] = useState({});
  const [whatIfResult, setWhatIfResult] = useState(null);

  // LLM Chat state
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatSending, setChatSending] = useState(false);
  const chatEndRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const [sr, br, fi, sh] = await Promise.all([
        fetch(`${API}/graph_state?batch_id=LATEST_KNOWN`),
        fetch(`${API}/briefing?batch_id=LATEST_KNOWN`),
        fetch(`${API}/feature_importance?top_n=12`),
        fetch(`${API}/shap_values`),
      ]);
      const sd = await sr.json();
      const bd = await br.json();
      const fiData = await fi.json();
      const shData = await sh.json();
      if (sd.status !== 'not_found') {
        setState(sd);
        const proposed = sd.proposed_settings || {};
        if (Object.keys(whatIfSettings).length === 0 && Object.keys(proposed).length > 0) {
          setWhatIfSettings({ ...proposed });
        }
      }
      setBriefing(bd);
      if (fiData?.features) setFeatureImp(fiData.features);
      if (shData?.targets) {
        setShapData(shData.targets);
        if (!shapTarget && Object.keys(shData.targets).length > 0) {
          setShapTarget(Object.keys(shData.targets)[0]);
        }
      }
    } catch (_) {}
  }, [whatIfSettings, shapTarget]);

  useEffect(() => {
    poll();
    const iv = setInterval(poll, 5000);
    return () => clearInterval(iv);
  }, [poll]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const runWhatIf = async () => {
    try {
      const r = await fetch(`${API}/what_if`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: whatIfSettings }),
      });
      setWhatIfResult(await r.json());
    } catch (e) { console.error(e); }
  };

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setChatSending(true);
    try {
      const r = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, batch_id: state?.batch_id }),
      });
      const data = await r.json();
      let aiText = data.response || 'No response from AI.';
      if (data.action === 'approve') aiText = `✅ ${aiText}`;
      else if (data.action === 'reject') aiText = `❌ ${aiText}`;
      if (data.execution_resumed) {
        aiText += '\n\n📋 Decision has been executed.';
        setTimeout(poll, 2000);
      }
      setChatMessages(prev => [...prev, { role: 'ai', text: aiText, action: data.action }]);
    } catch (e) {
      setChatMessages(prev => [...prev, { role: 'ai', text: '⚠️ Connection error.' }]);
    }
    setChatSending(false);
  };

  const sv = state || {};
  const proposed = sv.proposed_settings || {};
  const rawSettings = sv.raw_settings || {};
  const paused = sv.paused_for_hitl;
  const novelty = sv.novelty_warning || {};
  const pastWarnings = sv.past_decision_warnings || [];
  const confPct = ((sv.baseline_score || 0) * 100).toFixed(0);

  // Feature Importance sorted
  const fiEntries = Object.entries(featureImp).sort((a, b) => b[1] - a[1]);
  const maxFI = fiEntries.length > 0 ? Math.max(...fiEntries.map(e => e[1])) : 1;

  // SHAP data for selected target
  const selectedShap = shapData[shapTarget] || {};
  const shapFeatures = selectedShap.features || [];
  const maxAbsShap = shapFeatures.length > 0 ? Math.max(...shapFeatures.map(f => Math.abs(f.shap_value))) : 1;

  return (
    <div>
      <h1 style={{ margin: '0 0 1.5rem', fontSize: '1.5rem', fontWeight: 700 }}>Decision Center</h1>

      {/* AI Briefing */}
      <div style={{ ...card, marginBottom: '1rem', borderLeft: '4px solid #1152d4', background: '#f8faff' }}>
        <div style={{ ...label, color: '#1152d4' }}>AI Briefing (Gemini LLM)</div>
        <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: 1.6, color: '#333', whiteSpace: 'pre-wrap' }}>
          {briefing?.briefing || 'Waiting for batch data...'}
        </p>
      </div>

      {/* Proposed Parameters */}
      <div style={{ ...card, marginBottom: '1rem' }}>
        <div style={label}>Proposed Parameters</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #eee' }}>
              <th style={{ textAlign: 'left', padding: '0.5rem', color: '#888', fontWeight: 600 }}>Variable</th>
              <th style={{ textAlign: 'right', padding: '0.5rem', color: '#888', fontWeight: 600 }}>Raw NN</th>
              <th style={{ textAlign: 'right', padding: '0.5rem', color: '#888', fontWeight: 600 }}>Proposed</th>
            </tr>
          </thead>
          <tbody>
            {DECISION_VARS.map(({ key, label: l, unit }) => {
              const raw = rawSettings[key];
              const prop = proposed[key];
              const clamped = raw != null && prop != null && Math.abs(raw - prop) > 0.01;
              return (
                <tr key={key} style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <td style={{ padding: '0.4rem 0.5rem', fontWeight: 500 }}>{l}</td>
                  <td style={{ padding: '0.4rem 0.5rem', textAlign: 'right', color: '#999' }}>
                    {raw != null ? raw.toFixed(2) : '--'}
                  </td>
                  <td style={{
                    padding: '0.4rem 0.5rem', textAlign: 'right', fontWeight: 600,
                    color: clamped ? '#e74c3c' : '#27ae60'
                  }}>
                    {prop != null ? `${prop.toFixed(2)} ${unit}` : '--'}
                    {clamped && <span style={{ fontSize: '0.65rem', marginLeft: '4px' }}>(clamped)</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Feature Importance + SHAP side by side */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        {/* Feature Importance (Model-Level) */}
        <div style={{ ...card, flex: 1, minWidth: '320px' }}>
          <div style={label}>Feature Importance (Model-Level, XGBoost Gain)</div>
          <p style={{ fontSize: '0.72rem', color: '#aaa', margin: '0 0 0.75rem' }}>
            How important each feature is <strong>across all batches</strong> for the model's predictions.
          </p>
          {fiEntries.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {fiEntries.map(([name, val]) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '130px', fontSize: '0.72rem', color: '#555', textAlign: 'right', flexShrink: 0 }}>
                    {name.replace(/_/g, ' ').slice(0, 22)}
                  </div>
                  <div style={{ flex: 1, background: '#f0f0f0', borderRadius: '4px', height: '16px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${(val / maxFI) * 100}%`, height: '100%',
                      background: 'linear-gradient(90deg, #6366f1, #8b5cf6)', borderRadius: '4px',
                    }} />
                  </div>
                  <div style={{ width: '50px', fontSize: '0.68rem', color: '#888', flexShrink: 0 }}>
                    {(val * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          ) : <p style={{ color: '#999', fontSize: '0.85rem' }}>Run a batch to see feature importances.</p>}
        </div>

        {/* SHAP Waterfall (Instance-Level) */}
        <div style={{ ...card, flex: 1, minWidth: '380px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
            <div style={label}>SHAP Values (This Batch, Per-Feature Contribution)</div>
            {Object.keys(shapData).length > 1 && (
              <select
                value={shapTarget}
                onChange={e => setShapTarget(e.target.value)}
                style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', borderRadius: '6px', border: '1px solid #ddd' }}
              >
                {Object.keys(shapData).map(t => (
                  <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
                ))}
              </select>
            )}
          </div>
          <p style={{ fontSize: '0.72rem', color: '#aaa', margin: '0 0 0.5rem' }}>
            How each feature pushed <strong>this specific prediction</strong> up (🔴) or down (🔵) from the baseline.
          </p>

          {selectedShap.base_value != null && (
            <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.78rem' }}>
              <div><span style={{ color: '#888' }}>Base Value: </span><strong>{selectedShap.base_value.toFixed(2)}</strong></div>
              <div><span style={{ color: '#888' }}>Predicted: </span><strong style={{ color: '#1152d4' }}>{selectedShap.predicted.toFixed(2)}</strong></div>
              <div><span style={{ color: '#888' }}>Δ: </span>
                <strong style={{ color: selectedShap.predicted > selectedShap.base_value ? '#e74c3c' : '#2563eb' }}>
                  {(selectedShap.predicted - selectedShap.base_value) > 0 ? '+' : ''}
                  {(selectedShap.predicted - selectedShap.base_value).toFixed(3)}
                </strong>
              </div>
            </div>
          )}

          {shapFeatures.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              {shapFeatures.slice(0, 12).map(({ name, shap_value, feature_value }) => {
                const pct = (Math.abs(shap_value) / maxAbsShap) * 100;
                const isPositive = shap_value > 0;
                return (
                  <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div style={{ width: '120px', fontSize: '0.7rem', color: '#555', textAlign: 'right', flexShrink: 0 }}>
                      {name.replace(/_/g, ' ').slice(0, 20)}
                    </div>
                    {/* Diverging bar from center */}
                    <div style={{ flex: 1, position: 'relative', height: '18px' }}>
                      <div style={{
                        position: 'absolute', left: '50%', top: 0, bottom: 0,
                        width: '1px', background: '#ddd',
                      }} />
                      <div style={{
                        position: 'absolute', top: '1px', height: '16px', borderRadius: '3px',
                        ...(isPositive
                          ? { left: '50%', width: `${pct / 2}%`, background: 'linear-gradient(90deg, #ef4444, #f87171)' }
                          : { right: '50%', width: `${pct / 2}%`, background: 'linear-gradient(90deg, #3b82f6, #60a5fa)' }
                        ),
                      }} />
                    </div>
                    <div style={{ width: '55px', fontSize: '0.65rem', color: isPositive ? '#dc2626' : '#2563eb', fontWeight: 600, flexShrink: 0 }}>
                      {shap_value > 0 ? '+' : ''}{shap_value.toFixed(4)}
                    </div>
                    <div style={{ width: '40px', fontSize: '0.62rem', color: '#aaa', flexShrink: 0 }}>
                      ={feature_value?.toFixed?.(1) || '?'}
                    </div>
                  </div>
                );
              })}
              <div style={{ fontSize: '0.65rem', color: '#aaa', textAlign: 'center', marginTop: '0.25rem' }}>
                🔴 Pushes prediction UP &nbsp;&nbsp;|&nbsp;&nbsp; 🔵 Pushes prediction DOWN
              </div>
            </div>
          ) : <p style={{ color: '#999', fontSize: '0.85rem' }}>Run a batch to see SHAP values.</p>}
        </div>
      </div>

      {/* Digital Twin Sandbox */}
      <div style={{ ...card, marginBottom: '1rem' }}>
        <div style={{ ...label, color: '#8e44ad' }}>Digital Twin Sandbox (What-If)</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
          {DECISION_VARS.map(({ key, label: l, unit, min, max, step }) => (
            <div key={key}>
              <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem' }}>{l} ({unit})</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input type="range" min={min} max={max} step={step || 1}
                  value={whatIfSettings[key] || (min + max) / 2}
                  onChange={e => setWhatIfSettings(p => ({ ...p, [key]: +e.target.value }))}
                  style={{ flex: 1, accentColor: '#8e44ad' }}
                />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#8e44ad', minWidth: '40px' }}>
                  {(whatIfSettings[key] || (min + max) / 2).toFixed(step === 0.1 ? 1 : 0)}
                </span>
              </div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          <button onClick={runWhatIf} style={{
            padding: '0.5rem 1.5rem', background: '#8e44ad', color: '#fff', border: 'none',
            borderRadius: '6px', fontWeight: 600, cursor: 'pointer',
          }}>Simulate</button>
          {whatIfResult?.predictions && (
            <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem', flexWrap: 'wrap' }}>
              {Object.entries(whatIfResult.predictions).filter(([k]) => k !== 'carbon_estimate_kg').map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: '#888' }}>{k.replace(/_/g, ' ')}: </span>
                  <strong>{v.predicted?.toFixed(2)}</strong>
                </div>
              ))}
              <div>
                <span style={{ color: '#888' }}>Carbon: </span>
                <strong style={{ color: '#27ae60' }}>{whatIfResult.predictions.carbon_estimate_kg?.toFixed(4)} kgCO2</strong>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Anomaly Alerts */}
      {(confPct < 85 || novelty?.is_novel || pastWarnings.length > 0) && (
        <div style={{ ...card, marginBottom: '1rem', borderLeft: '4px solid #e67e22', background: '#fff8e1' }}>
          <div style={{ ...label, color: '#e67e22' }}>Anomaly Alerts</div>
          {confPct < 85 && (
            <p style={{ margin: '0.25rem 0', fontSize: '0.85rem', color: '#d35400' }}>
              Low model confidence ({confPct}%). Outside known training distribution.
            </p>
          )}
          {novelty?.is_novel && (
            <p style={{ margin: '0.25rem 0', fontSize: '0.85rem', color: '#c0392b' }}>
              Novelty detected: match score {(novelty.score * 100).toFixed(1)}%
            </p>
          )}
          {pastWarnings.map((w, i) => (
            <p key={i} style={{ margin: '0.25rem 0', fontSize: '0.85rem', color: '#e67e22' }}>
              Past rejection: "{w.feedback}" (Batch {w.batch_id})
            </p>
          ))}
        </div>
      )}

      {/* LLM Chat Interface */}
      <div style={{ ...card, borderLeft: '4px solid #27ae60', background: '#f0fff4' }}>
        <div style={{ ...label, color: '#27ae60' }}>
          Operator ↔ AI Chat {paused ? '(Awaiting Your Decision)' : ''}
        </div>
        <div style={{
          maxHeight: '300px', overflowY: 'auto', marginBottom: '0.75rem',
          padding: '0.75rem', background: '#fff', borderRadius: '8px', border: '1px solid #e0e0e0',
        }}>
          {chatMessages.length === 0 && (
            <p style={{ color: '#aaa', fontSize: '0.85rem', textAlign: 'center' }}>
              {paused
                ? 'Type a message to communicate with the AI agent. Say "approve", "reject", or ask a question.'
                : 'Chat with the AI agent about the current batch. Run a batch first to start.'}
            </p>
          )}
          {chatMessages.map((msg, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '0.5rem',
            }}>
              <div style={{
                maxWidth: '75%', padding: '0.6rem 0.9rem', borderRadius: '12px',
                fontSize: '0.9rem', lineHeight: 1.5, whiteSpace: 'pre-wrap',
                background: msg.role === 'user' ? '#1152d4' : '#f0f0f0',
                color: msg.role === 'user' ? '#fff' : '#333',
              }}>
                {msg.role === 'ai' && <strong style={{ fontSize: '0.7rem', color: '#27ae60' }}>AI Agent: </strong>}
                {msg.text}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <input
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendChat()}
            placeholder={paused ? 'e.g. "Looks good, approve it" or "What if we lower speed?"' : 'Ask the AI about the batch...'}
            style={{ flex: 1, padding: '0.65rem 1rem', border: '1px solid #ddd', borderRadius: '8px', fontSize: '0.9rem', fontFamily: 'inherit' }}
            disabled={chatSending}
          />
          <button onClick={sendChat} disabled={chatSending || !chatInput.trim()} style={{
            padding: '0.65rem 1.5rem', background: chatSending ? '#95a5a6' : '#27ae60', color: '#fff',
            border: 'none', borderRadius: '8px', fontWeight: 700, cursor: chatSending ? 'wait' : 'pointer', fontSize: '0.9rem',
          }}>{chatSending ? 'Sending...' : 'Send'}</button>
        </div>
        {paused && (
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            {['Approve this batch', 'Reject - needs review', 'Explain the recommendation', 'What are the risks?'].map(q => (
              <button key={q} onClick={() => setChatInput(q)} style={{
                padding: '0.35rem 0.75rem', background: '#e8f5e9', color: '#27ae60', border: '1px solid #c8e6c9',
                borderRadius: '16px', fontSize: '0.75rem', cursor: 'pointer', fontWeight: 500,
              }}>{q}</button>
            ))}
          </div>
        )}
      </div>

      {/* Success Banner */}
      {!paused && sv.execution_status === 'executed' && (
        <div style={{ ...card, marginTop: '1rem', borderLeft: '4px solid #27ae60', background: '#f0fff4', textAlign: 'center' }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#27ae60' }}>Batch Executed Successfully</div>
          <div style={{ fontSize: '0.85rem', color: '#888', marginTop: '0.25rem' }}>
            Quality Delta: {sv.quality_delta?.toFixed(4)} | Qdrant Updated: {sv.qdrant_updated ? 'Yes' : 'No'}
          </div>
        </div>
      )}
    </div>
  );
}
