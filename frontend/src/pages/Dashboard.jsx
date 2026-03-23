import React, { useState, useEffect, useCallback } from 'react';

const API = 'http://localhost:8000/api';

const card = {
  background: '#fff', borderRadius: '12px', padding: '1.25rem',
  boxShadow: '0 1px 4px rgba(0,0,0,0.06)', border: '1px solid #e8e8e8',
};
const kpiCard = {
  ...card, textAlign: 'center', flex: 1, minWidth: '140px',
};
const label = { fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.25rem' };
const bigNum = { fontSize: '1.6rem', fontWeight: 700, color: '#1a1a1a' };

const MODE_LABELS = {
  yield_vs_energy: { label: 'Yield vs Energy', left: 'Yield', right: 'Energy', leftColor: '#1152d4', rightColor: '#e67e22' },
  quality_vs_speed: { label: 'Quality vs Speed', left: 'Quality', right: 'Speed', leftColor: '#27ae60', rightColor: '#8e44ad' },
  carbon_min: { label: 'Carbon Minimize', left: 'Production', right: 'Low Carbon', leftColor: '#e74c3c', rightColor: '#27ae60' },
  balanced: { label: 'Balanced', left: 'Quality', right: 'Efficiency', leftColor: '#2980b9', rightColor: '#16a085' },
};

export default function Dashboard() {
  const [batchId, setBatchId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [state, setState] = useState(null);
  const [carbon, setCarbon] = useState(null);
  const [priority, setPriority] = useState(50);
  const [priorityMode, setPriorityMode] = useState('yield_vs_energy');
  const [priorityMsg, setPriorityMsg] = useState('');

  const pollState = useCallback(async (bid) => {
    try {
      const r = await fetch(`${API}/graph_state?batch_id=${bid || 'LATEST_KNOWN'}`);
      const d = await r.json();
      if (d.status !== 'not_found') setState(d);
    } catch (_) {}
    try {
      const r2 = await fetch(`${API}/carbon_metrics`);
      setCarbon(await r2.json());
    } catch (_) {}
  }, []);

  useEffect(() => {
    pollState(batchId);
    const iv = setInterval(() => pollState(batchId), 2000);
    return () => clearInterval(iv);
  }, [batchId, pollState]);

  const runBatch = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/new_batch`, { method: 'POST' });
      const d = await r.json();
      setBatchId(d.batch_id);
      setTimeout(() => pollState(d.batch_id), 1500);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const applyPriority = async () => {
    try {
      const r = await fetch(`${API}/update_priorities`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority_value: priority, priority_type: priorityMode }),
      });
      const d = await r.json();
      if (d.status === 'success') {
        const modeLabel = MODE_LABELS[priorityMode]?.label || priorityMode;
        setPriorityMsg(`✅ ${modeLabel} strategy applied (${priority}%)`);
        setTimeout(() => setPriorityMsg(''), 3000);
      }
    } catch (_) {
      setPriorityMsg('❌ Failed to update priority');
      setTimeout(() => setPriorityMsg(''), 3000);
    }
  };

  const sv = state || {};
  const telem = sv.current_telemetry || {};
  const execStatus = sv.execution_status || 'idle';
  const baselineScore = sv.baseline_score || 0;
  const confPct = (baselineScore * 100).toFixed(0);
  const healthScore = sv.asset_health_score || 100;
  const predIntervals = sv.prediction_intervals || {};
  const twPred = predIntervals?.Tablet_Weight?.predicted;
  const pwPred = predIntervals?.Power_Consumption_kW?.predicted;
  const modeConfig = MODE_LABELS[priorityMode] || MODE_LABELS.yield_vs_energy;

  // Agent notifications from state
  const notifications = sv.agent_notifications || [];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Dashboard</h1>
          {batchId && <span style={{ fontSize: '0.8rem', color: '#888' }}>Active: {batchId}</span>}
        </div>
        <button onClick={runBatch} disabled={loading} style={{
          padding: '0.65rem 1.5rem', background: '#1152d4', color: '#fff', border: 'none',
          borderRadius: '8px', fontWeight: 600, cursor: loading ? 'wait' : 'pointer', fontSize: '0.9rem',
        }}>
          {loading ? 'Starting...' : 'Run New Batch'}
        </button>
      </div>

      {/* Priority Balancing — 4 Modes */}
      <div style={{ ...card, marginBottom: '1rem' }}>
        <div style={label}>Priority Balancing</div>

        {/* Mode Selector Pills */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
          {Object.entries(MODE_LABELS).map(([key, { label: l }]) => (
            <button key={key} onClick={() => setPriorityMode(key)} style={{
              padding: '0.4rem 1rem', borderRadius: '20px', fontSize: '0.78rem', fontWeight: 600,
              cursor: 'pointer', transition: 'all 0.2s',
              background: priorityMode === key ? '#1152d4' : '#f0f4ff',
              color: priorityMode === key ? '#fff' : '#1152d4',
              border: priorityMode === key ? '1px solid #1152d4' : '1px solid #c5d5f0',
            }}>{l}</button>
          ))}
        </div>

        {/* Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.8rem', color: modeConfig.leftColor, fontWeight: 600, minWidth: '60px' }}>{modeConfig.left}</span>
              <input type="range" min="0" max="100" value={priority} onChange={e => setPriority(+e.target.value)}
                style={{ flex: 1, accentColor: modeConfig.leftColor }} />
              <span style={{ fontSize: '0.8rem', color: modeConfig.rightColor, fontWeight: 600, minWidth: '60px', textAlign: 'right' }}>{modeConfig.right}</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#aaa', marginTop: '0.2rem', textAlign: 'center' }}>
              {priority < 30 ? `Max ${modeConfig.left}` : priority > 70 ? `Max ${modeConfig.right}` : 'Balanced'} ({priority}%)
            </div>
          </div>
          <button onClick={applyPriority} style={{
            padding: '0.5rem 1.25rem', background: '#f0f4ff', color: '#1152d4', border: '1px solid #1152d4',
            borderRadius: '6px', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem',
          }}>Apply</button>
          {priorityMsg && (
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#27ae60' }}>{priorityMsg}</span>
          )}
        </div>
      </div>

      {/* KPI Row */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <div style={kpiCard}>
          <div style={label}>Asset Health</div>
          <div style={{ ...bigNum, color: healthScore >= 80 ? '#27ae60' : '#e74c3c' }}>{healthScore.toFixed(0)}%</div>
        </div>
        <div style={kpiCard}>
          <div style={label}>Model Confidence</div>
          <div style={{ ...bigNum, color: confPct >= 85 ? '#27ae60' : confPct >= 60 ? '#e67e22' : '#e74c3c' }}>{confPct}%</div>
        </div>
        <div style={kpiCard}>
          <div style={label}>Execution Status</div>
          <div style={{ ...bigNum, fontSize: '1.1rem', color: execStatus === 'executed' ? '#27ae60' : '#1152d4' }}>
            {execStatus.toUpperCase()}
          </div>
        </div>
        <div style={kpiCard}>
          <div style={label}>Predicted Weight</div>
          <div style={bigNum}>{twPred ? `${twPred.toFixed(1)}g` : '--'}</div>
        </div>
      </div>

      {/* Agent Notifications (if any) */}
      {notifications.length > 0 && (
        <div style={{ ...card, marginBottom: '1rem', padding: '0.75rem 1.25rem' }}>
          <div style={{ ...label, marginBottom: '0.5rem' }}>Agent Notifications ({notifications.length})</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {notifications.map((n, i) => {
              const severityColors = {
                critical: { bg: '#fff5f5', border: '#e74c3c', text: '#c0392b' },
                warning: { bg: '#fffbeb', border: '#e67e22', text: '#d35400' },
                proposal: { bg: '#f0fff4', border: '#27ae60', text: '#1e8449' },
                info: { bg: '#f8faff', border: '#1152d4', text: '#333' },
              };
              const sc = severityColors[n.severity] || severityColors.info;
              return (
                <div key={i} style={{
                  padding: '0.5rem 0.75rem', borderRadius: '8px',
                  background: sc.bg, borderLeft: `3px solid ${sc.border}`,
                  fontSize: '0.82rem', color: sc.text, lineHeight: 1.5,
                }}>
                  <strong>{n.icon} {n.agent}:</strong> {n.message}
                  {n.action_required && <span style={{ fontSize: '0.7rem', fontWeight: 700, color: sc.border, marginLeft: '0.5rem' }}>ACTION NEEDED</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Telemetry + Carbon Row */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        {/* Telemetry Signature */}
        <div style={{ ...card, flex: 2, minWidth: '400px' }}>
          <div style={{ ...label, marginBottom: '0.75rem' }}>Live Telemetry Signature</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
            {[
              { k: 'Temperature_C', l: 'Temperature', u: 'C', c: '#e74c3c' },
              { k: 'Vibration_mm_s', l: 'Vibration', u: 'mm/s', c: '#e67e22' },
              { k: 'Motor_Speed_RPM', l: 'Motor Speed', u: 'RPM', c: '#1152d4' },
              { k: 'Power_Consumption_kW', l: 'Power', u: 'kW', c: '#8e44ad' },
              { k: 'Flow_Rate_LPM', l: 'Flow Rate', u: 'LPM', c: '#27ae60' },
              { k: 'Pressure_Bar', l: 'Pressure', u: 'Bar', c: '#2980b9' },
              { k: 'Humidity_Percent', l: 'Humidity', u: '%', c: '#16a085' },
              { k: 'Compression_Force_kN', l: 'Compression', u: 'kN', c: '#d35400' },
            ].map(({ k, l, u, c }) => (
              <div key={k} style={{ padding: '0.5rem', borderRadius: '8px', background: `${c}10`, borderLeft: `3px solid ${c}` }}>
                <div style={{ fontSize: '0.7rem', color: '#888' }}>{l}</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: c }}>
                  {telem[k] != null ? Number(telem[k]).toFixed(1) : '--'} <span style={{ fontSize: '0.7rem', fontWeight: 400 }}>{u}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Carbon Summary */}
        <div style={{ ...card, flex: 1, minWidth: '220px' }}>
          <div style={{ ...label, marginBottom: '0.75rem' }}>Carbon Summary</div>
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#888' }}>Est. Carbon / Batch</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#27ae60' }}>
              {pwPred ? (pwPred * 0.82 / 60).toFixed(3) : carbon?.last_batch_carbon_kg?.toFixed(3) || '0.000'} kgCO2
            </div>
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.75rem', color: '#888' }}>Cumulative Carbon</div>
            <div style={{ fontSize: '1.2rem', fontWeight: 600, color: '#2c3e50' }}>
              {carbon?.cumulative_carbon_kg?.toFixed(3) || '0.000'} kgCO2
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: '#888' }}>Emission Factor</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#7f8c8d' }}>
              {carbon?.emission_factor || 0.82} kgCO2/kWh
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
