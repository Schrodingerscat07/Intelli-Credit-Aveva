import React, { useState, useEffect, useCallback } from 'react';

const API = 'http://localhost:8000/api';

const card = {
  background: '#fff', borderRadius: '12px', padding: '1.25rem',
  boxShadow: '0 1px 4px rgba(0,0,0,0.06)', border: '1px solid #e8e8e8',
};
const label = { fontSize: '0.75rem', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' };

export default function ComplianceAudit() {
  const [audit, setAudit] = useState(null);
  const [batches, setBatches] = useState(null);
  const [regulatory, setRegulatory] = useState(null);
  const [regForm, setRegForm] = useState({});

  const poll = useCallback(async () => {
    try {
      const [ar, br, rr] = await Promise.all([
        fetch(`${API}/audit_trail`),
        fetch(`${API}/batch_history`),
        fetch(`${API}/regulatory_targets`),
      ]);
      setAudit(await ar.json());
      setBatches(await br.json());
      const rd = await rr.json();
      setRegulatory(rd);
      if (Object.keys(regForm).length === 0) setRegForm(rd);
    } catch (_) {}
  }, [regForm]);

  useEffect(() => {
    poll();
    const iv = setInterval(poll, 5000);
    return () => clearInterval(iv);
  }, [poll]);

  const downloadPdf = async () => {
    try {
      const r = await fetch(`${API}/audit_pdf`);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = r.headers.get('content-type')?.includes('pdf') ? 'audit_report.pdf' : 'audit_report.txt';
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) { console.error(e); }
  };

  const saveRegulatory = async () => {
    try {
      await fetch(`${API}/regulatory_targets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(regForm),
      });
      poll();
    } catch (e) { console.error(e); }
  };

  const records = audit?.records || [];
  const integrity = audit?.integrity || {};
  const iso = audit?.iso_summary || {};
  const batchRecords = batches?.records || [];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Compliance & Audit</h1>
        <button onClick={downloadPdf} style={{
          padding: '0.6rem 1.5rem', background: '#1152d4', color: '#fff', border: 'none',
          borderRadius: '8px', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem',
        }}>Export PDF Report</button>
      </div>

      {/* Chain Integrity + ISO Summary */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <div style={{ ...card, flex: 1, minWidth: '250px' }}>
          <div style={label}>Hash Chain Integrity</div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            fontSize: '1.3rem', fontWeight: 700,
            color: integrity.valid ? '#27ae60' : '#e74c3c',
          }}>
            <span style={{ fontSize: '1.5rem' }}>{integrity.valid ? '✓' : '✗'}</span>
            {integrity.valid ? 'VERIFIED' : `BROKEN at #${integrity.broken_at}`}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#888', marginTop: '0.25rem' }}>
            {integrity.length || 0} records in chain
          </div>
        </div>

        <div style={{ ...card, flex: 2, minWidth: '350px' }}>
          <div style={label}>ISO 14064 Carbon Summary</div>
          <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#888' }}>Scope 1 (Direct)</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700 }}>{iso.scope_1_kg?.toFixed(3) || '0.000'} <small>kgCO2</small></div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#888' }}>Scope 2 (Electricity)</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#e67e22' }}>{iso.scope_2_kg?.toFixed(3) || '0.000'} <small>kgCO2</small></div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#888' }}>Scope 3 (Upstream)</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700 }}>{iso.scope_3_kg?.toFixed(3) || '0.000'} <small>kgCO2</small></div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#888' }}>TOTAL</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#27ae60' }}>{iso.total_kg?.toFixed(3) || '0.000'} <small>kgCO2</small></div>
            </div>
          </div>
        </div>
      </div>

      {/* Immutable Audit Ledger Table */}
      <div style={{ ...card, marginBottom: '1rem', overflowX: 'auto' }}>
        <div style={label}>Immutable Audit Ledger (SHA-256 Hash-Chained)</div>
        {records.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #eee', background: '#fafafa' }}>
                {['#', 'Batch ID', 'Timestamp', 'Decision', 'Carbon (kg)', 'Power (kW)', 'ISO Scope', 'Hash (16)'].map(h => (
                  <th key={h} style={{ padding: '0.5rem', textAlign: 'left', color: '#888', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <td style={{ padding: '0.4rem 0.5rem' }}>{r.index}</td>
                  <td style={{ padding: '0.4rem 0.5rem', fontFamily: 'monospace', fontSize: '0.75rem' }}>{r.batch_id}</td>
                  <td style={{ padding: '0.4rem 0.5rem', fontSize: '0.75rem' }}>{r.timestamp?.slice(0, 19)}</td>
                  <td style={{ padding: '0.4rem 0.5rem' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 600,
                      background: r.human_decision === 'approved' ? '#d4edda' : '#f8d7da',
                      color: r.human_decision === 'approved' ? '#155724' : '#721c24',
                    }}>{r.human_decision}</span>
                  </td>
                  <td style={{ padding: '0.4rem 0.5rem', fontWeight: 600 }}>{r.carbon_kg?.toFixed(3)}</td>
                  <td style={{ padding: '0.4rem 0.5rem' }}>{r.power_kw?.toFixed(1)}</td>
                  <td style={{ padding: '0.4rem 0.5rem', fontSize: '0.7rem' }}>{r.iso_label}</td>
                  <td style={{ padding: '0.4rem 0.5rem', fontFamily: 'monospace', fontSize: '0.7rem', color: '#888' }}>
                    {r.hash?.slice(0, 16)}...
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p style={{ color: '#999', fontSize: '0.85rem' }}>No audit records yet. Run and approve a batch to generate records.</p>}
      </div>

      {/* Batch History + Quality Delta */}
      <div style={{ ...card, marginBottom: '1rem', overflowX: 'auto' }}>
        <div style={label}>Batch Records</div>
        {batchRecords.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #eee', background: '#fafafa' }}>
                {['Batch ID', 'Quality Delta', 'Carbon (kg)', 'Qdrant Updated', 'Status', 'Feedback'].map(h => (
                  <th key={h} style={{ padding: '0.5rem', textAlign: 'left', color: '#888', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {batchRecords.slice().reverse().map((b, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f0f0f0' }}>
                  <td style={{ padding: '0.4rem 0.5rem', fontFamily: 'monospace', fontSize: '0.75rem' }}>{b.batch_id}</td>
                  <td style={{
                    padding: '0.4rem 0.5rem', fontWeight: 600,
                    color: (b.quality_delta || 0) > 0 ? '#27ae60' : '#e74c3c',
                  }}>{(b.quality_delta || 0).toFixed(4)}</td>
                  <td style={{ padding: '0.4rem 0.5rem' }}>{b.carbon_metrics?.carbon_kg?.toFixed(3) || '--'}</td>
                  <td style={{ padding: '0.4rem 0.5rem' }}>{b.qdrant_updated ? 'Yes' : 'No'}</td>
                  <td style={{ padding: '0.4rem 0.5rem' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 600,
                      background: b.human_approved ? '#d4edda' : '#f8d7da',
                      color: b.human_approved ? '#155724' : '#721c24',
                    }}>{b.human_approved ? 'Approved' : 'Rejected'}</span>
                  </td>
                  <td style={{ padding: '0.4rem 0.5rem', fontSize: '0.75rem', color: '#888', maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {b.human_feedback || '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p style={{ color: '#999', fontSize: '0.85rem' }}>No batch records yet.</p>}
      </div>

      {/* Regulatory Settings */}
      <div style={card}>
        <div style={label}>Regulatory Settings</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem' }}>
          {[
            { k: 'max_carbon_per_batch_kg', l: 'Max Carbon/Batch (kg)', step: 0.5 },
            { k: 'max_power_per_batch_kwh', l: 'Max Power/Batch (kWh)', step: 1 },
            { k: 'min_yield_pct', l: 'Min Yield (%)', step: 1 },
            { k: 'min_hardness', l: 'Min Hardness (kP)', step: 0.5 },
            { k: 'max_friability', l: 'Max Friability', step: 0.1 },
          ].map(({ k, l, step }) => (
            <div key={k}>
              <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '0.25rem' }}>{l}</div>
              <input type="number" step={step} value={regForm[k] || ''} onChange={e => setRegForm(p => ({ ...p, [k]: +e.target.value }))}
                style={{
                  width: '100%', padding: '0.5rem', border: '1px solid #ddd', borderRadius: '6px',
                  fontSize: '0.9rem', fontFamily: 'inherit',
                }}
              />
            </div>
          ))}
        </div>
        <button onClick={saveRegulatory} style={{
          marginTop: '1rem', padding: '0.5rem 1.5rem', background: '#e67e22', color: '#fff',
          border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer',
        }}>Save Regulatory Targets</button>
      </div>
    </div>
  );
}
