import React, { useState, useEffect } from 'react';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';

export default function Dashboard() {
  const [gameState, setGameState] = useState(null);
  const [error, setError] = useState(null);
  const [batchId, setBatchId] = useState(null);
  const [isStarting, setIsStarting] = useState(false);

  const handleNewBatch = async () => {
    setIsStarting(true);
    setGameState(null);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/new_batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Failed");
      setBatchId(result.batch_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsStarting(false);
    }
  };

  useEffect(() => {
    let active = true;
    const pollBackend = async () => {
      try {
        const queryId = batchId ? batchId : 'LATEST_KNOWN';
        const res = await fetch(`http://127.0.0.1:8000/api/graph_state?batch_id=${queryId}`);
        if (!active) return;
        if (res.ok) {
          const data = await res.json();
          if (data.status !== "not_found") {
            setGameState(data);
            if (!batchId && data.batch_id) setBatchId(data.batch_id);
            setError(null);
          }
        }
      } catch (err) {
        if (active) setError(err.message);
      }
    };
    pollBackend();
    const interval = setInterval(pollBackend, 1500);
    return () => { active = false; clearInterval(interval); };
  }, [batchId]);

  const liveTelemetry = gameState?.current_telemetry || {};
  const baselineValues = gameState?.historical_baseline || {};
  const healthScore = gameState?.asset_health_score || 100;
  const carbonMetrics = gameState?.carbon_metrics || {};
  const outcome = gameState?.simulated_outcome || {};
  const qualityDelta = gameState?.quality_delta || 0;
  const baselineScore = gameState?.baseline_score || 0;

  // Real values from backend
  const sysHealth = healthScore.toFixed(1);
  const tabletWeight = outcome?.Tablet_Weight;
  const yieldDisplay = tabletWeight ? tabletWeight.toFixed(1) : '---';
  const matchScore = baselineScore ? (baselineScore * 100).toFixed(1) : '---';

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '2rem', letterSpacing: '-0.5px' }}>System Overview</h1>
          <p style={{ color: '#666', marginTop: '0.5rem' }}>{batchId ? `Active Batch: ${batchId}` : 'No active batch'}</p>
        </div>
        <button 
          onClick={handleNewBatch}
          disabled={isStarting}
          style={{ 
            backgroundColor: '#1152d4', color: 'white', border: 'none', padding: '0.75rem 1.5rem', 
            borderRadius: '6px', fontWeight: 600, cursor: isStarting ? 'not-allowed' : 'pointer',
            opacity: isStarting ? 0.7 : 1
          }}>
          {isStarting ? "Initializing..." : "Run New Batch"}
        </button>
      </div>
      
      {error && <div style={{ color: '#d32f2f', padding: '1rem', backgroundColor: '#ffebee', borderRadius: '4px' }}>{error}</div>}

      {/* Top KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Asset Health</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem', color: healthScore < 85 ? '#d32f2f' : healthScore < 95 ? '#ed6c02' : '#1a1a1a' }}>
            {sysHealth}%
          </div>
          <div style={{ width: '100%', height: '4px', backgroundColor: '#eee', borderRadius: '2px', marginTop: '0.5rem' }}>
            <div style={{ width: `${healthScore}%`, height: '100%', backgroundColor: healthScore < 85 ? '#d32f2f' : healthScore < 95 ? '#ed6c02' : '#2e7d32', borderRadius: '2px', transition: 'width 0.5s ease' }} />
          </div>
        </div>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Tablet Weight</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {yieldDisplay}
            {qualityDelta !== 0 && (
              <span style={{ fontSize: '0.9rem', color: qualityDelta > 0 ? '#2e7d32' : '#d32f2f', display: 'flex', alignItems: 'center', fontWeight: 600 }}>
                {qualityDelta > 0 ? <ArrowUpward fontSize="inherit" /> : <ArrowDownward fontSize="inherit" />}
                {qualityDelta > 0 ? '+' : ''}{(qualityDelta * 100).toFixed(2)}%
              </span>
            )}
          </div>
        </div>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Carbon / Batch</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {(carbonMetrics?.carbon_kg || 0).toFixed(1)}
            <span style={{ fontSize: '1rem', color: '#666', fontWeight: 400 }}>kgCO₂</span>
          </div>
        </div>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Qdrant Match</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem', color: baselineScore < 0.85 ? '#d32f2f' : '#1a1a1a' }}>
            {matchScore}
            <span style={{ fontSize: '1rem', color: '#666', fontWeight: 400 }}>%</span>
          </div>
          {baselineScore > 0 && (
            <div style={{ fontSize: '0.75rem', color: baselineScore < 0.85 ? '#d32f2f' : '#2e7d32', fontWeight: 600, marginTop: '0.25rem' }}>
              {baselineScore < 0.85 ? '⚠ Low confidence' : '✓ Strong match'}
            </div>
          )}
        </div>
      </div>

      {/* Telemetry Signature */}
      <div style={{ backgroundColor: 'white', borderRadius: '12px', padding: '2rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
        <h2 style={{ margin: '0 0 1.5rem 0', fontSize: '1.25rem' }}>Telemetry Signature</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem' }}>
          <div>
            <div style={{ color: '#666', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Temperature (°C)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>
              {(liveTelemetry?.Temperature_C || 0).toFixed(1)}
            </div>
            {baselineValues?.ctx_Temperature_C && (
              <div style={{ fontSize: '0.85rem', color: liveTelemetry.Temperature_C > baselineValues.ctx_Temperature_C ? '#d32f2f' : '#2e7d32', marginTop: '0.25rem' }}>
                vs Baseline: {(liveTelemetry.Temperature_C - baselineValues.ctx_Temperature_C).toFixed(1)}°C
              </div>
            )}
          </div>
          <div>
            <div style={{ color: '#666', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Vibration (mm/s)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>
              {(liveTelemetry?.Vibration_mm_s || 0).toFixed(2)}
            </div>
          </div>
          <div>
            <div style={{ color: '#666', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Speed (RPM)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>
              {(liveTelemetry?.Motor_Speed_RPM || 0).toFixed(1)}
            </div>
          </div>
          <div>
            <div style={{ color: '#666', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Power (kW)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>
              {(liveTelemetry?.Power_Consumption_kW || 0).toFixed(1)}
            </div>
          </div>
        </div>
      </div>

      {/* Energy Recommendations */}
      {gameState?.energy_recommendations && gameState.energy_recommendations.length > 0 && (
        <div style={{ backgroundColor: 'white', borderRadius: '12px', padding: '2rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <h2 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem' }}>Energy & Maintenance Alerts</h2>
          {gameState.energy_recommendations.map((rec, i) => (
            <div key={i} style={{
              padding: '0.75rem 1rem', marginBottom: '0.5rem', borderRadius: '6px', fontSize: '0.85rem',
              backgroundColor: rec.startsWith('URGENT') ? '#ffebee' : rec.startsWith('MONITOR') ? '#fff3e0' : '#e8f5e9',
              borderLeft: `3px solid ${rec.startsWith('URGENT') ? '#d32f2f' : rec.startsWith('MONITOR') ? '#ed6c02' : '#2e7d32'}`,
              color: '#333', lineHeight: 1.5,
            }}>
              {rec}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
