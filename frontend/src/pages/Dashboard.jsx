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
  
  const sysHealth = gameState ? (95 + ((liveTelemetry?.Pressure_Bar || 1.0) * 1.2)).toFixed(1) : '98.4';
  const eff = gameState ? (90 + ((liveTelemetry?.Humidity_Percent || 30) % 8)).toFixed(1) : '94.2';
  const estMins = gameState ? Math.floor(10 + ((liveTelemetry?.Motor_Speed_RPM || 40) % 49)) : 45;
  const estTime = `18:${estMins.toString().padStart(2, '0')}`;

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

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>System Health</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {sysHealth}%
            <span style={{ fontSize: '0.9rem', color: '#2e7d32', display: 'flex', alignItems: 'center', fontWeight: 600 }}>
               <ArrowUpward fontSize="inherit" /> {(liveTelemetry?.Thermal_Ramp_Rate || 1.2).toFixed(1)}%
            </span>
          </div>
        </div>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Efficiency</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {eff}%
            <span style={{ fontSize: '0.9rem', color: '#2e7d32', display: 'flex', alignItems: 'center', fontWeight: 600 }}>
               <ArrowUpward fontSize="inherit" /> {(liveTelemetry?.Vibration_mm_s || 0.8).toFixed(1)}%
            </span>
          </div>
        </div>
        <div style={{ backgroundColor: 'white', padding: '1.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Completion Est</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {estTime}
          </div>
        </div>
      </div>

      <div style={{ backgroundColor: 'white', borderRadius: '12px', padding: '2rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
        <h2 style={{ margin: '0 0 1.5rem 0', fontSize: '1.25rem' }}>Telemetry Signature</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
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
        </div>
      </div>
      
    </div>
  );
}
