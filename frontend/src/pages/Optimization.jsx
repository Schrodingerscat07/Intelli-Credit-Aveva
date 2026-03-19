import React, { useState, useEffect } from 'react';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';

const OBJECTIVE_PRESETS = [
  { label: 'Max Yield + Min Energy', primary: 'Tablet_Weight', secondary: 'Power_Consumption_kW' },
  { label: 'Best Quality + Max Yield', primary: 'Hardness', secondary: 'Tablet_Weight' },
  { label: 'Max Performance + Min Carbon', primary: 'Tablet_Weight', secondary: 'Carbon_Emissions' },
  { label: 'Min Energy + Min Friability', primary: 'Power_Consumption_kW', secondary: 'Friability' },
];

export default function Optimization() {
  const [priority, setPriority] = useState(50);
  const [selectedPreset, setSelectedPreset] = useState(0);
  const [gameState, setGameState] = useState(null);
  const [carbonMetrics, setCarbonMetrics] = useState(null);
  const [confirmation, setConfirmation] = useState({ show: false, message: '', type: '' });

  // Poll state
  useEffect(() => {
    let active = true;
    const pollBackend = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/graph_state?batch_id=LATEST_KNOWN');
        if (!active) return;
        if (res.ok) {
          const data = await res.json();
          if (data.status !== "not_found") setGameState(data);
        }
      } catch { /* ignore */ }
    };
    pollBackend();
    const interval = setInterval(pollBackend, 1500);
    return () => { active = false; clearInterval(interval); };
  }, []);

  // Fetch carbon metrics
  useEffect(() => {
    const fetchCarbon = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/carbon_metrics');
        if (res.ok) setCarbonMetrics(await res.json());
      } catch { /* ignore */ }
    };
    fetchCarbon();
    const interval = setInterval(fetchCarbon, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleApplyStrategy = async () => {
    const preset = OBJECTIVE_PRESETS[selectedPreset];
    try {
      await fetch('http://127.0.0.1:8000/api/update_priorities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          priority_value: priority,
          priority_type: 'yield_vs_energy',
          objective_primary: preset.primary,
          objective_secondary: preset.secondary,
        })
      });
      setConfirmation({ show: true, message: 'Strategy Applied successfully!', type: 'success' });
      setTimeout(() => setConfirmation({ show: false, message: '', type: '' }), 3000);
    } catch (err) {
      setConfirmation({ show: true, message: 'Failed to apply strategy: ' + err.message, type: 'error' });
      setTimeout(() => setConfirmation({ show: false, message: '', type: '' }), 4000);
    }
  };

  const rawYield = gameState ? (92.0 + (gameState.current_telemetry?.Pressure_Bar || 0) * 0.5) : 94.2;
  const currentYield = rawYield.toFixed(1);
  const theoreticYield = (rawYield + ((priority - 50) * 0.08)).toFixed(1);
  const rawPower = gameState ? (gameState.current_telemetry?.Power_Consumption_kW || 428) : 428;
  const currentPower = (rawPower - ((priority - 50) * 1.5)).toFixed(0);

  // Real carbon metrics
  const carbonKg = gameState?.carbon_metrics?.carbon_kg || carbonMetrics?.avg_carbon_per_batch || 0;
  const carbonIntensity = carbonKg > 0 ? (carbonKg / Math.max(rawPower * 1.5, 0.01)).toFixed(3) : '0.000';
  const cumulativeCarbon = carbonMetrics?.cumulative_carbon_kg || 0;

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', fontFamily: '"Inter", sans-serif' }}>
      
      {/* Top Metrics Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eee', paddingBottom: '2rem', marginBottom: '3rem' }}>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Current Yield</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {currentYield}% <span style={{ fontSize: '0.9rem', color: '#2e7d32', display: 'flex', alignItems: 'center', fontWeight: 600 }}><ArrowUpward fontSize="inherit" /> 1.2%</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Energy Footprint</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {currentPower} <span style={{ fontSize: '1rem', color: '#666', fontWeight: 400 }}>kwh/unit</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Carbon / Batch</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {carbonKg.toFixed(2)} <span style={{ fontSize: '0.9rem', color: '#666', display: 'flex', alignItems: 'center', fontWeight: 600 }}>kgCO₂</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.25rem' }}>
            Cumulative: {cumulativeCarbon.toFixed(1)} kgCO₂
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '4rem' }}>
        {/* Left: Sliders + Objective Selector */}
        <div style={{ flex: 1 }}>
          {/* NEW: Multi-Target Objective Selector */}
          <div style={{ marginBottom: '2.5rem' }}>
            <h2 style={{ fontSize: '1rem', letterSpacing: '1px', textTransform: 'uppercase', margin: '0 0 1rem 0' }}>Optimization Objective</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {OBJECTIVE_PRESETS.map((preset, i) => (
                <label key={i} style={{
                  display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1rem',
                  borderRadius: '8px', cursor: 'pointer',
                  backgroundColor: selectedPreset === i ? '#f0f4ff' : '#fafafa',
                  border: `1px solid ${selectedPreset === i ? '#1152d4' : '#eee'}`,
                  transition: 'all 0.2s',
                }}>
                  <input
                    type="radio" name="objective" checked={selectedPreset === i}
                    onChange={() => setSelectedPreset(i)}
                    style={{ accentColor: '#1152d4' }}
                  />
                  <span style={{ fontWeight: selectedPreset === i ? 600 : 400, fontSize: '0.9rem' }}>{preset.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Priority Slider */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <h2 style={{ fontSize: '1rem', letterSpacing: '1px', textTransform: 'uppercase', margin: 0 }}>Priority Balancing</h2>
            <span style={{ backgroundColor: '#f0f4ff', color: '#1152d4', padding: '0.25rem 0.75rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>LIVE SIM</span>
          </div>

          <div style={{ marginBottom: '3rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '1.5rem', color: '#444' }}>
              <span>MAXIMIZE {OBJECTIVE_PRESETS[selectedPreset].primary.replace('_', ' ').toUpperCase()}</span>
              <span>MINIMIZE {OBJECTIVE_PRESETS[selectedPreset].secondary.replace('_', ' ').toUpperCase()}</span>
            </div>
            <input 
              type="range" min="0" max="100" value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer', accentColor: '#1a1a1a', height: '4px' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '1rem', fontStyle: 'italic' }}>
              PRIORITIZING {priority > 50 ? OBJECTIVE_PRESETS[selectedPreset].secondary.replace('_', ' ').toUpperCase() : OBJECTIVE_PRESETS[selectedPreset].primary.replace('_', ' ').toUpperCase()}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
            <button onClick={handleApplyStrategy} style={{ backgroundColor: '#131b2f', color: 'white', border: 'none', padding: '1rem 2.5rem', fontSize: '0.85rem', fontWeight: 700, letterSpacing: '1px', cursor: 'pointer', borderRadius: '4px' }}>
              APPLY STRATEGY
            </button>
            <button onClick={() => { setPriority(50); setSelectedPreset(0); }} style={{ backgroundColor: 'transparent', color: '#888', border: 'none', padding: '1rem 2rem', fontSize: '0.85rem', fontWeight: 700, letterSpacing: '1px', cursor: 'pointer' }}>
              RESET
            </button>
            {confirmation.show && (
              <span style={{ color: confirmation.type === 'success' ? '#2e7d32' : '#d32f2f', fontSize: '0.85rem', fontWeight: 600 }}>
                {confirmation.message}
              </span>
            )}
          </div>
        </div>

        {/* Right Preview Panel */}
        <div style={{ width: '320px', backgroundColor: '#fafafa', border: '1px solid #eee', padding: '2.5rem', display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          <h3 style={{ fontSize: '0.85rem', letterSpacing: '1px', textTransform: 'uppercase', margin: 0 }}>Predicted Outcome</h3>
          
          <div>
            <div style={{ fontSize: '0.75rem', color: '#888', fontWeight: 700, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Estimated Yield</div>
            <div style={{ fontSize: '2.25rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
              {theoreticYield}% <span style={{ fontSize: '0.85rem', color: '#2e7d32', fontWeight: 600 }}>{(theoreticYield - parseFloat(currentYield)).toFixed(1) > 0 ? '+' : ''}{(theoreticYield - parseFloat(currentYield)).toFixed(1)}%</span>
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: '#888', fontWeight: 700, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Est. Carbon Impact</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
              {(carbonKg * (1 - (priority - 50) * 0.005)).toFixed(2)} <span style={{ fontSize: '0.85rem', color: '#666', fontWeight: 400 }}>kgCO₂</span>
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.75rem', color: '#888', fontWeight: 700, letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: '1rem' }}>Confidence</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ height: '4px', backgroundColor: '#1a1a1a', flex: 1 }}></div>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>{gameState ? (90 + (gameState.current_telemetry?.Vibration_mm_s || 0) % 8).toFixed(0) : '92'}%</span>
            </div>
          </div>
          
          <div style={{ fontSize: '0.75rem', lineHeight: 1.6, color: '#666', borderTop: '1px solid #eaeaea', paddingTop: '1.5rem' }}>
            <strong style={{ color: '#1a1a1a' }}>REC:</strong> {priority > 50 ? 'OPTIMIZE THERMAL REACTORS FOR LOWER POWER DRAW WHILE MAINTAINING STRUCTURAL INTEGRITY.' : 'INCREASE THERMAL ENVELOPE TO MAXIMIZE THROUGHPUT VELOCITY.'}
          </div>
        </div>
      </div>
    </div>
  );
}
