import React, { useState, useEffect } from 'react';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';

export default function Optimization() {
  const [priority, setPriority] = useState(50);
  const [gameState, setGameState] = useState(null);

  // Poll state to get latest yield/energy
  useEffect(() => {
    let active = true;
    const pollBackend = async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/graph_state?batch_id=LATEST_KNOWN`);
        if (!active) return;
        if (res.ok) {
          const data = await res.json();
          if (data.status !== "not_found") {
            setGameState(data);
          }
        }
      } catch {
        // Ignored in loop
      }
    };
    pollBackend();
    const interval = setInterval(pollBackend, 1500);
    return () => { active = false; clearInterval(interval); };
  }, []);

  const handleApplyStrategy = async () => {
    try {
      await fetch('http://127.0.0.1:8000/api/update_priorities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority_value: priority, priority_type: 'yield_vs_energy' })
      });
      alert('Strategy Applied successfully!');
    } catch (err) {
      alert('Failed to apply strategy: ' + err.message);
    }
  };

  const currentYield = gameState ? (92.0 + (gameState.current_telemetry?.Pressure_Bar || 0) * 0.5).toFixed(1) : '94.2';
  const theoreticYield = gameState ? (parseFloat(currentYield) + (priority - 50) * 0.05).toFixed(1) : (94.2 + (priority - 50) * 0.05).toFixed(1);

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
            {gameState ? (gameState.current_telemetry?.Power_Consumption_kW?.toFixed(0) || '428') : '428'} <span style={{ fontSize: '1rem', color: '#666', fontWeight: 400 }}>kwh/unit</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Carbon Intensity</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, marginTop: '0.5rem', display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            {gameState ? (0.8 + ((priority - 50) * -0.01)).toFixed(2) : '0.82'} <span style={{ fontSize: '0.9rem', color: '#666', display: 'flex', alignItems: 'center', fontWeight: 600 }}><ArrowDownward fontSize="inherit" /> 4.5%</span>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '4rem' }}>
        {/* Sliders Area */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem' }}>
            <h2 style={{ fontSize: '1rem', letterSpacing: '1px', textTransform: 'uppercase', margin: 0 }}>Priority Balancing</h2>
            <span style={{ backgroundColor: '#f0f4ff', color: '#1152d4', padding: '0.25rem 0.75rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>LIVE SIM</span>
          </div>

          <div style={{ marginBottom: '4rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '1.5rem', color: '#444' }}>
              <span>MAXIMIZE YIELD</span>
              <span>MINIMIZE ENERGY</span>
            </div>
            <input 
              type="range"
              min="0" max="100"
              value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer', accentColor: '#1a1a1a', height: '4px' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '1rem', fontStyle: 'italic' }}>
              PRIORITIZING {priority > 50 ? 'LOW CONSUMPTION AND GRID HEALTH' : 'YIELD OUTPUT AND VOLUME'}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <button onClick={handleApplyStrategy} style={{ backgroundColor: '#131b2f', color: 'white', border: 'none', padding: '1rem 2.5rem', fontSize: '0.85rem', fontWeight: 700, letterSpacing: '1px', cursor: 'pointer' }}>
              APPLY STRATEGY
            </button>
            <button onClick={() => setPriority(50)} style={{ backgroundColor: 'transparent', color: '#888', border: 'none', padding: '1rem 2rem', fontSize: '0.85rem', fontWeight: 700, letterSpacing: '1px', cursor: 'pointer' }}>
              RESET
            </button>
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
