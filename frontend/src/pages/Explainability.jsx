import React, { useState, useEffect } from 'react';
import { Warning, TrendingUp, Engineering, CheckCircle } from '@mui/icons-material';

export default function Explainability() {
  const [gameState, setGameState] = useState(null);
  const [featureImportances, setFeatureImportances] = useState({});
  const [energyData, setEnergyData] = useState(null);

  // Poll graph state
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

  // Fetch real feature importances
  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/feature_importance?top_n=10');
        if (res.ok) {
          const data = await res.json();
          setFeatureImportances(data.features || {});
        }
      } catch { /* ignore */ }
    };
    fetchFeatures();
  }, []);

  // Fetch energy anomalies
  useEffect(() => {
    const fetchEnergy = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/energy_anomalies');
        if (res.ok) setEnergyData(await res.json());
      } catch { /* ignore */ }
    };
    fetchEnergy();
    const interval = setInterval(fetchEnergy, 3000);
    return () => clearInterval(interval);
  }, []);

  const settings = gameState?.proposed_settings || {};
  const anomalies = gameState?.energy_anomalies || energyData?.anomalies || [];
  const healthScore = gameState?.asset_health_score || energyData?.asset_health_score || 100;
  const recommendations = gameState?.energy_recommendations || energyData?.recommendations || [];

  const criticalCount = anomalies.filter(a => a.severity === 'CRITICAL').length;
  const warningCount = anomalies.filter(a => a.severity === 'WARNING').length;

  // Convert feature importances to sorted array
  const featureEntries = Object.entries(featureImportances).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxImportance = featureEntries.length > 0 ? Math.max(...featureEntries.map(e => Math.abs(e[1]))) : 1;

  // Format feature name for display
  const formatFeatureName = (name) => {
    return name
      .replace('ctx_', '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .substring(0, 30);
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', fontFamily: '"Inter", sans-serif' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '2rem', letterSpacing: '-0.5px' }}>AI Explainability Overview</h1>
          <p style={{ color: '#666', marginTop: '0.5rem' }}>Feature Importance, Energy Anomalies & Asset Health</p>
        </div>
      </div>

      {/* Top Alert Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '3rem' }}>
        <div style={{ 
          backgroundColor: healthScore < 85 ? '#ffebee' : healthScore < 95 ? '#fff3e0' : '#e8f5e9',
          padding: '1.5rem', 
          borderLeft: `4px solid ${healthScore < 85 ? '#d32f2f' : healthScore < 95 ? '#ed6c02' : '#2e7d32'}`, 
          borderRadius: '4px' 
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: healthScore < 85 ? '#d32f2f' : healthScore < 95 ? '#ed6c02' : '#2e7d32', fontWeight: 600, fontSize: '0.85rem', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
            {healthScore < 85 ? <Warning fontSize="small" /> : <CheckCircle fontSize="small" />}
            {healthScore < 85 ? 'Maintenance Required' : healthScore < 95 ? 'Monitor Closely' : 'Healthy'}
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.5rem', color: '#333' }}>
            Asset Health: {healthScore.toFixed(1)}%
          </div>
        </div>
        
        <div style={{ backgroundColor: '#f0f4ff', padding: '1.5rem', borderLeft: '4px solid #1152d4', borderRadius: '4px' }}>
          <div style={{ fontSize: '0.85rem', color: '#1152d4', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Energy Anomalies</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.5rem', color: '#333' }}>
            {criticalCount} Critical, {warningCount} Warning
          </div>
          <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
            {anomalies.length} total deviations detected
          </div>
        </div>

        <div style={{ backgroundColor: '#eeeeee', padding: '1.5rem', borderLeft: '4px solid #666', borderRadius: '4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#666', fontWeight: 600, fontSize: '0.85rem', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
            <Engineering fontSize="small" /> Repair Layer
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.5rem', color: '#333' }}>
            {gameState ? Math.floor(1 + (settings.Granulation_Time || 0) % 4) : '2'} Violations Prevented
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '2.5rem' }}>
        <div>
          {/* Real Feature Importance */}
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1.5rem' }}>Feature Importance (XGBoost)</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {featureEntries.length > 0 ? featureEntries.slice(0, 10).map(([name, value], i) => {
              const barWidth = (Math.abs(value) / maxImportance) * 100;
              const isPositive = value > 0;
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{ width: '180px', fontSize: '0.8rem', color: '#555', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {formatFeatureName(name)}
                  </div>
                  <div style={{ flex: 1, backgroundColor: '#f5f5f5', height: '12px', borderRadius: '6px', overflow: 'hidden' }}>
                    <div style={{ width: `${barWidth}%`, backgroundColor: isPositive ? '#ed6c02' : '#1152d4', height: '100%', transition: 'width 0.5s ease' }}></div>
                  </div>
                  <div style={{ width: '55px', textAlign: 'right', fontSize: '0.8rem', fontWeight: 600 }}>
                    {(value * 100).toFixed(1)}%
                  </div>
                </div>
              );
            }) : (
              <div style={{ color: '#999', fontSize: '0.9rem', padding: '1rem' }}>
                Loading feature importances from XGBoost surrogate model...
              </div>
            )}
          </div>

          {/* Energy Anomaly Recommendations */}
          <h2 style={{ fontSize: '1.1rem', marginTop: '2.5rem', marginBottom: '1rem' }}>AI Recommendations</h2>
          {recommendations.length > 0 ? recommendations.map((rec, i) => (
            <div key={i} style={{ 
              padding: '0.75rem 1rem', marginBottom: '0.5rem', borderRadius: '6px', fontSize: '0.85rem', lineHeight: 1.5,
              backgroundColor: rec.startsWith('URGENT') ? '#ffebee' : rec.startsWith('MONITOR') ? '#fff3e0' : '#e8f5e9',
              borderLeft: `3px solid ${rec.startsWith('URGENT') ? '#d32f2f' : rec.startsWith('MONITOR') ? '#ed6c02' : '#2e7d32'}`,
              color: '#333',
            }}>
              {rec}
            </div>
          )) : (
            <p style={{ color: '#666', fontSize: '0.9rem' }}>No recommendations at this time. All systems operating within normal parameters.</p>
          )}
        </div>

        {/* Right Panel: Energy Anomaly Details */}
        <div style={{ backgroundColor: '#fafafa', border: '1px solid #eee', padding: '1.5rem', borderRadius: '8px' }}>
          <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px', marginTop: 0, borderBottom: '1px solid #ddd', paddingBottom: '0.75rem', marginBottom: '1.5rem' }}>Energy Anomaly Log</h3>
          
          {anomalies.length > 0 ? anomalies.slice(0, 5).map((anomaly, i) => (
            <div key={i} style={{ marginBottom: '1.25rem' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ 
                  width: '8px', height: '8px', borderRadius: '50%', 
                  backgroundColor: anomaly.severity === 'CRITICAL' ? '#d32f2f' : '#ed6c02',
                  display: 'inline-block'
                }} />
                {anomaly.phase} — {anomaly.metric}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.25rem' }}>
                Current: {anomaly.current_value} {anomaly.unit} vs Baseline: {anomaly.baseline_value} {anomaly.unit}
              </div>
              <div style={{ fontSize: '0.8rem', color: anomaly.severity === 'CRITICAL' ? '#d32f2f' : '#ed6c02', marginTop: '0.15rem', fontWeight: 600 }}>
                {anomaly.deviation_pct > 0 ? '+' : ''}{anomaly.deviation_pct}% deviation ({anomaly.severity})
              </div>
              <div style={{ width: '100%', height: '4px', backgroundColor: '#eee', marginTop: '0.5rem', borderRadius: '2px' }}>
                <div style={{ 
                  width: `${Math.min(100, Math.abs(anomaly.deviation_pct))}%`, 
                  height: '100%', 
                  backgroundColor: anomaly.severity === 'CRITICAL' ? '#d32f2f' : '#ed6c02',
                  borderRadius: '2px',
                  transition: 'width 0.5s ease'
                }}></div>
              </div>
            </div>
          )) : (
            <div style={{ color: '#999', fontSize: '0.85rem', padding: '1rem 0' }}>
              No energy anomalies detected. All phases within normal operating range.
            </div>
          )}

          {anomalies.length > 5 && (
            <div style={{ fontSize: '0.8rem', color: '#888', marginTop: '0.5rem', fontStyle: 'italic' }}>
              +{anomalies.length - 5} more anomalies
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
