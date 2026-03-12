import React, { useState, useEffect } from 'react';

export default function Execution() {
  const [gameState, setGameState] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  const handleDecision = async (approved) => {
    if (!gameState?.batch_id) return;
    setIsSubmitting(true);
    try {
      await fetch('http://127.0.0.1:8000/api/execute_decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          batch_id: gameState.batch_id,
          approved: approved,
          feedback: approved ? "Approved" : "Rejected"
        })
      });
    } catch (err) {
      alert("Failed to submit decision: " + err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isComplete = gameState?.status === "executed" || gameState?.status === "rejected";
  const settings = gameState?.proposed_settings || {};
  const delta = gameState?.quality_delta || 0;

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', fontFamily: '"Inter", sans-serif' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '2rem', letterSpacing: '-0.5px' }}>Execution Management</h1>
          <p style={{ color: '#666', marginTop: '0.5rem' }}>Human-in-the-Loop Gateway</p>
        </div>
      </div>

      <div style={{ backgroundColor: 'white', border: '1px solid #e0e0e0', borderRadius: '12px', padding: '2.5rem', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eee', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
          <div>
            <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Workflow Status</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.25rem', color: isComplete ? (gameState.status === 'executed' ? '#2e7d32' : '#d32f2f') : '#f57c00' }}>
              {isComplete ? (gameState.status === 'executed' ? 'Executed Successfully' : 'Rejected by Operator') : (gameState?.paused_for_hitl ? 'APPROVAL REQUIRED' : 'Pending Synthesis')}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Batch ID</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: '0.25rem', fontFamily: '"SF Mono", monospace' }}>
              {gameState?.batch_id || '---'}
            </div>
          </div>
        </div>

        <h3 style={{ fontSize: '1rem', marginTop: 0, marginBottom: '1.5rem', color: '#1a1a1a' }}>Proposed Parameters</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '3rem' }}>
          <div style={{ backgroundColor: '#f9f9f9', padding: '1.25rem', borderRadius: '8px' }}>
             <div style={{ fontSize: '0.85rem', color: '#666' }}>Granulation Time</div>
             <div style={{ fontSize: '1.5rem', fontWeight: 600, marginTop: '0.25rem' }}>{(settings.Granulation_Time || 0).toFixed(1)} s</div>
          </div>
          <div style={{ backgroundColor: '#f9f9f9', padding: '1.25rem', borderRadius: '8px' }}>
             <div style={{ fontSize: '0.85rem', color: '#666' }}>Binder Amount</div>
             <div style={{ fontSize: '1.5rem', fontWeight: 600, marginTop: '0.25rem' }}>{(settings.Binder_Amount || 0).toFixed(1)} L</div>
          </div>
          <div style={{ backgroundColor: '#f9f9f9', padding: '1.25rem', borderRadius: '8px' }}>
             <div style={{ fontSize: '0.85rem', color: '#666' }}>Drying Temp</div>
             <div style={{ fontSize: '1.5rem', fontWeight: 600, marginTop: '0.25rem' }}>{(settings.Drying_Temp || 0).toFixed(1)} °C</div>
          </div>
          <div style={{ backgroundColor: '#f9f9f9', padding: '1.25rem', borderRadius: '8px' }}>
             <div style={{ fontSize: '0.85rem', color: '#666' }}>Machine Speed</div>
             <div style={{ fontSize: '1.5rem', fontWeight: 600, marginTop: '0.25rem' }}>{(settings.Machine_Speed || 0).toFixed(1)} RPM</div>
          </div>
        </div>

        {isComplete ? (
          <div style={{ backgroundColor: gameState.status === 'executed' ? '#e8f5e9' : '#ffebee', padding: '1.5rem', borderRadius: '8px', border: `1px solid ${gameState.status === 'executed' ? '#c8e6c9' : '#ffcdd2'}` }}>
            <h4 style={{ margin: '0 0 0.5rem 0', color: gameState.status === 'executed' ? '#2e7d32' : '#d32f2f' }}>
              {gameState.status === 'executed' ? 'Settings Applied to Floor' : 'Batch Discarded'}
            </h4>
            <p style={{ margin: 0, color: '#444', fontSize: '0.95rem' }}>
              {gameState.status === 'executed' ? `Quality Delta vs Baseline: ${delta > 0 ? '+' : ''}${delta.toFixed(4)}. Qdrant Vector DB updated: ${gameState.qdrant_updated ? 'Yes' : 'No'}.` : 'No changes were made to the baseline parameters.'}
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button 
              onClick={() => handleDecision(true)}
              disabled={isSubmitting || !gameState?.paused_for_hitl}
              style={{
                flex: 1, backgroundColor: '#1152d4', color: 'white', padding: '1.25rem', border: 'none', borderRadius: '8px',
                fontSize: '1rem', fontWeight: 600, cursor: (!gameState?.paused_for_hitl || isSubmitting) ? 'not-allowed' : 'pointer',
                opacity: (!gameState?.paused_for_hitl || isSubmitting) ? 0.5 : 1
              }}>
              {isSubmitting ? 'PROCESSING...' : 'APPROVE & EXECUTE'}
            </button>
            <button 
              onClick={() => handleDecision(false)}
              disabled={isSubmitting || !gameState?.paused_for_hitl}
              style={{
                flex: 1, backgroundColor: 'transparent', color: '#d32f2f', padding: '1.25rem', border: '1px solid #d32f2f', borderRadius: '8px',
                fontSize: '1rem', fontWeight: 600, cursor: (!gameState?.paused_for_hitl || isSubmitting) ? 'not-allowed' : 'pointer',
                opacity: (!gameState?.paused_for_hitl || isSubmitting) ? 0.5 : 1
              }}>
              REJECT / OVERRIDE
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
