import React from 'react';
import useStore from '../store';

const Header = () => {
  const { points, selectedIds } = useStore();

  return (
    <div style={{
      height: '56px',
      background: '#2a2a2a',
      borderBottom: '1px solid #404040',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      color: '#fff',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      flexShrink: 0
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <h1 style={{
          fontSize: '20px',
          fontWeight: 600,
          margin: 0,
          letterSpacing: '-0.5px'
        }}>
          HyperView
        </h1>
        <span style={{
          fontSize: '13px',
          color: '#999',
          padding: '4px 8px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '4px'
        }}>
          Local-First Multimodal Explorer
        </span>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '24px',
        fontSize: '13px',
        color: '#ccc'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#888' }}>Total samples:</span>
          <span style={{ fontWeight: 600, color: '#fff' }}>{points.length}</span>
        </div>
        {selectedIds.size > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: '#888' }}>Selected:</span>
            <span style={{
              fontWeight: 600,
              color: '#4a9eff',
              background: 'rgba(74, 158, 255, 0.15)',
              padding: '2px 8px',
              borderRadius: '4px'
            }}>
              {selectedIds.size}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Header;
