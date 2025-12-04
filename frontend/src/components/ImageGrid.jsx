import React from 'react';
import useStore from '../store';

const ImageGrid = () => {
  const { points, selectedIds, toggleSelection, hoveredId, setHoveredId } = useStore();

  return (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', 
      gap: '12px', 
      padding: '20px',
      background: '#1e1e1e',
      minHeight: '100%'
    }}>
      {points.map(point => {
        const isSelected = selectedIds.has(point.id);
        const isHovered = hoveredId === point.id;
        
        return (
          <div 
            key={point.id}
            style={{ 
              border: isSelected 
                ? '3px solid #4a9eff' 
                : isHovered 
                  ? '3px solid rgba(255, 255, 255, 0.5)' 
                  : '3px solid transparent',
              cursor: 'pointer',
              aspectRatio: '1',
              overflow: 'hidden',
              borderRadius: '6px',
              background: '#2a2a2a',
              position: 'relative',
              transition: 'all 0.15s ease',
              boxShadow: isSelected 
                ? '0 0 0 2px rgba(74, 158, 255, 0.3)' 
                : isHovered 
                  ? '0 4px 12px rgba(0, 0, 0, 0.4)'
                  : '0 2px 4px rgba(0, 0, 0, 0.2)',
              transform: isHovered ? 'scale(1.02)' : 'scale(1)'
            }}
            onClick={() => toggleSelection(point.id)}
            onMouseEnter={() => setHoveredId(point.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            <img 
              src={point.image_url} 
              alt={point.id} 
              style={{ 
                width: '100%', 
                height: '100%', 
                objectFit: 'cover', 
                display: 'block',
                opacity: isSelected ? 1 : isHovered ? 0.95 : 0.9
              }}
              loading="lazy"
            />
            {isSelected && (
              <div style={{
                position: 'absolute',
                top: '6px',
                right: '6px',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: '#4a9eff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '12px',
                fontWeight: 'bold',
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)'
              }}>
                ✓
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ImageGrid;
