import React from 'react';
import useStore from '../store';

const ImageGrid = () => {
  const { points, selectedIds, toggleSelection, hoveredId, setHoveredId } = useStore();

  return (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', 
      gap: '16px', 
      padding: '16px',
      background: '#1e1e1e',
      minHeight: '100%'
    }}>
      {points.map(point => (
        <div 
          key={point.id}
          style={{ 
            border: selectedIds.has(point.id) ? '2px solid #ff0000' : (hoveredId === point.id ? '2px solid #ffffff' : '2px solid transparent'),
            cursor: 'pointer',
            aspectRatio: '1',
            overflow: 'hidden',
            borderRadius: '8px',
            background: '#333',
            position: 'relative',
            transition: 'all 0.2s ease'
          }}
          onClick={() => toggleSelection(point.id)}
          onMouseEnter={() => setHoveredId(point.id)}
          onMouseLeave={() => setHoveredId(null)}
        >
          <img 
            src={point.image_url} 
            alt={point.id} 
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            loading="lazy"
          />
          {/* Overlay for selection/hover state if needed, but border is good */}
        </div>
      ))}
    </div>
  );
};

export default ImageGrid;
