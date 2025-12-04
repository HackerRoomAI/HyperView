import React, { useState, useRef, useCallback, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { OrthographicView } from '@deck.gl/core';
import { Maximize2, Circle, Square, MousePointer } from 'lucide-react';
import useStore from '../store';

// Default fallback colors (used if API doesn't provide)
const DEFAULT_COLORS = [
  [65, 105, 225], [255, 140, 0], [50, 205, 50], [138, 43, 226],
  [255, 99, 71], [0, 206, 209], [255, 215, 0], [169, 169, 169],
  [255, 20, 147], [34, 139, 34], [210, 105, 30], [244, 164, 96],
  [255, 182, 193], [220, 20, 60], [70, 130, 180], [160, 82, 45],
  [0, 100, 0], [139, 69, 19], [128, 128, 128], [255, 105, 180],
];

const EmbeddingView = () => {
  const { points, labels, viewMode, setViewMode, hoveredId, setHoveredId, selectedIds, toggleSelection, selectMultiple, clearSelection } = useStore();

  // Build color map from API labels
  const colorMap = useMemo(() => {
    const map = {};
    if (labels && labels.length > 0) {
      labels.forEach(l => {
        map[l.label_id] = [l.color_r, l.color_g, l.color_b];
      });
    }
    return map;
  }, [labels]);

  // Get label name for display
  const labelNames = useMemo(() => {
    if (labels && labels.length > 0) {
      return labels.map(l => l.label_name);
    }
    return [];
  }, [labels]);

  // Get color for a label
  const getColor = useCallback((labelId) => {
    if (colorMap[labelId]) return colorMap[labelId];
    return DEFAULT_COLORS[labelId % DEFAULT_COLORS.length];
  }, [colorMap]);
  const [selectionMode, setSelectionMode] = useState('click'); // 'click' | 'lasso'
  const [isDrawing, setIsDrawing] = useState(false);
  const [lassoPoints, setLassoPoints] = useState([]);
  const [viewState, setViewState] = useState({
    target: [0, 0, 0],
    zoom: 1
  });
  const deckRef = useRef(null);
  
  // Button styling helpers
  const getButtonStyle = useCallback((isActive, type = 'tool') => ({
    background: isActive 
      ? (type === 'mode' ? 'rgba(74, 158, 255, 0.2)' : '#4a9eff')
      : 'rgba(255, 255, 255, 0.05)',
    border: type === 'mode' && isActive ? '1px solid #4a9eff' : '1px solid transparent',
    color: 'white',
    padding: type === 'mode' ? '8px 12px' : '8px 10px',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: type === 'mode' ? '6px' : '4px',
    transition: 'all 0.15s ease'
  }), []);

  // Get point position based on view mode
  const getPointPosition = useCallback((d) => {
    // Use same scale for both views so points fit
    return viewMode === 'euclidean'
      ? [d.euclidean_x * 50, d.euclidean_y * 50]
      : [d.hyperbolic_x * 100, d.hyperbolic_y * 100];
  }, [viewMode]);

  // Reset view when switching modes
  const handleViewModeChange = useCallback((mode) => {
    setViewMode(mode);
    setViewState({ target: [0, 0, 0], zoom: 1 });
  }, [setViewMode]);

  // Check if a point is inside a polygon (lasso)
  const isPointInPolygon = (point, polygon) => {
    if (polygon.length < 3) return false;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      if (((yi > point[1]) !== (yj > point[1])) &&
          (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  };

  // Handle lasso selection
  const handleMouseDown = (e) => {
    if (selectionMode !== 'lasso') return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setIsDrawing(true);
    setLassoPoints([[x, y]]);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || selectionMode !== 'lasso') return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setLassoPoints(prev => [...prev, [x, y]]);
  };

  const handleMouseUp = () => {
    if (!isDrawing || selectionMode !== 'lasso') return;
    setIsDrawing(false);

    if (lassoPoints.length > 2 && deckRef.current) {
      // Convert screen coordinates to world coordinates and check points
      const deck = deckRef.current.deck;
      const viewport = deck.getViewports()[0];

      // Convert lasso screen points to world coordinates
      const lassoWorld = lassoPoints.map(p => {
        const [x, y] = viewport.unproject(p);
        return [x, y];
      });

      // Find all points inside the lasso
      const selectedPointIds = points
        .filter(p => {
          const pos = getPointPosition(p);
          return isPointInPolygon(pos, lassoWorld);
        })
        .map(p => p.id);

      if (selectedPointIds.length > 0) {
        selectMultiple(selectedPointIds);
      }
    }

    setLassoPoints([]);
  };

  const layers = [
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data: points,
      pickable: true,
      opacity: 0.9,
      stroked: true,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 4,
      radiusMaxPixels: 8,
      lineWidthMinPixels: 2,
      getPosition: getPointPosition,
      getFillColor: d => {
        if (selectedIds.has(d.id)) return [255, 255, 255]; // Selected = White
        const label = d.label !== undefined ? d.label : 0;
        return getColor(label);
      },
      getLineColor: d => {
        if (selectedIds.has(d.id)) return [255, 0, 0]; // Red border for selected
        if (hoveredId === d.id) return [255, 255, 255];
        return [40, 40, 40];
      },
      getLineWidth: d => selectedIds.has(d.id) ? 3 : 1,
      onHover: info => setHoveredId(info.object ? info.object.id : null),
      onClick: info => {
        if (selectionMode === 'click' && info.object) {
          toggleSelection(info.object.id);
        }
      },
      updateTriggers: {
        getPosition: [viewMode],
        getFillColor: [selectedIds, colorMap],
        getLineColor: [selectedIds, hoveredId],
        getLineWidth: [selectedIds]
      }
    })
  ];

  return (
    <div
      style={{ width: '100%', height: '100%', position: 'relative', background: '#1a1a1a' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <DeckGL
        ref={deckRef}
        views={new OrthographicView({ controller: selectionMode === 'click' })}
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={selectionMode === 'click'}
        layers={layers}
        getTooltip={({object}) => object && {
          html: `<div style="background: #333; padding: 8px; border-radius: 4px; color: white;">
            <strong>ID:</strong> ${object.id}<br/>
            <strong>Label:</strong> ${object.label !== undefined ? object.label : 'N/A'}
          </div>`,
          style: { backgroundColor: 'transparent', border: 'none' }
        }}
      />

      {/* Lasso drawing overlay */}
      {isDrawing && lassoPoints.length > 1 && (
        <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }}>
          <polygon
            points={lassoPoints.map(p => p.join(',')).join(' ')}
            fill="rgba(255, 255, 255, 0.1)"
            stroke="rgba(255, 255, 255, 0.8)"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
        </svg>
      )}

      {/* Toolbar */}
      <div style={{
        position: 'absolute',
        top: 12,
        left: 12,
        background: 'rgba(42, 42, 42, 0.98)',
        padding: '6px',
        borderRadius: '8px',
        zIndex: 100,
        display: 'flex',
        gap: '4px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
      }}>
        {/* Selection Mode */}
        <button
          onClick={() => setSelectionMode('click')}
          style={getButtonStyle(selectionMode === 'click', 'tool')}
          title="Click to select"
          className="toolbar-button"
        >
          <MousePointer size={16} />
        </button>
        <button
          onClick={() => setSelectionMode('lasso')}
          style={getButtonStyle(selectionMode === 'lasso', 'tool')}
          title="Lasso select"
          className="toolbar-button"
        >
          <Square size={16} />
        </button>

        <div style={{ width: '1px', background: 'rgba(255, 255, 255, 0.15)', margin: '0 6px' }} />

        {/* View Mode */}
        <button
          onClick={() => handleViewModeChange('euclidean')}
          style={getButtonStyle(viewMode === 'euclidean', 'mode')}
          title="Euclidean View"
          className="toolbar-button"
        >
          <Maximize2 size={16} />
          <span style={{fontSize: '12px', fontWeight: 500}}>Euclidean</span>
        </button>
        <button
          onClick={() => handleViewModeChange('hyperbolic')}
          style={getButtonStyle(viewMode === 'hyperbolic', 'mode')}
          title="Hyperbolic View (Poincaré Disk)"
          className="toolbar-button"
        >
          <Circle size={16} />
          <span style={{fontSize: '12px', fontWeight: 500}}>Hyperbolic</span>
        </button>
      </div>

      {/* Legend */}
      {labels && labels.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 12,
          right: 12,
          background: 'rgba(42, 42, 42, 0.98)',
          padding: '10px 14px',
          borderRadius: '8px',
          color: 'white',
          fontSize: '11px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          zIndex: 100,
          maxHeight: '400px',
          overflowY: 'auto',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
        }}>
          <div style={{ 
            marginBottom: '8px', 
            fontWeight: 600, 
            fontSize: '12px',
            color: '#ccc'
          }}>
            Categories
          </div>
          {labels.map((l, i) => (
            <div key={l.label_id} style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              marginBottom: i < labels.length - 1 ? '6px' : 0,
              padding: '2px 0'
            }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: `rgb(${l.color_r},${l.color_g},${l.color_b})`,
                flexShrink: 0,
                boxShadow: `0 0 4px rgba(${l.color_r},${l.color_g},${l.color_b},0.5)`
              }} />
              <span style={{ 
                whiteSpace: 'nowrap', 
                overflow: 'hidden', 
                textOverflow: 'ellipsis', 
                maxWidth: '140px',
                fontSize: '11px'
              }}>
                {l.label_name.replace(/_/g, ' ')}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Selection info */}
      {selectedIds.size > 0 && (
        <div style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          background: 'rgba(42, 42, 42, 0.98)',
          padding: '10px 14px',
          borderRadius: '8px',
          color: 'white',
          fontSize: '13px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
        }}>
          <span>
            <strong style={{ color: '#4a9eff' }}>{selectedIds.size}</strong> sample{selectedIds.size !== 1 ? 's' : ''} selected
          </span>
          <button
            onClick={clearSelection}
            style={{
              background: 'rgba(255, 68, 68, 0.9)',
              border: 'none',
              color: 'white',
              padding: '6px 12px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 500,
              transition: 'all 0.15s ease'
            }}
            className="clear-button"
          >
            Clear Selection
          </button>
        </div>
      )}
      
      {/* Inline styles for hover effects */}
      <style>{`
        .toolbar-button:hover {
          background: rgba(255, 255, 255, 0.15) !important;
        }
        .clear-button:hover {
          background: #ff4444 !important;
        }
      `}</style>
    </div>
  );
};

export default EmbeddingView;
