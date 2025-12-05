"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { scaleLinear } from "d3-scale";
import { useStore } from "@/store/useStore";
import type { ViewMode } from "@/types";

// Color utility
function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (result) {
    return [
      parseInt(result[1], 16) / 255,
      parseInt(result[2], 16) / 255,
      parseInt(result[3], 16) / 255,
    ];
  }
  return [0.5, 0.5, 0.5];
}

// Default colors for points without labels
const DEFAULT_COLORS = [
  "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
  "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
  "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
];

interface ScatterPanelProps {
  className?: string;
}

export function ScatterPanel({ className = "" }: ScatterPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgGroupRef = useRef<SVGGElement>(null);
  const scatterplotRef = useRef<any>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  const {
    embeddings,
    viewMode,
    setViewMode,
    selectedIds,
    setSelectedIds,
    hoveredId,
    setHoveredId,
  } = useStore();

  // Sync SVG transform
  const syncSvg = useCallback((event: any) => {
    const { xScale, yScale } = event;

    if (svgGroupRef.current && xScale && yScale) {
      // Calculate transform based on the actual scales used by the scatterplot
      // The SVG is defined in [-1, 1] coordinate space (r=1 circle at 0,0)
      // We want to map [-1, 1] to screen coordinates.

      // xScale(0) is the screen x-coordinate of the origin
      // xScale(1) is the screen x-coordinate of x=1
      // So the scaling factor for x is xScale(1) - xScale(0)

      const scaleX = xScale(1) - xScale(0);
      const scaleY = yScale(1) - yScale(0);
      const translateX = xScale(0);
      const translateY = yScale(0);

      svgGroupRef.current.setAttribute(
        "transform",
        `matrix(${scaleX}, 0, 0, ${scaleY}, ${translateX}, ${translateY})`
      );
    }
  }, []);

  // Initialize scatterplot
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current || isInitialized) return;

    let mounted = true;

    const initScatterplot = async () => {
      try {
        const createScatterplot = (await import("regl-scatterplot")).default;

        if (!mounted || !canvasRef.current || !containerRef.current) return;

        const { width, height } = containerRef.current.getBoundingClientRect();

        // Initialize D3 scales for synchronization
        // Our data is normalized to [-1, 1]
        const xScale = scaleLinear().domain([-1, 1]);
        const yScale = scaleLinear().domain([-1, 1]);

        const scatterplot = createScatterplot({
          canvas: canvasRef.current,
          width,
          height,
          xScale,
          yScale,
          pointSize: 4,
          pointSizeSelected: 8,
          opacity: 0.8,
          opacityInactiveMax: 0.2,
          lassoColor: [0.31, 0.27, 0.90, 1], // Indigo primary #4F46E5
          lassoMinDelay: 10,
          lassoMinDist: 2,
          showReticle: true,
          reticleColor: [1, 1, 1, 0.5],
          colorBy: 'category',
          pointColor: DEFAULT_COLORS,
        });

        // Handle view changes to sync SVG
        scatterplot.subscribe("view", syncSvg);

        // Initial sync
        const currentXScale = scatterplot.get("xScale");
        const currentYScale = scatterplot.get("yScale");
        if (currentXScale && currentYScale) {
          syncSvg({ xScale: currentXScale, yScale: currentYScale });
        }

        // Handle lasso selection
        scatterplot.subscribe("select", ({ points }: { points: number[] }) => {
          if (points.length > 0) {
            const currentEmbeddings = useStore.getState().embeddings;
            if (currentEmbeddings) {
              const selectedSampleIds = new Set(
                points.map((idx) => currentEmbeddings.ids[idx])
              );
              setSelectedIds(selectedSampleIds);
            }
          }
        });

        // Handle deselection
        scatterplot.subscribe("deselect", () => {
          setSelectedIds(new Set<string>());
        });

        // Handle point hover
        scatterplot.subscribe(
          "pointOver",
          (pointIndex: number) => {
            const currentEmbeddings = useStore.getState().embeddings;
            if (currentEmbeddings && pointIndex >= 0) {
              setHoveredId(currentEmbeddings.ids[pointIndex]);
            }
          }
        );

        scatterplot.subscribe("pointOut", () => {
          setHoveredId(null);
        });

        scatterplotRef.current = scatterplot;
        setIsInitialized(true);
      } catch (error) {
        console.error("Failed to initialize scatterplot:", error);
      }
    };

    initScatterplot();

    return () => {
      if (scatterplotRef.current) {
        scatterplotRef.current.destroy();
        scatterplotRef.current = null;
        setIsInitialized(false);
      }
    };
  }, [syncSvg]);

  // Update data when embeddings or viewMode changes
  useEffect(() => {
    if (!scatterplotRef.current || !embeddings) return;

    const coords = viewMode === "euclidean" ? embeddings.euclidean : embeddings.hyperbolic;

    // If switching to hyperbolic, try to sync SVG immediately
    if (viewMode === "hyperbolic") {
      // Small timeout to ensure SVG is rendered
      setTimeout(() => {
        if (scatterplotRef.current) {
          const xScale = scatterplotRef.current.get("xScale");
          const yScale = scatterplotRef.current.get("yScale");
          if (xScale && yScale) {
            syncSvg({ xScale, yScale });
          }
        }
      }, 0);
    }

    // Build unique categories for color mapping
    // Handle nulls by converting to "undefined"
    const uniqueLabels = [...new Set(embeddings.labels.map((l) => l || "undefined"))];
    
    const labelToCategory: Record<string, number> = {};
    uniqueLabels.forEach((label, idx) => {
      labelToCategory[label] = idx;
    });

    // Build category array (integer indices for each point)
    const categories = embeddings.labels.map((label) => {
      const key = label || "undefined";
      return labelToCategory[key];
    });

    // Build color palette from label colors
    const colorPalette = uniqueLabels.map((label) => {
      if (label === "undefined") return "#008080"; // Dark teal for undefined
      return embeddings.label_colors[label] || "#808080";
    });

    // Set the color palette first
    if (colorPalette.length > 0) {
      scatterplotRef.current.set({ pointColor: colorPalette });
    }

    scatterplotRef.current.draw({
      x: coords.map((c) => c[0]),
      y: coords.map((c) => c[1]),
      category: categories,
    });
    
    // Reset view to fit new points
    scatterplotRef.current.reset();
    
    // Try to sync again after draw
    if (viewMode === "hyperbolic") {
        setTimeout(() => {
            if (scatterplotRef.current) {
                const xScale = scatterplotRef.current.get("xScale");
                const yScale = scatterplotRef.current.get("yScale");
                if (xScale && yScale) {
                    syncSvg({ xScale, yScale });
                }
            }
        }, 100);
    }
  }, [embeddings, viewMode, isInitialized, syncSvg]);

  // Sync selection from store to scatterplot
  useEffect(() => {
    if (!scatterplotRef.current || !embeddings) return;

    const selectedIndices = Array.from(selectedIds)
      .map((id) => embeddings.ids.indexOf(id))
      .filter((idx) => idx !== -1);

    scatterplotRef.current.select(selectedIndices, { preventEvent: true });
  }, [selectedIds, embeddings, isInitialized]);

  // Handle resize
  useEffect(() => {
    if (!containerRef.current || !scatterplotRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0 && scatterplotRef.current) {
          scatterplotRef.current.set({ width, height });
        }
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [isInitialized]);

  // Get unique labels for legend
  const uniqueLabels = embeddings
    ? [...new Set(embeddings.labels.map((l) => l || "undefined"))]
    : [];

  return (
    <div className={`flex flex-col h-full bg-surface rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-surface-light">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium">Embeddings</span>

          {/* View mode toggle */}
          <div className="flex rounded-md overflow-hidden border border-border">
            <button
              onClick={() => setViewMode("euclidean")}
              className={`px-3 py-1 text-xs transition-colors ${
                viewMode === "euclidean"
                  ? "bg-primary text-white"
                  : "bg-surface hover:bg-surface-light text-text-muted"
              }`}
            >
              Euclidean
            </button>
            <button
              onClick={() => setViewMode("hyperbolic")}
              className={`px-3 py-1 text-xs transition-colors ${
                viewMode === "hyperbolic"
                  ? "bg-primary text-white"
                  : "bg-surface hover:bg-surface-light text-text-muted"
              }`}
            >
              Hyperbolic
            </button>
          </div>
        </div>

        <span className="text-xs text-text-muted">
          {embeddings ? `${embeddings.ids.length} points` : "Loading..."}
        </span>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex">
        {/* Canvas container */}
        <div ref={containerRef} className="flex-1 relative">
          <canvas
            ref={canvasRef}
            className="absolute inset-0"
            style={{ zIndex: 1 }}
          />

          {/* Poincaré disk boundary for hyperbolic mode */}
          {viewMode === "hyperbolic" && (
            <svg
              className="absolute inset-0 pointer-events-none"
              width="100%"
              height="100%"
              style={{ zIndex: 10 }}
            >
              <g ref={svgGroupRef}>
                {/* Main Boundary Circle - scaled to match data (max r ≈ 0.9) */}
                <circle
                  cx="0"
                  cy="0"
                  r="0.95"
                  fill="none"
                  stroke="rgba(255,255,255,0.6)"
                  strokeWidth="0.008"
                />

                {/* Grid Circles - adjusted for 0.65 hyperbolic scaling factor */}
                {/* After scaling: d=1 => r≈0.316, d=2 => r≈0.569, d=3 => r≈0.748 */}
                <circle cx="0" cy="0" r="0.316" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
                <circle cx="0" cy="0" r="0.569" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
                <circle cx="0" cy="0" r="0.748" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />

                {/* Radial Lines - scaled to boundary */}
                <line x1="-0.95" y1="0" x2="0.95" y2="0" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
                <line x1="0" y1="-0.95" x2="0" y2="0.95" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
                {/* Diagonals */}
                <line x1="-0.672" y1="-0.672" x2="0.672" y2="0.672" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
                <line x1="-0.672" y1="0.672" x2="0.672" y2="-0.672" stroke="rgba(255,255,255,0.15)" strokeWidth="0.004" />
              </g>
            </svg>
          )}

          {/* Loading overlay */}
          {!embeddings && (
            <div className="absolute inset-0 flex items-center justify-center bg-surface/80 z-10">
              <div className="text-text-muted">Loading embeddings...</div>
            </div>
          )}
        </div>

        {/* Legend */}
        {uniqueLabels.length > 0 && (
          <div className="w-36 border-l border-border bg-surface-light p-2 overflow-y-auto">
            <div className="text-xs font-medium mb-2 text-text-muted">Labels</div>
            <div className="space-y-1">
              {uniqueLabels.slice(0, 20).map((label) => (
                <div key={label} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor: label === "undefined" ? "#008080" : (embeddings?.label_colors[label!] || "#888"),
                    }}
                  />
                  <span className="text-xs truncate" title={label!}>
                    {label}
                  </span>
                </div>
              ))}
              {uniqueLabels.length > 20 && (
                <div className="text-xs text-text-muted">
                  +{uniqueLabels.length - 20} more
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="px-3 py-2 text-xs text-text-muted border-t border-border bg-surface-light">
        <span className="opacity-70">
          Shift+drag to lasso select • Scroll to zoom • Drag to pan
        </span>
      </div>
    </div>
  );
}
