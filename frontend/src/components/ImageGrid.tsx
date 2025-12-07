"use client";

import { useCallback, useEffect, useRef, useMemo, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStore } from "@/store/useStore";
import type { Sample } from "@/types";

interface ImageGridProps {
  samples: Sample[];
  onLoadMore?: () => void;
  hasMore?: boolean;
}

const GAP = 8;
const ITEM_HEIGHT = 200;
const MIN_ITEM_WIDTH = 200; // Minimum width for each image

export function ImageGrid({ samples, onLoadMore, hasMore }: ImageGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { selectedIds, isLassoSelection, toggleSelection, addToSelection, setHoveredId, hoveredId } = useStore();
  const [columnCount, setColumnCount] = useState(4);

  // Calculate column count based on container width
  useEffect(() => {
    const updateColumnCount = () => {
      if (!containerRef.current) return;
      const containerWidth = containerRef.current.clientWidth;
      const padding = 16; // Total horizontal padding (8px each side)
      const availableWidth = containerWidth - padding;

      // Calculate how many columns can fit
      const columns = Math.max(1, Math.floor((availableWidth + GAP) / (MIN_ITEM_WIDTH + GAP)));
      setColumnCount(columns);
    };

    updateColumnCount();

    const resizeObserver = new ResizeObserver(updateColumnCount);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  // Filter samples based on selection
  const filteredSamples = useMemo(() => {
    // Only filter (hide non-selected) when it's a lasso selection
    if (isLassoSelection && selectedIds.size > 0) {
      return samples.filter((sample) => selectedIds.has(sample.id));
    }

    // Otherwise, show all samples
    return samples;
  }, [samples, selectedIds, isLassoSelection]);

  // Calculate rows from filtered samples
  const rowCount = Math.ceil(filteredSamples.length / columnCount);

  // Create stable row keys based on the sample IDs in each row
  const getRowKey = useCallback(
    (index: number) => {
      const startIndex = index * columnCount;
      const rowSamples = filteredSamples.slice(startIndex, startIndex + columnCount);
      return rowSamples.map((s) => s.id).join("-") || `row-${index}`;
    },
    [filteredSamples, columnCount]
  );

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => containerRef.current,
    estimateSize: () => ITEM_HEIGHT + GAP,
    overscan: 5,
    getItemKey: getRowKey,
  });

  // Load more when scrolling near the bottom
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !onLoadMore || !hasMore) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      if (scrollHeight - scrollTop - clientHeight < 500) {
        onLoadMore();
      }
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [onLoadMore, hasMore]);

  // Reset virtualizer measurements when selection or filter mode changes
  useEffect(() => {
    virtualizer.measure();
  }, [selectedIds, isLassoSelection, virtualizer]);

  const handleClick = useCallback(
    (sample: Sample, event: React.MouseEvent) => {
      if (event.metaKey || event.ctrlKey) {
        // Multi-select with Cmd/Ctrl
        toggleSelection(sample.id);
      } else if (event.shiftKey && selectedIds.size > 0) {
        // Range select with Shift - use original samples array, not filtered
        const selectedArray = Array.from(selectedIds);
        const lastSelected = selectedArray[selectedArray.length - 1];
        const lastIndex = samples.findIndex((s) => s.id === lastSelected);
        const currentIndex = samples.findIndex((s) => s.id === sample.id);

        if (lastIndex !== -1 && currentIndex !== -1) {
          const start = Math.min(lastIndex, currentIndex);
          const end = Math.max(lastIndex, currentIndex);
          const rangeIds = samples.slice(start, end + 1).map((s) => s.id);
          addToSelection(rangeIds);
        }
      } else {
        // Single select
        const newSet = new Set<string>();
        newSet.add(sample.id);
        useStore.getState().setSelectedIds(newSet);
      }
    },
    [samples, selectedIds, toggleSelection, addToSelection]
  );

  const items = virtualizer.getVirtualItems();

  return (
    <div className="flex flex-col h-full bg-surface overflow-hidden">
      {/* Header */}
      <div className="h-12 flex items-center justify-between px-4 border-b border-border bg-surface-light">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Samples</span>
          <span className="text-xs text-text-muted">
            {selectedIds.size > 0 ? `${selectedIds.size} selected` : `${filteredSamples.length} items`}
          </span>
        </div>
        {selectedIds.size > 0 && (
          <button
            onClick={() => useStore.getState().setSelectedIds(new Set())}
            className="px-2 py-1 text-xs text-text-muted hover:text-text hover:bg-surface rounded transition-colors"
          >
            Clear Selection
          </button>
        )}
      </div>

      {/* Grid Container */}
      <div ref={containerRef} className="flex-1 overflow-auto p-2">
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: "100%",
            position: "relative",
          }}
        >
          {items.map((virtualRow) => {
            const rowIndex = virtualRow.index;
            const startIndex = rowIndex * columnCount;
            const rowSamples = filteredSamples.slice(startIndex, startIndex + columnCount);

            return (
              <div
                key={virtualRow.key}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: `${ITEM_HEIGHT}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
                className="flex gap-2 px-1"
              >
                {rowSamples.map((sample) => {
                  const isSelected = selectedIds.has(sample.id);
                  const isHovered = hoveredId === sample.id;

                  return (
                    <div
                      key={sample.id}
                      className={`
                        relative flex-1 rounded-md overflow-hidden cursor-pointer
                        transition-all duration-150 ease-out
                        ${isSelected ? "ring-2 ring-primary" : ""}
                        ${isHovered ? "ring-2 ring-primary-light" : ""}
                      `}
                      onClick={(e) => handleClick(sample, e)}
                      onMouseEnter={() => setHoveredId(sample.id)}
                      onMouseLeave={() => setHoveredId(null)}
                    >
                      {/* Image */}
                      {sample.thumbnail ? (
                        <img
                          src={`data:image/jpeg;base64,${sample.thumbnail}`}
                          alt={sample.filename}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      ) : (
                        <div className="w-full h-full bg-surface-light flex items-center justify-center">
                          <span className="text-text-muted text-xs">No image</span>
                        </div>
                      )}

                      {/* Label badge */}
                      {sample.label && (
                        <div className="absolute bottom-1 left-1 right-1">
                          <span
                            className="inline-block px-1.5 py-0.5 text-xs rounded truncate max-w-full"
                            style={{
                              backgroundColor: "rgba(0,0,0,0.7)",
                              color: "#fff",
                            }}
                          >
                            {sample.label}
                          </span>
                        </div>
                      )}

                      {/* Selection indicator */}
                      {isSelected && (
                        <div className="absolute top-1 right-1 w-5 h-5 rounded-full bg-primary flex items-center justify-center">
                          <svg
                            className="w-3 h-3 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={3}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        </div>
                      )}
                    </div>
                  );
                })}
                {/* Fill empty cells */}
                {Array.from({ length: columnCount - rowSamples.length }).map((_, i) => (
                  <div key={`empty-${i}`} className="flex-1" />
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Instructions footer */}
      <div className="px-3 py-2 text-xs text-text-muted border-t border-border bg-surface-light">
        <span className="opacity-70">
          Click to select • Cmd/Ctrl+click to multi-select • Shift+click for range
        </span>
      </div>
    </div>
  );
}
