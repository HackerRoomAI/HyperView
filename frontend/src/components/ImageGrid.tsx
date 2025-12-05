"use client";

import { useCallback, useEffect, useRef, useMemo } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStore } from "@/store/useStore";
import type { Sample } from "@/types";

interface ImageGridProps {
  samples: Sample[];
  onLoadMore?: () => void;
  hasMore?: boolean;
}

const COLUMN_COUNT = 4;
const GAP = 8;
const ITEM_HEIGHT = 140;

export function ImageGrid({ samples, onLoadMore, hasMore }: ImageGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { selectedIds, toggleSelection, addToSelection, setHoveredId, hoveredId } = useStore();

  // Sort samples so selected ones appear at the top
  const sortedSamples = useMemo(() => {
    if (selectedIds.size === 0) return samples;

    const selected: Sample[] = [];
    const unselected: Sample[] = [];

    samples.forEach((sample) => {
      if (selectedIds.has(sample.id)) {
        selected.push(sample);
      } else {
        unselected.push(sample);
      }
    });

    return [...selected, ...unselected];
  }, [samples, selectedIds]);

  // Calculate rows from sorted samples
  const rowCount = Math.ceil(sortedSamples.length / COLUMN_COUNT);

  // Create stable row keys based on the sample IDs in each row
  const getRowKey = useCallback(
    (index: number) => {
      const startIndex = index * COLUMN_COUNT;
      const rowSamples = sortedSamples.slice(startIndex, startIndex + COLUMN_COUNT);
      return rowSamples.map((s) => s.id).join("-") || `row-${index}`;
    },
    [sortedSamples]
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

  // Track previous selection to detect changes
  const prevSelectedIdsRef = useRef<Set<string>>(new Set());

  // Scroll to top and reset virtualizer when selection changes
  useEffect(() => {
    const prevIds = prevSelectedIdsRef.current;
    const currentIds = selectedIds;

    // Check if selection actually changed (not just same set)
    const selectionChanged =
      prevIds.size !== currentIds.size ||
      [...currentIds].some((id) => !prevIds.has(id));

    if (selectionChanged && currentIds.size > 0) {
      // Reset the virtualizer measurements to force re-render with new data
      virtualizer.measure();
      // Scroll to top using virtualizer's method
      virtualizer.scrollToOffset(0, { behavior: "smooth" });
    }

    // Update the ref with a new Set copy
    prevSelectedIdsRef.current = new Set(currentIds);
  }, [selectedIds, virtualizer]);

  const handleClick = useCallback(
    (sample: Sample, event: React.MouseEvent) => {
      if (event.metaKey || event.ctrlKey) {
        // Multi-select with Cmd/Ctrl
        toggleSelection(sample.id);
      } else if (event.shiftKey && selectedIds.size > 0) {
        // Range select with Shift
        const selectedArray = Array.from(selectedIds);
        const lastSelected = selectedArray[selectedArray.length - 1];
        const lastIndex = sortedSamples.findIndex((s) => s.id === lastSelected);
        const currentIndex = sortedSamples.findIndex((s) => s.id === sample.id);

        if (lastIndex !== -1 && currentIndex !== -1) {
          const start = Math.min(lastIndex, currentIndex);
          const end = Math.max(lastIndex, currentIndex);
          const rangeIds = sortedSamples.slice(start, end + 1).map((s) => s.id);
          addToSelection(rangeIds);
        }
      } else {
        // Single select
        const newSet = new Set<string>();
        newSet.add(sample.id);
        useStore.getState().setSelectedIds(newSet);
      }
    },
    [sortedSamples, selectedIds, toggleSelection, addToSelection]
  );

  const items = virtualizer.getVirtualItems();

  return (
    <div className="flex flex-col h-full bg-surface rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-surface-light">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Samples</span>
          <span className="text-xs text-text-muted">
            {sortedSamples.length} items
            {selectedIds.size > 0 && ` (${selectedIds.size} selected)`}
          </span>
        </div>
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
            const startIndex = rowIndex * COLUMN_COUNT;
            const rowSamples = sortedSamples.slice(startIndex, startIndex + COLUMN_COUNT);

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
                {Array.from({ length: COLUMN_COUNT - rowSamples.length }).map((_, i) => (
                  <div key={`empty-${i}`} className="flex-1" />
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
