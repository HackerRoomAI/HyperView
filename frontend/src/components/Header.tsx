"use client";

import { useStore } from "@/store/useStore";

export function Header() {
  const { datasetInfo, selectedIds, clearSelection } = useStore();

  return (
    <header className="h-14 bg-surface border-b border-border flex items-center justify-between px-4">
      {/* Logo and title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <svg
            className="w-5 h-5 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <div>
          <h1 className="text-lg font-semibold text-text">HyperView</h1>
          {datasetInfo && (
            <p className="text-xs text-text-muted">{datasetInfo.name}</p>
          )}
        </div>
      </div>

      {/* Dataset info and actions */}
      <div className="flex items-center gap-4">
        {datasetInfo && (
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-text-muted">Samples:</span>
              <span className="text-text font-medium">{datasetInfo.num_samples.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-text-muted">Labels:</span>
              <span className="text-text font-medium">{datasetInfo.labels.length}</span>
            </div>
          </div>
        )}

        {selectedIds.size > 0 && (
          <button
            onClick={clearSelection}
            className="px-3 py-1.5 text-sm bg-surface-light hover:bg-border rounded-md transition-colors"
          >
            Clear selection ({selectedIds.size})
          </button>
        )}
      </div>
    </header>
  );
}
