"use client";

import { useEffect, useCallback, useState, useRef } from "react";
import { Header, ImageGrid, ScatterPanel } from "@/components";
import { useStore } from "@/store/useStore";
import { fetchDataset, fetchSamples, fetchEmbeddings, fetchSamplesBatch } from "@/lib/api";

const SAMPLES_PER_PAGE = 100;

export default function Home() {
  const {
    samples,
    totalSamples,
    setSamples,
    appendSamples,
    addSamplesIfMissing,
    setDatasetInfo,
    setEmbeddings,
    setIsLoading,
    isLoading,
    error,
    setError,
    selectedIds,
  } = useStore();

  const [loadingMore, setLoadingMore] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // percentage
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Fetch dataset info, samples, and embeddings in parallel
        const [datasetInfo, samplesRes, embeddingsData] = await Promise.all([
          fetchDataset(),
          fetchSamples(0, SAMPLES_PER_PAGE),
          fetchEmbeddings(),
        ]);

        setDatasetInfo(datasetInfo);
        setSamples(samplesRes.samples, samplesRes.total);
        setEmbeddings(embeddingsData);
      } catch (err) {
        console.error("Failed to load data:", err);
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  // Fetch selected samples that aren't already loaded
  useEffect(() => {
    const fetchSelectedSamples = async () => {
      if (selectedIds.size === 0) return;

      // Find IDs that are selected but not in our samples array
      const loadedIds = new Set(samples.map((s) => s.id));
      const missingIds = Array.from(selectedIds).filter((id) => !loadedIds.has(id));

      if (missingIds.length === 0) return;

      try {
        const fetchedSamples = await fetchSamplesBatch(missingIds);
        addSamplesIfMissing(fetchedSamples);
      } catch (err) {
        console.error("Failed to fetch selected samples:", err);
      }
    };

    fetchSelectedSamples();
  }, [selectedIds, samples, addSamplesIfMissing]);

  // Load more samples
  const loadMore = useCallback(async () => {
    if (loadingMore || samples.length >= totalSamples) return;

    setLoadingMore(true);
    try {
      const res = await fetchSamples(samples.length, SAMPLES_PER_PAGE);
      appendSamples(res.samples);
    } catch (err) {
      console.error("Failed to load more samples:", err);
    } finally {
      setLoadingMore(false);
    }
  }, [samples.length, totalSamples, loadingMore, appendSamples]);

  // Handle resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current || !containerRef.current) return;

      const container = containerRef.current;
      const containerRect = container.getBoundingClientRect();
      const newWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

      // Constrain between 20% and 80%
      const clampedWidth = Math.min(Math.max(newWidth, 20), 80);
      setLeftPanelWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  const handleMouseDown = () => {
    isDraggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  if (error) {
    return (
      <div className="h-screen flex flex-col bg-background">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-red-500 text-lg mb-2">Error</div>
            <div className="text-text-muted">{error}</div>
            <p className="text-text-muted mt-4 text-sm">
              Make sure the HyperView backend is running on port 5151.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-screen flex flex-col bg-background">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <div className="text-text-muted">Loading dataset...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      <Header />

      {/* Main content - two panels */}
      <div ref={containerRef} className="flex-1 flex overflow-hidden">
        {/* Left panel - Image Grid */}
        <div
          className="min-w-0"
          style={{ width: `calc(${leftPanelWidth}% - 2px)` }}
        >
          <ImageGrid
            samples={samples}
            onLoadMore={loadMore}
            hasMore={samples.length < totalSamples}
          />
        </div>

        {/* Resizable divider */}
        <div
          className="w-1 bg-border hover:bg-primary cursor-col-resize flex-shrink-0 transition-colors"
          onMouseDown={handleMouseDown}
        />

        {/* Right panel - Scatter Plot */}
        <div
          className="min-w-0"
          style={{ width: `calc(${100 - leftPanelWidth}% - 2px)` }}
        >
          <ScatterPanel />
        </div>
      </div>
    </div>
  );
}
