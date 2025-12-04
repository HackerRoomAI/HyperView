import { create } from 'zustand';

const useStore = create((set) => ({
  points: [],
  labels: [],
  selectedIds: new Set(),
  hoveredId: null,
  viewMode: 'euclidean', // 'euclidean' | 'hyperbolic'

  setPoints: (points) => set({ points }),
  setLabels: (labels) => set({ labels }),

  fetchData: async () => {
    try {
      // Fetch points and labels in parallel
      const [pointsRes, labelsRes] = await Promise.all([
        fetch('/api/points'),
        fetch('/api/labels')
      ]);
      const points = await pointsRes.json();
      const labels = await labelsRes.json();
      set({ points, labels });
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  },

  toggleSelection: (id) => set((state) => {
    const newSelected = new Set(state.selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    return { selectedIds: newSelected };
  }),

  selectMultiple: (ids) => set({ selectedIds: new Set(ids) }),

  clearSelection: () => set({ selectedIds: new Set() }),

  setHoveredId: (id) => set({ hoveredId: id }),

  setViewMode: (mode) => set({ viewMode: mode }),
}));

export default useStore;
