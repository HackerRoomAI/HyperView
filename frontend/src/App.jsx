import React, { useEffect } from 'react';
import ImageGrid from './components/ImageGrid';
import EmbeddingView from './components/EmbeddingView';
import useStore from './store';
import './App.css';

function App() {
  const { fetchData } = useStore();

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className="app-container">
      <div className="panel left-panel">
        <ImageGrid />
      </div>
      <div className="panel right-panel">
        <EmbeddingView />
      </div>
    </div>
  );
}

export default App;
