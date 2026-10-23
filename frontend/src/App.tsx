import React, { useEffect, useState } from 'react';
import { Header, type AppWorkstationMode } from './components/Header';
import { ClearanceLabWorkstation } from './layout/ClearanceLabWorkstation';
import { BodyExplorerWorkstation } from './layout/BodyExplorerWorkstation';
import { apiClient } from './services/apiClient';

export const App: React.FC = () => {
  // Mode router: derived from URL hash or defaults to 'clearance'
  const [activeMode, setActiveMode] = useState<AppWorkstationMode>(() => {
    if (typeof window !== 'undefined' && window.location.hash.toLowerCase().includes('explorer')) {
      return 'explorer';
    }
    return 'clearance';
  });

  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const [checkingConnection, setCheckingConnection] = useState<boolean>(true);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);

  // Synchronize URL hash with active workstation mode
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.toLowerCase();
      if (hash.includes('explorer')) {
        setActiveMode('explorer');
      } else {
        setActiveMode('clearance');
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleModeChange = (newMode: AppWorkstationMode) => {
    setActiveMode(newMode);
    if (typeof window !== 'undefined') {
      window.location.hash = newMode === 'explorer' ? '#explorer' : '#clearance';
      // Trigger resize after layout switch so Three.js viewports recalculate their bounding sizes
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 60);
    }
  };

  const checkConnection = async () => {
    setCheckingConnection(true);
    try {
      await apiClient.checkHealth();
      setBackendConnected(true);
    } catch {
      setBackendConnected(false);
    } finally {
      setCheckingConnection(false);
    }
  };

  useEffect(() => {
    checkConnection();
  }, []);

  return (
    <div className="app-layout">
      {/* Persistent Shell Header with Workstation Mode Switcher */}
      <Header
        backendConnected={backendConnected}
        checkingConnection={checkingConnection}
        onRefreshConnection={checkConnection}
        selectedCaseId={activeCaseId}
        activeMode={activeMode}
        onModeChange={handleModeChange}
      />

      {/* Dynamic Workstation Shell */}
      {activeMode === 'clearance' ? (
        <ClearanceLabWorkstation
          onCaseSelected={setActiveCaseId}
          onConnectionStatusChanged={(connected, checking) => {
            setBackendConnected(connected);
            setCheckingConnection(checking);
          }}
        />
      ) : (
        <BodyExplorerWorkstation />
      )}

      {/* Persistent Footer with Clinical Disclaimer */}
      <footer className="app-footer">
        <span className="footer-item">Human Digital Twin (HDT) Research Prototype</span>
        <span className="footer-item">
          {activeMode === 'clearance'
            ? 'Clearance Lab — Physical RAS Space (mm)'
            : 'Body Explorer — 3D Anthropometric Joint Coordinates (mm)'}
        </span>
        <span className="footer-item footer-disclaimer">
          Demonstration system with synthetic data. NOT for diagnostic or surgical planning use.
        </span>
      </footer>
    </div>
  );
};

export default App;
