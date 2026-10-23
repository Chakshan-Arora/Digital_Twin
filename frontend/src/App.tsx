import React, { useEffect, useState } from 'react';
import { BarChart3, Box, Columns2, Layers, Ruler } from 'lucide-react';
import { CaseDetailsCard } from './components/CaseDetailsCard';
import { ClearancePanel } from './components/ClearancePanel';
import { Header } from './components/Header';
import { SliceViewer } from './components/SliceViewer';
import { UncertaintyPanel } from './components/UncertaintyPanel';
import { Viewer3D } from './components/Viewer3D';
import { apiClient } from './services/apiClient';
import type { CaseDetail, CaseSummary, ClearanceResponse, ClosestPointsRAS } from './types/api';

export type ViewLayoutMode = 'split' | 'slices' | '3d';

export const App: React.FC = () => {
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const [checkingConnection, setCheckingConnection] = useState<boolean>(true);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [closestPoints, setClosestPoints] = useState<ClosestPointsRAS | null>(null);

  // Synchronized crosshair point in physical RAS millimetres across 2D MPR and 3D mesh viewers
  const [syncPointRas, setSyncPointRas] = useState<[number, number, number] | null>(null);

  // View layout: split (both 2D and 3D), 2D slices only, or 3D meshes only
  const [viewLayout, setViewLayout] = useState<ViewLayoutMode>('split');

  const [loadingCases, setLoadingCases] = useState<boolean>(false);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'clearance' | 'uncertainty'>('clearance');

  // Verify backend health and list cases
  const checkConnectionAndLoad = async () => {
    setCheckingConnection(true);
    setError(null);
    try {
      await apiClient.checkHealth();
      setBackendConnected(true);

      setLoadingCases(true);
      const casesList = await apiClient.listCases();
      setCases(casesList);

      if (casesList.length > 0) {
        const firstId = casesList[0].id;
        setSelectedCaseId(firstId);
        loadCaseDetail(firstId);
      }
    } catch (err) {
      setBackendConnected(false);
      const msg = err instanceof Error ? err.message : 'Failed to connect to backend';
      setError(msg);
    } finally {
      setCheckingConnection(false);
      setLoadingCases(false);
    }
  };

  const loadCaseDetail = async (caseId: string) => {
    setLoadingDetail(true);
    setClosestPoints(null); // Reset closest points on case switch
    setSyncPointRas(null);
    try {
      const detail = await apiClient.getCaseDetail(caseId);
      setSelectedCase(detail);
    } catch (err) {
      const msg = err instanceof Error ? err.message : `Failed to load case ${caseId}`;
      setError(msg);
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleSelectCase = (caseId: string) => {
    setSelectedCaseId(caseId);
    loadCaseDetail(caseId);
  };

  const handleClearanceComputed = (result: ClearanceResponse) => {
    if (result.closest_points_mm) {
      setClosestPoints(result.closest_points_mm);
      // Automatically synchronize focal point to target closest point in RAS mm
      const ptTarget = result.closest_points_mm.point_target_ras_mm;
      if (ptTarget && ptTarget.length === 3) {
        setSyncPointRas([ptTarget[0], ptTarget[1], ptTarget[2]]);
      }
    }
  };

  const handlePointSelected = React.useCallback((ptRas: [number, number, number]) => {
    setSyncPointRas((prev) => {
      if (
        prev &&
        Math.abs(prev[0] - ptRas[0]) < 1e-3 &&
        Math.abs(prev[1] - ptRas[1]) < 1e-3 &&
        Math.abs(prev[2] - ptRas[2]) < 1e-3
      ) {
        return prev;
      }
      return ptRas;
    });
  }, []);

  useEffect(() => {
    checkConnectionAndLoad();
  }, []);

  // Dispatch window resize event when switching layout mode so Three.js and Niivue recalculate projection matrices
  useEffect(() => {
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 60);
    return () => clearTimeout(timer);
  }, [viewLayout]);

  return (
    <div className="app-layout">
      {/* Top Navigation & Status Bar */}
      <Header
        backendConnected={backendConnected}
        checkingConnection={checkingConnection}
        onRefreshConnection={checkConnectionAndLoad}
        selectedCaseId={selectedCaseId}
      />

      {/* Main Multi-Pane Workspace */}
      <main className={`workspace-grid layout-${viewLayout}`}>
        {/* Left Column: Case Discovery, Metadata & Structure Info */}
        <section className="col-sidebar" aria-label="Case and metadata">
          <CaseDetailsCard
            cases={cases}
            selectedCase={selectedCase}
            selectedCaseId={selectedCaseId}
            onSelectCase={handleSelectCase}
            loading={loadingCases || loadingDetail}
            error={error}
          />
        </section>

        {/* Center Column: Primary Visualization Area (2D Slices & 3D Anatomy) */}
        <section className="col-viewer" aria-label="Anatomical Viewers">
          {/* Top Viewer Layout Bar */}
          <div className="viewer-layout-bar">
            <span className="layout-bar-title">Workstation View:</span>
            <div className="btn-group-toggle" role="group" aria-label="Workstation view mode">
              <button
                className={`btn btn-layout-mode ${viewLayout === 'split' ? 'active' : ''}`}
                onClick={() => setViewLayout('split')}
                title="Synchronized Split View: 2D Orthogonal Slices & 3D Surface Mesh"
              >
                <Columns2 size={14} />
                <span>Split (2D + 3D)</span>
              </button>
              <button
                className={`btn btn-layout-mode ${viewLayout === 'slices' ? 'active' : ''}`}
                onClick={() => setViewLayout('slices')}
                title="Focus on Orthogonal 2D MRI Slice Views"
              >
                <Layers size={14} />
                <span>2D Slices</span>
              </button>
              <button
                className={`btn btn-layout-mode ${viewLayout === '3d' ? 'active' : ''}`}
                onClick={() => setViewLayout('3d')}
                title="Focus on 3D Surface Anatomy Mesh"
              >
                <Box size={14} />
                <span>3D Mesh</span>
              </button>
            </div>
          </div>

          {/* Viewer Layout Container */}
          <div className={`viewers-container layout-${viewLayout}`}>
            {(viewLayout === 'split' || viewLayout === 'slices') && (
              <div className="viewer-pane pane-slices">
                <SliceViewer
                  caseDetail={selectedCase}
                  selectedPointRas={syncPointRas}
                  onPointSelected={handlePointSelected}
                />
              </div>
            )}

            {(viewLayout === 'split' || viewLayout === '3d') && (
              <div className="viewer-pane pane-3d">
                <Viewer3D
                  caseDetail={selectedCase}
                  closestPoints={closestPoints}
                  synchronizedPointRas={syncPointRas}
                />
              </div>
            )}
          </div>
        </section>

        {/* Analysis Workstation: Underneath viewers in split mode; in dedicated side column in 2D/3D focus modes */}
        <section
          className={`col-analysis ${viewLayout === 'split' ? 'col-analysis-stacked' : ''}`}
          aria-label="Clearance and uncertainty analysis"
        >
          {viewLayout === 'split' ? (
            <div className="analysis-split-grid">
              <ClearancePanel
                caseDetail={selectedCase}
                onClearanceComputed={handleClearanceComputed}
              />
              <UncertaintyPanel caseDetail={selectedCase} />
            </div>
          ) : (
            <>
              <div className="tab-bar">
                <button
                  className={`tab-btn ${activeTab === 'clearance' ? 'tab-active' : ''}`}
                  onClick={() => setActiveTab('clearance')}
                >
                  <Ruler size={15} />
                  <span>Clearance</span>
                </button>
                <button
                  className={`tab-btn ${activeTab === 'uncertainty' ? 'tab-active' : ''}`}
                  onClick={() => setActiveTab('uncertainty')}
                >
                  <BarChart3 size={15} />
                  <span>Uncertainty</span>
                </button>
              </div>

              <div className="tab-content">
                {activeTab === 'clearance' ? (
                  <ClearancePanel
                    caseDetail={selectedCase}
                    onClearanceComputed={handleClearanceComputed}
                  />
                ) : (
                  <UncertaintyPanel caseDetail={selectedCase} />
                )}
              </div>
            </>
          )}
        </section>
      </main>

      {/* Persistent Footer */}
      <footer className="app-footer">
        <span className="footer-item">Human Digital Twin (HDT) Research Prototype</span>
        <span className="footer-item">Coordinates: RAS (Right, Anterior, Superior) mm</span>
        <span className="footer-item footer-disclaimer">
          Demonstration system with synthetic data. NOT for diagnostic or surgical planning use.
        </span>
      </footer>
    </div>
  );
};

export default App;
