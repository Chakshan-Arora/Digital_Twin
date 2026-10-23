import React from 'react';
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, User, XCircle } from 'lucide-react';

export type AppWorkstationMode = 'clearance' | 'explorer';

interface HeaderProps {
  backendConnected: boolean | null;
  checkingConnection: boolean;
  onRefreshConnection: () => void;
  selectedCaseId: string | null;
  activeMode: AppWorkstationMode;
  onModeChange: (mode: AppWorkstationMode) => void;
}

export const Header: React.FC<HeaderProps> = ({
  backendConnected,
  checkingConnection,
  onRefreshConnection,
  selectedCaseId,
  activeMode,
  onModeChange,
}) => {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="header-logo">
          <Activity className="icon-pulse" size={24} />
        </div>
        <div>
          <div className="header-title-row">
            <h1>Human Digital Twin (HDT)</h1>
            <span className="badge badge-demo">Research Prototype</span>
          </div>
          <p className="header-subtitle">
            Uncertainty-Aware 3D Clearance & Anatomical Digital Twin Platform
          </p>
        </div>
      </div>

      {/* Central Mode Switcher */}
      <div className="header-center-nav">
        <div className="mode-toggle-group" role="group" aria-label="Workstation mode selector">
          <button
            className={`btn-mode-toggle ${activeMode === 'clearance' ? 'active' : ''}`}
            onClick={() => onModeChange('clearance')}
            title="Clearance Lab: 3D Tumor/Vessel Clearance and Monte Carlo Uncertainty"
          >
            <Activity size={15} />
            <span>Clearance Lab</span>
          </button>
          <button
            className={`btn-mode-toggle ${activeMode === 'explorer' ? 'active' : ''}`}
            onClick={() => onModeChange('explorer')}
            title="Body Explorer: 3D Digital Twin Avatar, Dermatomes & Neural Pathways"
          >
            <User size={15} />
            <span>Body Explorer</span>
          </button>
        </div>
      </div>

      <div className="header-meta">
        {activeMode === 'clearance' && selectedCaseId && (
          <div className="case-badge">
            <span className="label">Active Case:</span>
            <span className="value font-mono">{selectedCaseId}</span>
            <span className="tag-synthetic">
              {selectedCaseId.startsWith('VS-SEG')
                ? 'TCIA VS-SEG'
                : selectedCaseId.startsWith('BraTS')
                ? 'BraTS 2024'
                : 'Synthetic Data'}
            </span>
          </div>
        )}

        {activeMode === 'explorer' && (
          <div className="case-badge">
            <span className="label">Avatar Model:</span>
            <span className="value font-mono">20-Joint Skeletal Mesh</span>
            <span className="tag-synthetic">Contract Mode</span>
          </div>
        )}

        <div className="connection-status">
          {checkingConnection ? (
            <div className="status-pill status-checking">
              <RefreshCw className="icon-spin" size={14} />
              <span>Connecting...</span>
            </div>
          ) : backendConnected === true ? (
            <div className="status-pill status-connected">
              <CheckCircle2 size={14} />
              <span>Backend Connected</span>
            </div>
          ) : (
            <div className="status-pill status-disconnected">
              <XCircle size={14} />
              <span>Backend Offline</span>
            </div>
          )}

          <button
            className="btn-icon"
            onClick={onRefreshConnection}
            title="Refresh backend status"
            aria-label="Refresh backend status"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="disclaimer-banner">
        <AlertTriangle size={14} />
        <span>
          <strong>RESEARCH PROTOTYPE ONLY:</strong> This system is a research prototype for uncertainty-aware 3D clearance analysis.
          It does NOT predict clinical validity, surgical success, or patient safety.
        </span>
      </div>
    </header>
  );
};

