import React from 'react';
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, XCircle } from 'lucide-react';

interface HeaderProps {
  backendConnected: boolean | null;
  checkingConnection: boolean;
  onRefreshConnection: () => void;
  selectedCaseId: string | null;
}

export const Header: React.FC<HeaderProps> = ({
  backendConnected,
  checkingConnection,
  onRefreshConnection,
  selectedCaseId,
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
            Uncertainty-Aware 3D Clearance Analysis for Surgical Digital Twins
          </p>
        </div>
      </div>

      <div className="header-meta">
        {selectedCaseId && (
          <div className="case-badge">
            <span className="label">Active Case:</span>
            <span className="value font-mono">{selectedCaseId}</span>
            <span className="tag-synthetic">Synthetic Data</span>
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
          <strong>DEMONSTRATION ONLY:</strong> This system uses synthetic phantom data and is a research prototype.
          It does NOT predict clinical validity, surgical success, or patient safety.
        </span>
      </div>
    </header>
  );
};
