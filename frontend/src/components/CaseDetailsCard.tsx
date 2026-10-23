import React from 'react';
import { Database, Download, FileText, Layers, ShieldAlert, Target } from 'lucide-react';
import type { CaseDetail, CaseSummary } from '../types/api';
import { apiClient } from '../services/apiClient';

interface CaseDetailsCardProps {
  cases: CaseSummary[];
  selectedCase: CaseDetail | null;
  selectedCaseId: string | null;
  onSelectCase: (caseId: string) => void;
  loading: boolean;
  error: string | null;
}

export const CaseDetailsCard: React.FC<CaseDetailsCardProps> = ({
  cases,
  selectedCase,
  selectedCaseId,
  onSelectCase,
  loading,
  error,
}) => {
  return (
    <div className="card case-details-card">
      <div className="card-header">
        <div className="card-title">
          <Database size={18} />
          <h2>Case Discovery & Geometry</h2>
        </div>
      </div>

      <div className="card-body">
        {/* Case Selection */}
        <div className="form-group">
          <label htmlFor="case-select">Select Case</label>
          <select
            id="case-select"
            className="select-input"
            value={selectedCaseId || ''}
            onChange={(e) => onSelectCase(e.target.value)}
            disabled={loading || cases.length === 0}
          >
            {cases.length === 0 ? (
              <option value="">No cases available</option>
            ) : (
              cases.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ({c.structures.map((s) => s.name).join(', ')})
                </option>
              ))
            )}
          </select>
        </div>

        {error && (
          <div className="alert alert-error">
            <ShieldAlert size={16} />
            <span>{error}</span>
          </div>
        )}

        {loading && !selectedCase && (
          <div className="loading-state">
            <span className="spinner"></span>
            <span>Loading case geometry...</span>
          </div>
        )}

        {selectedCase && (
          <div className="case-meta-content">
            {/* Spatial Geometry Details */}
            <div className="meta-section">
              <h3 className="section-title">
                <Layers size={14} />
                <span>Spatial Metadata (RAS Space)</span>
              </h3>
              <div className="grid-meta">
                <div className="meta-item">
                  <span className="meta-label">Cohort / Source</span>
                  <span className="meta-value font-mono">
                    {selectedCase.id.startsWith('VS-SEG')
                      ? 'TCIA VS-SEG'
                      : selectedCase.id.startsWith('BraTS')
                      ? 'BraTS 2024'
                      : 'Synthetic Phantom'}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Coord. System</span>
                  <span className="meta-value font-mono badge-pill">{selectedCase.coordinate_system}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Primary Modality</span>
                  <span className="meta-value font-mono">{selectedCase.primary_image_name}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Dimensions (z, y, x)</span>
                  <span className="meta-value font-mono">
                    {selectedCase.shape.join(' × ')} voxels
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Voxel Spacing (RAS mm)</span>
                  <span className="meta-value font-mono">
                    [{selectedCase.spacing_mm.map((s) => s.toFixed(2)).join(', ')}]
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Origin (RAS mm)</span>
                  <span className="meta-value font-mono">
                    [{selectedCase.origin_ras_mm.map((o) => o.toFixed(1)).join(', ')}]
                  </span>
                </div>
              </div>
            </div>

            {/* Segmented Structures */}
            <div className="meta-section">
              <h3 className="section-title">
                <Target size={14} />
                <span>Segmented Structures ({selectedCase.structures.length})</span>
              </h3>
              <div className="structures-list">
                {selectedCase.structures.map((s) => {
                  const roleBadgeClass =
                    s.role === 'target'
                      ? 'role-badge-target'
                      : s.role === 'risk'
                      ? 'role-badge-risk'
                      : 'role-badge-context';
                  return (
                    <div key={s.name} className={`structure-item role-${s.role}`}>
                      <div className="structure-info">
                        <span className={`structure-bullet role-bullet-${s.role}`}></span>
                        <span className="structure-name font-mono">{s.name}</span>
                      </div>
                      <span className={`role-badge ${roleBadgeClass}`}>
                        {s.role}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* NIfTI Volume & Mask Downloads */}
            <div className="meta-section">
              <h3 className="section-title">
                <Download size={14} />
                <span>Volume & Mask Downloads</span>
              </h3>
              <div className="download-buttons">
                <a
                  href={apiClient.getVolumeDownloadUrl(selectedCase.id)}
                  download={`${selectedCase.id}_volume.nii.gz`}
                  className="btn btn-secondary btn-sm"
                  title="Download primary NIfTI volume"
                >
                  <FileText size={14} />
                  <span>Volume (.nii.gz)</span>
                </a>

                {selectedCase.structures.map((s) => (
                  <a
                    key={s.name}
                    href={apiClient.getMaskDownloadUrl(selectedCase.id, s.name)}
                    download={`${selectedCase.id}_${s.name}_mask.nii.gz`}
                    className="btn btn-secondary btn-sm"
                    title={`Download ${s.name} binary mask (.nii.gz)`}
                  >
                    <Download size={14} />
                    <span>{s.name} mask</span>
                  </a>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
