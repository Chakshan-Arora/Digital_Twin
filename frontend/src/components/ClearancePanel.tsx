import React, { useState } from 'react';
import { ArrowRight, ChevronRight, Ruler, ShieldAlert, Sparkles } from 'lucide-react';
import type { CaseDetail, ClearanceResponse } from '../types/api';
import { apiClient } from '../services/apiClient';

interface ClearancePanelProps {
  caseDetail: CaseDetail | null;
  onClearanceComputed: (result: ClearanceResponse) => void;
}

export const ClearancePanel: React.FC<ClearancePanelProps> = ({
  caseDetail,
  onClearanceComputed,
}) => {
  const [targetName, setTargetName] = useState<string>('tumor');
  const [riskName, setRiskName] = useState<string>('vessel');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClearanceResponse | null>(null);

  const structures = caseDetail?.structures || [];

  const handleCompute = async () => {
    if (!caseDetail) return;
    setLoading(true);
    setError(null);

    try {
      const res = await apiClient.computeClearance(caseDetail.id, {
        target_name: targetName,
        risk_name: riskName,
      });
      setResult(res);
      onClearanceComputed(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Clearance computation failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const getClearanceBadgeClass = (val: number) => {
    if (val > 0) return 'badge-clearance-gap';
    if (val === 0) return 'badge-clearance-touching';
    return 'badge-clearance-overlap';
  };

  return (
    <div className="card clearance-card">
      <div className="card-header">
        <div className="card-title">
          <Ruler size={18} />
          <h2>Pairwise Clearance Analysis</h2>
        </div>
      </div>

      <div className="card-body">
        {/* Structure Selection Row */}
        <div className="clearance-selectors-grid">
          <div className="form-group">
            <label htmlFor="clearance-target">Target Structure</label>
            <select
              id="clearance-target"
              className="select-input"
              value={targetName}
              onChange={(e) => setTargetName(e.target.value)}
              disabled={loading || structures.length === 0}
            >
              {structures.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} ({s.role})
                </option>
              ))}
            </select>
          </div>

          <div className="form-arrow">
            <ArrowRight size={18} />
          </div>

          <div className="form-group">
            <label htmlFor="clearance-risk">Risk Structure</label>
            <select
              id="clearance-risk"
              className="select-input"
              value={riskName}
              onChange={(e) => setRiskName(e.target.value)}
              disabled={loading || structures.length === 0}
            >
              {structures.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} ({s.role})
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          className="btn btn-primary btn-block"
          onClick={handleCompute}
          disabled={loading || !caseDetail || targetName === riskName}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              <span>Calculating Face-to-Face Clearance...</span>
            </>
          ) : (
            <>
              <Sparkles size={16} />
              <span>Calculate Clearance</span>
            </>
          )}
        </button>

        {targetName === riskName && (
          <p className="field-hint text-warning">
            Target and risk structures must be different.
          </p>
        )}

        {error && (
          <div className="alert alert-error">
            <ShieldAlert size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Clearance Results Display */}
        {result && (
          <div className="clearance-results">
            {/* Primary Clearance Metric */}
            <div className="metric-hero">
              <span className="metric-hero-label">Signed Face-to-Face Clearance:</span>
              <div className="metric-hero-value-row">
                <span className={`metric-hero-value font-mono ${getClearanceBadgeClass(result.signed_clearance_mm)}`}>
                  {result.signed_clearance_mm > 0 ? `+${result.signed_clearance_mm.toFixed(3)}` : result.signed_clearance_mm.toFixed(3)} mm
                </span>
                <span className="convention-tag">Convention: {result.convention}</span>
              </div>
              <p className="metric-hero-desc">
                Positive = gap between structures; 0 = touching; negative = penetration.
              </p>
            </div>

            {/* Closest Voxel Points & Distance */}
            <div className="metric-card">
              <div className="metric-card-header">
                <ChevronRight size={14} />
                <h4>Closest Voxel Pair (RAS Coordinates)</h4>
              </div>
              <div className="metric-card-body">
                <div className="closest-points-row">
                  <span className="label">Euclidean Voxel-Center Distance:</span>
                  <span className="val font-mono">{result.closest_points_mm.distance_mm.toFixed(3)} mm</span>
                </div>
                <div className="points-coords-grid">
                  <div className="point-box">
                    <span className="point-label font-mono">Target ({targetName})</span>
                    <span className="point-coord font-mono">
                      [{result.closest_points_mm.point_target_ras_mm.map((v) => v.toFixed(2)).join(', ')}]
                    </span>
                  </div>
                  <div className="point-box">
                    <span className="point-label font-mono">Risk ({riskName})</span>
                    <span className="point-coord font-mono">
                      [{result.closest_points_mm.point_risk_ras_mm.map((v) => v.toFixed(2)).join(', ')}]
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Surface Distance Metrics */}
            <div className="metric-card">
              <div className="metric-card-header">
                <ChevronRight size={14} />
                <h4>Surface Distance Statistics</h4>
              </div>
              <div className="stats-table-wrapper">
                <table className="stats-table">
                  <thead>
                    <tr>
                      <th>Min</th>
                      <th>Mean</th>
                      <th>HD95 (95% Hausdorff)</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="font-mono">{result.surface_distance_stats.min.toFixed(3)} mm</td>
                      <td className="font-mono">{result.surface_distance_stats.mean.toFixed(3)} mm</td>
                      <td className="font-mono font-bold text-accent">{result.surface_distance_stats.hd95.toFixed(3)} mm</td>
                      <td className="font-mono">{result.surface_distance_stats.max.toFixed(3)} mm</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Disclaimer */}
            <div className="alert alert-notice">
              <ShieldAlert size={14} />
              <span>
                <strong>Research Note:</strong> Clearance reflects Euclidean boundary distances under the face-to-face
                voxel sets convention. It is NOT a clinically validated surgical margin.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
