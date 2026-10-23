import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  Clock,
  Play,
  RotateCcw,
  ShieldAlert,
} from 'lucide-react';
import type { CaseDetail, UncertaintyResponse } from '../types/api';
import { apiClient } from '../services/apiClient';
import { HistogramChart } from './HistogramChart';

interface UncertaintyPanelProps {
  caseDetail: CaseDetail | null;
}

export const UncertaintyPanel: React.FC<UncertaintyPanelProps> = ({ caseDetail }) => {
  const [targetName, setTargetName] = useState<string>('tumor');
  const [riskName, setRiskName] = useState<string>('vessel');
  const [nSamples, setNSamples] = useState<number>(100);
  const [sigmaMm, setSigmaMm] = useState<number>(1.0);
  const [corrLengthMm, setCorrLengthMm] = useState<number>(5.0);
  const [biasSigmaMm, setBiasSigmaMm] = useState<number>(0.0);
  const [thresholdMm, setThresholdMm] = useState<number>(2.0);
  const [seed, setSeed] = useState<number>(42);

  const [loading, setLoading] = useState<boolean>(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UncertaintyResponse | null>(null);

  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const structures = caseDetail?.structures || [];

  // Elapsed timer during calculation
  useEffect(() => {
    if (loading) {
      startTimeRef.current = Date.now();
      timerRef.current = window.setInterval(() => {
        setElapsedSeconds((Date.now() - startTimeRef.current) / 1000);
      }, 100);
    } else {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, [loading]);

  const handleRunAnalysis = async () => {
    if (!caseDetail) return;
    setElapsedSeconds(0);
    setLoading(true);
    setError(null);

    // Validate n_samples capped at 300
    const boundedSamples = Math.min(300, Math.max(1, nSamples));

    try {
      const res = await apiClient.computeUncertainty(caseDetail.id, {
        target_name: targetName,
        risk_name: riskName,
        sigma_mm: sigmaMm,
        corr_length_mm: corrLengthMm,
        bias_sigma_mm: biasSigmaMm,
        n_samples: boundedSamples,
        seed: seed,
        threshold_mm: thresholdMm,
      });
      setResult(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Uncertainty analysis failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleResetDefaults = () => {
    setNSamples(100);
    setSigmaMm(1.0);
    setCorrLengthMm(5.0);
    setBiasSigmaMm(0.0);
    setThresholdMm(2.0);
    setSeed(42);
  };

  return (
    <div className="card uncertainty-card">
      <div className="card-header">
        <div className="card-title">
          <BarChart3 size={18} />
          <h2>Clearance Uncertainty Analysis</h2>
        </div>
        <button
          className="btn btn-toolbar btn-sm"
          onClick={handleResetDefaults}
          title="Reset parameters to defaults"
        >
          <RotateCcw size={12} />
          <span>Defaults</span>
        </button>
      </div>

      <div className="card-body">
        {/* Structure Selection Row */}
        <div className="clearance-selectors-grid">
          <div className="form-group">
            <label htmlFor="uncertainty-target">Target Structure</label>
            <select
              id="uncertainty-target"
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
            <label htmlFor="uncertainty-risk">Risk Structure</label>
            <select
              id="uncertainty-risk"
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

        {/* Parameters Form */}
        <div className="params-grid">
          {/* Realizations / Samples */}
          <div className="form-group">
            <div className="label-with-hint">
              <label htmlFor="param-nsamples">Realizations (N)</label>
              <span className="param-value-pill">{nSamples}</span>
            </div>
            <input
              id="param-nsamples"
              type="range"
              min="10"
              max="300"
              step="10"
              value={nSamples}
              onChange={(e) => setNSamples(parseInt(e.target.value, 10))}
              disabled={loading}
              className="range-input"
            />
            <span className="field-hint">Cap: 300 samples (backend enforced)</span>
          </div>

          {/* Amplitude Sigma */}
          <div className="form-group">
            <div className="label-with-hint">
              <label htmlFor="param-sigma">Perturbation Amplitude (σ)</label>
              <span className="param-value-pill">{sigmaMm.toFixed(1)} mm</span>
            </div>
            <input
              id="param-sigma"
              type="range"
              min="0.2"
              max="3.0"
              step="0.1"
              value={sigmaMm}
              onChange={(e) => setSigmaMm(parseFloat(e.target.value))}
              disabled={loading}
              className="range-input"
            />
            <span className="field-hint">Boundary displacement standard deviation</span>
          </div>

          {/* Correlation Length L */}
          <div className="form-group">
            <div className="label-with-hint">
              <label htmlFor="param-corr">Correlation Length (L)</label>
              <span className="param-value-pill">{corrLengthMm.toFixed(1)} mm</span>
            </div>
            <input
              id="param-corr"
              type="range"
              min="1.0"
              max="15.0"
              step="0.5"
              value={corrLengthMm}
              onChange={(e) => setCorrLengthMm(parseFloat(e.target.value))}
              disabled={loading}
              className="range-input"
            />
            <span className="field-hint">Spatial smoothness of boundary perturbations</span>
          </div>

          {/* Threshold Distance */}
          <div className="form-group">
            <div className="label-with-hint">
              <label htmlFor="param-threshold">Critical Distance Threshold</label>
              <span className="param-value-pill">{thresholdMm.toFixed(1)} mm</span>
            </div>
            <input
              id="param-threshold"
              type="range"
              min="0.0"
              max="10.0"
              step="0.5"
              value={thresholdMm}
              onChange={(e) => setThresholdMm(parseFloat(e.target.value))}
              disabled={loading}
              className="range-input"
            />
            <span className="field-hint">Threshold for probability estimation</span>
          </div>
        </div>

        {/* Action Button & Runtime Note */}
        <div className="action-row">
          <button
            className="btn btn-primary btn-block"
            onClick={handleRunAnalysis}
            disabled={loading || !caseDetail || targetName === riskName}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>
                  Simulating Realizations... ({elapsedSeconds.toFixed(1)}s elapsed)
                </span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Run Monte Carlo Uncertainty Analysis (N={nSamples})</span>
              </>
            )}
          </button>
        </div>

        {loading && (
          <div className="alert alert-info">
            <Clock className="icon-pulse" size={16} />
            <span>
              Monte Carlo perturbation generates 3D Gaussian noise fields on the Python backend.
              Execution typically takes ~5s for N=100 and ~14s for N=300. Please wait...
            </span>
          </div>
        )}

        {error && (
          <div className="alert alert-error">
            <ShieldAlert size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Results View */}
        {result && (
          <div className="uncertainty-results">
            {/* Summary KPI Grid */}
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Baseline Clearance</span>
                <span className="kpi-val font-mono">{result.baseline_mm.toFixed(3)} mm</span>
                <span className="kpi-sub">Unperturbed mask</span>
              </div>

              <div className="kpi-card">
                <span className="kpi-label">Perturbed Mean ± Std</span>
                <span className="kpi-val font-mono">
                  {result.mean.toFixed(3)} ± {result.std.toFixed(3)} mm
                </span>
                <span className="kpi-sub">Across {result.n_samples} realizations</span>
              </div>

              <div className="kpi-card kpi-card-highlight">
                <span className="kpi-label">P(Clearance &lt; {result.threshold_mm} mm)</span>
                <span className="kpi-val font-mono">
                  {(result.p_below_threshold * 100).toFixed(1)}%
                </span>
                <span className="kpi-sub font-mono">
                  95% Wilson CI: [{(result.p_below_ci95[0] * 100).toFixed(1)}%,{' '}
                  {(result.p_below_ci95[1] * 100).toFixed(1)}%]
                </span>
              </div>

              <div className="kpi-card">
                <span className="kpi-label">Overlap & Touching Prob.</span>
                <div className="kpi-row-compact">
                  <span className="kpi-val-compact font-mono">
                    P(overlap): {(result.p_overlap * 100).toFixed(1)}%
                  </span>
                  <span className="kpi-val-compact font-mono">
                    P(touching): {(result.p_touching * 100).toFixed(1)}%
                  </span>
                </div>
                <span className="kpi-sub">Realized penetrations</span>
              </div>
            </div>

            {/* Percentile Spread Table */}
            <div className="percentiles-bar">
              <div className="percentile-item">
                <span className="perc-label">5th Percentile (p05)</span>
                <span className="perc-val font-mono">{result.p05.toFixed(3)} mm</span>
              </div>
              <div className="percentile-item">
                <span className="perc-label">Median (p50)</span>
                <span className="perc-val font-mono">{result.p50.toFixed(3)} mm</span>
              </div>
              <div className="percentile-item">
                <span className="perc-label">95th Percentile (p95)</span>
                <span className="perc-val font-mono">{result.p95.toFixed(3)} mm</span>
              </div>
              <div className="percentile-item">
                <span className="perc-label">Random Seed</span>
                <span className="perc-val font-mono">{result.seed}</span>
              </div>
            </div>

            {/* Distribution Chart */}
            <HistogramChart
              samples={result.samples_mm}
              baselineMm={result.baseline_mm}
              thresholdMm={result.threshold_mm}
              p05={result.p05}
              p50={result.p50}
              p95={result.p95}
            />

            {/* Scientific Disclaimer */}
            <div className="alert alert-notice">
              <AlertCircle size={14} />
              <span>
                <strong>Model Assumption Disclaimer:</strong> Probabilities reflect the mathematical boundary
                perturbation model under the specified amplitude (σ={result.params.sigma_mm} mm) and correlation
                length (L={result.params.corr_length_mm} mm). They do NOT represent clinical outcomes or surgical success.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
