import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue';
import {
  AlertTriangle,
  Crosshair,
  Eye,
  EyeOff,
  Layers,
  Minus,
  Plus,
  RefreshCw,
  RotateCcw,
  Sliders,
} from 'lucide-react';
import type { CaseDetail } from '../types/api';
import { apiClient } from '../services/apiClient';
import {
  formatRasMm,
  formatVoxelZyx,
  rasMmToVoxelZyx,
  voxelZyxToRasMm,
} from '../utils/coordinates';

export interface SliceViewerProps {
  caseDetail: CaseDetail | null;
  selectedPointRas?: [number, number, number] | null;
  onPointSelected?: (pointRas: [number, number, number]) => void;
}

interface MaskLayerState {
  name: string;
  role: string;
  colorHex: string;
  colormap: string;
  visible: boolean;
  opacity: number;
  volIndex: number;
}

export const SliceViewer: React.FC<SliceViewerProps> = ({
  caseDetail,
  selectedPointRas,
  onPointSelected,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Keep ref to latest onPointSelected callback to avoid re-triggering init effect
  const onPointSelectedRef = useRef(onPointSelected);
  useEffect(() => {
    onPointSelectedRef.current = onPointSelected;
  }, [onPointSelected]);

  const caseDetailRef = useRef(caseDetail);
  useEffect(() => {
    caseDetailRef.current = caseDetail;
  }, [caseDetail]);

  // Guard flag to prevent re-entrant feedback loops between slider/canvas interaction and external sync
  const isInternalUpdateRef = useRef<boolean>(false);

  const [sliceMode, setSliceMode] = useState<number>(SLICE_TYPE.MULTIPLANAR);
  const [showCrosshair, setShowCrosshair] = useState<boolean>(true);

  // Current crosshair location in physical RAS mm and voxel (z, y, x)
  const [currentRasMm, setCurrentRasMm] = useState<[number, number, number] | null>(null);
  const [currentVoxelZyx, setCurrentVoxelZyx] = useState<[number, number, number] | null>(null);

  // Mask overlay states
  const [maskLayers, setMaskLayers] = useState<MaskLayerState[]>([]);

  // Active slice sliders indices (0 to shape-1)
  const [sliceIndices, setSliceIndices] = useState<{ z: number; y: number; x: number }>({
    z: 32,
    y: 32,
    x: 32,
  });

  const maxShapes = useMemo(
    () => caseDetail?.shape_zyx || caseDetail?.shape || [64, 64, 64],
    [caseDetail]
  );

  // Load volume and structure masks
  const loadVolumes = useCallback(async (nvInstance?: Niivue | null) => {
    const nv = nvInstance || nvRef.current;
    if (!caseDetail || !nv) return;

    setLoading(true);
    setError(null);
    setLoadingStage('Downloading NIfTI anatomical scan...');

    try {
      const volUrl = apiClient.getVolumeDownloadUrl(caseDetail.id);
      const volumeList: Array<{
        url: string;
        name: string;
        colormap: string;
        opacity: number;
        cal_min?: number;
        cal_max?: number;
      }> = [
        {
          url: volUrl,
          name: `${caseDetail.id}_volume.nii.gz`,
          colormap: 'gray',
          opacity: 1.0,
        },
      ];

      const initialMasks: MaskLayerState[] = [];
      let volIdx = 1;

      for (const s of caseDetail.structures) {
        const maskUrl = apiClient.getMaskDownloadUrl(caseDetail.id, s.name);
        const isTarget = s.name === 'tumor' || s.name === 'tumour' || s.name === 'gtv' || s.role === 'target';
        const colormap = isTarget ? 'red' : 'blue';
        const colorHex = isTarget ? '#ef4444' : '#3b82f6';

        volumeList.push({
          url: maskUrl,
          name: `${caseDetail.id}_${s.name}_mask.nii.gz`,
          colormap,
          opacity: 0.65,
          cal_min: 0.5,
          cal_max: 1.0,
        });

        initialMasks.push({
          name: s.name,
          role: s.role,
          colorHex,
          colormap,
          visible: true,
          opacity: 0.65,
          volIndex: volIdx,
        });

        volIdx++;
      }

      setLoadingStage('Decoding NIfTI volumes & parsing spatial affines...');
      await nv.loadVolumes(volumeList);

      setMaskLayers(initialMasks);

      // Default focal point: anatomical center of volume
      const centerZyx: [number, number, number] = [
        maxShapes[0] / 2,
        maxShapes[1] / 2,
        maxShapes[2] / 2,
      ];
      const centerRas = voxelZyxToRasMm(centerZyx, caseDetail);
      setCurrentRasMm(centerRas);
      setCurrentVoxelZyx(centerZyx);
      setSliceIndices({
        z: Math.round(centerZyx[0]),
        y: Math.round(centerZyx[1]),
        x: Math.round(centerZyx[2]),
      });

      // Jump Niivue crosshair to center
      if (nv.volumes && nv.volumes.length > 0) {
        const frac = nv.mm2frac(centerRas);
        nv.scene.crosshairPos = frac;
        nv.drawScene();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load NIfTI images';
      setError(msg);
    } finally {
      setLoading(false);
      setLoadingStage('');
    }
  }, [caseDetail, maxShapes]);

  // Initialize Niivue instance
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;

    // Check if WebGL2 is supported in current environment
    try {
      const gl = canvas.getContext('webgl2');
      if (!gl) {
        setError('WebGL 2 is not available in your browser or test environment.');
        return;
      }
    } catch {
      setError('Error acquiring WebGL 2 rendering context.');
      return;
    }

    let nv: Niivue;
    try {
      nv = new Niivue({
        show3Dcrosshair: true,
        backColor: [0.047, 0.059, 0.09, 1.0], // Dark slate #0c0f17
        crosshairColor: [0.984, 0.749, 0.141, 0.95], // Bright amber #fbbf24
        crosshairWidth: 1.5,
        isRadiologicalConvention: false, // Neurological / RAS convention (Right on right)
        isOrientationTextVisible: true,
        showAllOrientationMarkers: true,
        multiplanarShowRender: SHOW_RENDER.NEVER, // Show pure orthogonal 2D MPR
        multiplanarEqualSize: false, // Respect anisotropic voxel proportions
        multiplanarPadPixels: 4, // Clean separation between orthogonal cuts
        sliceType: SLICE_TYPE.MULTIPLANAR,
        loadingText: 'Loading Medical Volumes & Overlays...',
      });

      nv.attachToCanvas(canvas);
      nvRef.current = nv;
      (window as unknown as { nv?: Niivue }).nv = nv;
      loadVolumes(nv);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to initialize medical image viewer';
      setError(msg);
      return;
    }

    // Handle crosshair location updates from canvas interaction
    nv.onLocationChange = (data: unknown) => {
      if (isInternalUpdateRef.current) return;
      const loc = data as { mm?: number[]; vox?: number[] } | null | undefined;
      if (loc && loc.mm && loc.mm.length >= 3 && !isNaN(loc.mm[0])) {
        const rasPt: [number, number, number] = [loc.mm[0], loc.mm[1], loc.mm[2]];
        setCurrentRasMm(rasPt);

        const cd = caseDetailRef.current;
        if (cd) {
          const voxZyx = rasMmToVoxelZyx(rasPt, cd);
          setCurrentVoxelZyx(voxZyx);
          setSliceIndices({
            z: Math.max(0, Math.min(maxShapes[0] - 1, Math.round(voxZyx[0]))),
            y: Math.max(0, Math.min(maxShapes[1] - 1, Math.round(voxZyx[1]))),
            x: Math.max(0, Math.min(maxShapes[2] - 1, Math.round(voxZyx[2]))),
          });
        }

        onPointSelectedRef.current?.(rasPt);
      }
    };

    return () => {
      if (nvRef.current) {
        nvRef.current = null;
      }
    };
  }, [caseDetail?.id, loadVolumes, maxShapes]);

  // Synchronize external point if selected elsewhere (e.g., from 3D viewer or clearance panel)
  useEffect(() => {
    if (!selectedPointRas || !nvRef.current || !caseDetail) return;
    if (isInternalUpdateRef.current) return;
    const nv = nvRef.current;
    if (!nv.volumes || nv.volumes.length === 0) return;

    // Check if point actually changed to prevent infinite re-render loops
    if (
      currentRasMm &&
      Math.abs(currentRasMm[0] - selectedPointRas[0]) < 1e-3 &&
      Math.abs(currentRasMm[1] - selectedPointRas[1]) < 1e-3 &&
      Math.abs(currentRasMm[2] - selectedPointRas[2]) < 1e-3
    ) {
      return;
    }

    try {
      const frac = nv.mm2frac(selectedPointRas);
      nv.scene.crosshairPos = frac;
      nv.drawScene();

      setCurrentRasMm(selectedPointRas);
      const voxZyx = rasMmToVoxelZyx(selectedPointRas, caseDetail);
      setCurrentVoxelZyx(voxZyx);
      setSliceIndices({
        z: Math.max(0, Math.min(maxShapes[0] - 1, Math.round(voxZyx[0]))),
        y: Math.max(0, Math.min(maxShapes[1] - 1, Math.round(voxZyx[1]))),
        x: Math.max(0, Math.min(maxShapes[2] - 1, Math.round(voxZyx[2]))),
      });
    } catch {
      // Ignore conversion failures on extreme coordinates
    }
  }, [selectedPointRas, caseDetail, currentRasMm, maxShapes]);

  // View Mode switcher
  const handleSliceModeChange = (mode: number) => {
    setSliceMode(mode);
    if (!nvRef.current) return;
    nvRef.current.setSliceType(mode);
  };

  // Toggle crosshair visibility
  const handleToggleCrosshair = () => {
    const nextVal = !showCrosshair;
    setShowCrosshair(nextVal);
    if (!nvRef.current) return;
    nvRef.current.setCrosshairWidth(nextVal ? 1.5 : 0);
  };

  // Toggle mask overlay visibility
  const toggleMaskVisibility = (layerIdx: number) => {
    const updated = [...maskLayers];
    const layer = updated[layerIdx];
    layer.visible = !layer.visible;
    setMaskLayers(updated);

    if (!nvRef.current) return;
    const nv = nvRef.current;
    const effectiveOpacity = layer.visible ? layer.opacity : 0.0;
    nv.setOpacity(layer.volIndex, effectiveOpacity);
  };

  // Adjust mask overlay opacity
  const handleMaskOpacityChange = (layerIdx: number, newOpacity: number) => {
    const updated = [...maskLayers];
    const layer = updated[layerIdx];
    layer.opacity = newOpacity;
    setMaskLayers(updated);

    if (!nvRef.current) return;
    const nv = nvRef.current;
    if (layer.visible) {
      nv.setOpacity(layer.volIndex, newOpacity);
    }
  };

  // Programmatic slider navigation along specific orthogonal plane
  const handleSliderChange = (axis: 'z' | 'y' | 'x', newIdx: number) => {
    if (!caseDetail || !nvRef.current) return;
    const nv = nvRef.current;
    if (!nv.volumes || nv.volumes.length === 0) return;

    const newIndices = { ...sliceIndices, [axis]: newIdx };
    setSliceIndices(newIndices);

    // Update only the chosen orthogonal plane in crosshairPos:
    // In Niivue scene.crosshairPos: [0] = Sagittal (X), [1] = Coronal (Y), [2] = Axial (Z)
    const curPos = nv.scene.crosshairPos ? [...nv.scene.crosshairPos] : [0.5, 0.5, 0.5];
    if (axis === 'z') {
      curPos[2] = Math.max(0, Math.min(1, newIdx / Math.max(1, maxShapes[0] - 1)));
    } else if (axis === 'y') {
      curPos[1] = Math.max(0, Math.min(1, newIdx / Math.max(1, maxShapes[1] - 1)));
    } else if (axis === 'x') {
      curPos[0] = Math.max(0, Math.min(1, newIdx / Math.max(1, maxShapes[2] - 1)));
    }

    isInternalUpdateRef.current = true;
    nv.scene.crosshairPos = curPos;
    nv.drawScene();

    // Convert crosshair position to physical RAS mm
    try {
      const targetRas = nv.frac2mm(curPos) as [number, number, number];
      if (targetRas && !isNaN(targetRas[0])) {
        setCurrentRasMm(targetRas);
        const voxZyx = rasMmToVoxelZyx(targetRas, caseDetail);
        setCurrentVoxelZyx(voxZyx);
        onPointSelectedRef.current?.(targetRas);
      }
    } catch {
      // Fallback to voxel math if frac2mm fails
      const targetVoxelZyx: [number, number, number] = [newIndices.z, newIndices.y, newIndices.x];
      const targetRas = voxelZyxToRasMm(targetVoxelZyx, caseDetail);
      setCurrentRasMm(targetRas);
      setCurrentVoxelZyx(targetVoxelZyx);
      onPointSelectedRef.current?.(targetRas);
    } finally {
      setTimeout(() => {
        isInternalUpdateRef.current = false;
      }, 50);
    }
  };

  const stepSlice = (axis: 'z' | 'y' | 'x', delta: number) => {
    const maxVal = axis === 'z' ? maxShapes[0] : axis === 'y' ? maxShapes[1] : maxShapes[2];
    const curVal = sliceIndices[axis];
    const nextVal = Math.max(0, Math.min(maxVal - 1, curVal + delta));
    handleSliderChange(axis, nextVal);
  };

  const resetView = () => {
    if (!caseDetail || !nvRef.current) return;
    const centerZyx: [number, number, number] = [
      maxShapes[0] / 2,
      maxShapes[1] / 2,
      maxShapes[2] / 2,
    ];
    const centerRas = voxelZyxToRasMm(centerZyx, caseDetail);
    setCurrentRasMm(centerRas);
    setCurrentVoxelZyx(centerZyx);
    setSliceIndices({
      z: Math.round(centerZyx[0]),
      y: Math.round(centerZyx[1]),
      x: Math.round(centerZyx[2]),
    });

    const nv = nvRef.current;
    if (nv.volumes && nv.volumes.length > 0) {
      const frac = nv.mm2frac(centerRas);
      nv.scene.crosshairPos = frac;
      nv.drawScene();
    }
    onPointSelected?.(centerRas);
  };

  return (
    <div className="card slice-viewer-card">
      <div className="viewer-header">
        <div className="viewer-title">
          <Layers size={18} />
          <h2>Orthogonal MRI Slices & Overlays</h2>
          <span className="badge badge-subtle">Niivue WebGL 2</span>
        </div>

        <div className="viewer-toolbar">
          <div className="btn-group-toggle" role="group" aria-label="Slice view orientation">
            <button
              className={`btn btn-toolbar ${sliceMode === SLICE_TYPE.MULTIPLANAR ? 'active' : ''}`}
              onClick={() => handleSliceModeChange(SLICE_TYPE.MULTIPLANAR)}
              title="Multiplanar: Axial, Coronal, Sagittal"
            >
              Multiplanar
            </button>
            <button
              className={`btn btn-toolbar ${sliceMode === SLICE_TYPE.AXIAL ? 'active' : ''}`}
              onClick={() => handleSliceModeChange(SLICE_TYPE.AXIAL)}
              title="Axial View (Inferior-Superior plane)"
            >
              Axial
            </button>
            <button
              className={`btn btn-toolbar ${sliceMode === SLICE_TYPE.CORONAL ? 'active' : ''}`}
              onClick={() => handleSliceModeChange(SLICE_TYPE.CORONAL)}
              title="Coronal View (Anterior-Posterior plane)"
            >
              Coronal
            </button>
            <button
              className={`btn btn-toolbar ${sliceMode === SLICE_TYPE.SAGITTAL ? 'active' : ''}`}
              onClick={() => handleSliceModeChange(SLICE_TYPE.SAGITTAL)}
              title="Sagittal View (Left-Right plane)"
            >
              Sagittal
            </button>
          </div>

          <button
            className={`btn btn-toolbar ${showCrosshair ? 'active' : ''}`}
            onClick={handleToggleCrosshair}
            title="Toggle crosshair overlays"
            aria-label="Toggle crosshair"
          >
            <Crosshair size={13} />
            <span>Crosshair</span>
          </button>

          <button
            className="btn btn-toolbar"
            onClick={resetView}
            title="Reset focal point to volume center"
            aria-label="Reset slice view"
          >
            <RotateCcw size={13} />
            <span>Center</span>
          </button>

          <button
            className="btn btn-toolbar"
            onClick={() => loadVolumes()}
            disabled={loading}
            title="Reload NIfTI volume from backend"
            aria-label="Reload scans"
          >
            <RefreshCw className={loading ? 'icon-spin' : ''} size={13} />
          </button>
        </div>
      </div>

      {/* Dedicated Coordinate Readout Toolbar / Telemetry Bar (Non-overlapping) */}
      <div className="slice-telemetry-bar">
        <div className="slice-coordinate-hud">
          <div className="hud-item">
            <span className="hud-label">RAS Physical:</span>
            <span className="hud-value font-mono">{formatRasMm(currentRasMm, 2)}</span>
          </div>
          <div className="hud-item">
            <span className="hud-label">Voxel (z,y,x):</span>
            <span className="hud-value font-mono">{formatVoxelZyx(currentVoxelZyx, 1)}</span>
          </div>
          <div className="hud-item hud-sync">
            <span className="hud-dot"></span>
            <span className="hud-sync-text">Synchronized with 3D Scene</span>
          </div>
        </div>
      </div>

      <div ref={containerRef} className="slice-canvas-wrapper">
        <canvas ref={canvasRef} className="niivue-canvas" />

        {loading && (
          <div className="viewer-overlay-loading">
            <span className="spinner"></span>
            <span>{loadingStage || 'Streaming volumetric NIfTI data...'}</span>
          </div>
        )}

        {error && (
          <div className="viewer-overlay-error">
            <AlertTriangle size={24} />
            <p className="error-msg">{error}</p>
            <button className="btn btn-primary btn-sm" onClick={() => loadVolumes()}>
              Retry Loading
            </button>
          </div>
        )}

        {/* Disclaimer Watermark */}
        <div className="viewer-watermark">
          <div className="watermark-pill">
            <AlertTriangle size={12} />
            <span>SYNTHETIC PHANTOM VOLUME — RESEARCH PROTOTYPE</span>
          </div>
        </div>
      </div>

      {/* Slice Navigation Sliders Bar */}
      <div className="slice-navigation-bar">
        <div className="slice-slider-row">
          <span className="slider-axis-label axis-axial">Axial (Z/Sup):</span>
          <button
            className="btn-step"
            onClick={() => stepSlice('z', -1)}
            disabled={sliceIndices.z <= 0}
            title="Step down (Inferior)"
            aria-label="Step down (Inferior)"
          >
            <Minus size={12} />
          </button>
          <input
            type="range"
            className="slice-range-slider"
            min={0}
            max={maxShapes[0] - 1}
            value={sliceIndices.z}
            onChange={(e) => handleSliderChange('z', parseInt(e.target.value, 10))}
            aria-label="Axial slice slider"
          />
          <button
            className="btn-step"
            onClick={() => stepSlice('z', 1)}
            disabled={sliceIndices.z >= maxShapes[0] - 1}
            title="Step up (Superior)"
            aria-label="Step up (Superior)"
          >
            <Plus size={12} />
          </button>
          <span className="slice-counter font-mono">
            {sliceIndices.z + 1} / {maxShapes[0]}
          </span>
        </div>

        <div className="slice-slider-row">
          <span className="slider-axis-label axis-coronal">Coronal (Y/Ant):</span>
          <button
            className="btn-step"
            onClick={() => stepSlice('y', -1)}
            disabled={sliceIndices.y <= 0}
            title="Step back (Posterior)"
            aria-label="Step back (Posterior)"
          >
            <Minus size={12} />
          </button>
          <input
            type="range"
            className="slice-range-slider"
            min={0}
            max={maxShapes[1] - 1}
            value={sliceIndices.y}
            onChange={(e) => handleSliderChange('y', parseInt(e.target.value, 10))}
            aria-label="Coronal slice slider"
          />
          <button
            className="btn-step"
            onClick={() => stepSlice('y', 1)}
            disabled={sliceIndices.y >= maxShapes[1] - 1}
            title="Step forward (Anterior)"
            aria-label="Step forward (Anterior)"
          >
            <Plus size={12} />
          </button>
          <span className="slice-counter font-mono">
            {sliceIndices.y + 1} / {maxShapes[1]}
          </span>
        </div>

        <div className="slice-slider-row">
          <span className="slider-axis-label axis-sagittal">Sagittal (X/Right):</span>
          <button
            className="btn-step"
            onClick={() => stepSlice('x', -1)}
            disabled={sliceIndices.x <= 0}
            title="Step left"
            aria-label="Step left"
          >
            <Minus size={12} />
          </button>
          <input
            type="range"
            className="slice-range-slider"
            min={0}
            max={maxShapes[2] - 1}
            value={sliceIndices.x}
            onChange={(e) => handleSliderChange('x', parseInt(e.target.value, 10))}
            aria-label="Sagittal slice slider"
          />
          <button
            className="btn-step"
            onClick={() => stepSlice('x', 1)}
            disabled={sliceIndices.x >= maxShapes[2] - 1}
            title="Step right"
            aria-label="Step right"
          >
            <Plus size={12} />
          </button>
          <span className="slice-counter font-mono">
            {sliceIndices.x + 1} / {maxShapes[2]}
          </span>
        </div>
      </div>

      {/* Segmentation Mask Overlays Control */}
      <div className="slice-overlay-controls">
        <div className="overlay-controls-header">
          <Sliders size={13} />
          <span>Segmentation Mask Overlays:</span>
        </div>

        <div className="overlay-layers-list">
          <div className="overlay-layer-item layer-primary">
            <span className="layer-bullet" style={{ backgroundColor: '#94a3b8' }}></span>
            <span className="layer-name font-mono">Primary MRI (t1)</span>
            <span className="layer-role-tag">background</span>
          </div>

          {maskLayers.map((layer, idx) => (
            <div key={layer.name} className="overlay-layer-item">
              <button
                className={`btn-toggle-eye ${layer.visible ? 'eye-visible' : 'eye-hidden'}`}
                onClick={() => toggleMaskVisibility(idx)}
                title={layer.visible ? `Hide ${layer.name} mask` : `Show ${layer.name} mask`}
                aria-label={`Toggle ${layer.name} mask visibility`}
              >
                {layer.visible ? <Eye size={13} /> : <EyeOff size={13} />}
              </button>
              <span
                className="layer-bullet"
                style={{ backgroundColor: layer.colorHex }}
              ></span>
              <span className="layer-name font-mono">{layer.name}</span>
              <span className="layer-role-tag">({layer.role})</span>

              <div className="layer-opacity-control" title="Adjust mask overlay opacity">
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={layer.opacity}
                  onChange={(e) => handleMaskOpacityChange(idx, parseFloat(e.target.value))}
                  disabled={!layer.visible}
                  aria-label={`${layer.name} opacity`}
                />
                <span className="opacity-val">{Math.round(layer.opacity * 100)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
export default SliceViewer;
