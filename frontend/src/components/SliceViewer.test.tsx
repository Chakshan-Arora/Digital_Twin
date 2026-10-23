import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SliceViewer } from './SliceViewer';
import type { CaseDetail } from '../types/api';

// Mock Niivue
vi.mock('@niivue/niivue', () => {
  const SLICE_TYPE = {
    AXIAL: 0,
    CORONAL: 1,
    SAGITTAL: 2,
    MULTIPLANAR: 3,
    RENDER: 4,
  };

  const SHOW_RENDER = {
    NEVER: 0,
    ALWAYS: 1,
    AUTO: 2,
  };

  class MockNiivue {
    opts: Record<string, unknown>;
    volumes: unknown[] = [];
    scene = { crosshairPos: [0.5, 0.5, 0.5] };
    onLocationChange: ((data: { mm?: number[]; vox?: number[] }) => void) | null = null;

    constructor(opts: Record<string, unknown> = {}) {
      this.opts = opts;
    }

    attachToCanvas = vi.fn();
    loadVolumes = vi.fn().mockImplementation(async (volList) => {
      this.volumes = volList;
      return volList;
    });
    setSliceType = vi.fn();
    setCrosshairWidth = vi.fn();
    setOpacity = vi.fn();
    drawScene = vi.fn();
    resizeListener = vi.fn();
    mm2frac = vi.fn().mockReturnValue([0.5, 0.5, 0.5]);
    frac2mm = vi.fn().mockReturnValue([-5.0, 20.0, 10.0]);
  }

  return {
    Niivue: MockNiivue,
    SLICE_TYPE,
    SHOW_RENDER,
  };
});

// Mock apiClient
vi.mock('../services/apiClient', () => ({
  apiClient: {
    getVolumeDownloadUrl: vi.fn((id: string) => `http://127.0.0.1:8000/cases/${id}/volume.nii.gz`),
    getMaskDownloadUrl: vi.fn((id: string, s: string) => `http://127.0.0.1:8000/cases/${id}/structures/${s}/mask.nii.gz`),
  },
}));

const mockCaseDetail: CaseDetail = {
  id: 'phantom_001',
  name: 'phantom_001',
  primary_image_name: 't1',
  spacing_mm: [1.0, 1.0, 2.0],
  spacing_zyx_mm: [2.0, 1.0, 1.0],
  shape: [64, 64, 64],
  shape_zyx: [64, 64, 64],
  size: [64, 64, 64],
  origin_ras_mm: [-5.0, 20.0, 10.0],
  direction_ras: [
    [-0.9659, 0, -0.2588],
    [0, -1, 0],
    [-0.2588, 0, 0.9659],
  ],
  structures: [
    { name: 'tumor', role: 'target' },
    { name: 'vessel', role: 'risk' },
  ],
  metadata: {},
  coordinate_system: 'RAS',
};

describe('SliceViewer Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Mock HTMLCanvasElement getContext for WebGL 2 in JSDOM
    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((contextId) => {
      if (contextId === 'webgl2') {
        return {
          viewport: vi.fn(),
          clearColor: vi.fn(),
          clear: vi.fn(),
          getExtension: vi.fn(),
        };
      }
      return null;
    }) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  it('renders the orthogonal slice viewer canvas, HUD with RAS coordinates and watermark', () => {
    const { container } = render(<SliceViewer caseDetail={mockCaseDetail} />);

    expect(screen.getByRole('heading', { name: /Orthogonal MRI Slices & Overlays/i })).toBeInTheDocument();
    expect(screen.getByText(/Niivue WebGL 2/i)).toBeInTheDocument();
    expect(screen.getByText(/SYNTHETIC PHANTOM VOLUME — RESEARCH PROTOTYPE/i)).toBeInTheDocument();
    expect(screen.getByText(/RAS Physical:/i)).toBeInTheDocument();
    expect(screen.getByText(/Synchronized with 3D Scene/i)).toBeInTheDocument();

    // Verify coordinate readout is positioned in dedicated telemetry bar outside slice-canvas-wrapper
    const telemetryBar = container.querySelector('.slice-telemetry-bar');
    const canvasWrapper = container.querySelector('.slice-canvas-wrapper');
    expect(telemetryBar).toBeInTheDocument();
    expect(canvasWrapper).toBeInTheDocument();
    expect(telemetryBar?.querySelector('.slice-coordinate-hud')).toBeInTheDocument();
    expect(canvasWrapper?.querySelector('.slice-coordinate-hud')).toBeNull();
  });

  it('renders orientation switcher buttons and changes slice modes', () => {
    render(<SliceViewer caseDetail={mockCaseDetail} />);

    const multiplanarBtn = screen.getByRole('button', { name: /Multiplanar/i });
    const axialBtn = screen.getByRole('button', { name: /Axial/i });
    const coronalBtn = screen.getByRole('button', { name: /Coronal/i });
    const sagittalBtn = screen.getByRole('button', { name: /Sagittal/i });

    expect(multiplanarBtn).toBeInTheDocument();
    expect(axialBtn).toBeInTheDocument();
    expect(coronalBtn).toBeInTheDocument();
    expect(sagittalBtn).toBeInTheDocument();

    fireEvent.click(axialBtn);
    expect(axialBtn).toHaveClass('active');

    fireEvent.click(coronalBtn);
    expect(coronalBtn).toHaveClass('active');
  });

  it('renders slice navigation sliders for Axial, Coronal, and Sagittal axes', () => {
    render(<SliceViewer caseDetail={mockCaseDetail} />);

    expect(screen.getByText(/Axial \(Z\/Sup\):/i)).toBeInTheDocument();
    expect(screen.getByText(/Coronal \(Y\/Ant\):/i)).toBeInTheDocument();
    expect(screen.getByText(/Sagittal \(X\/Right\):/i)).toBeInTheDocument();

    const axialSlider = screen.getByRole('slider', { name: /Axial slice slider/i });
    expect(axialSlider).toBeInTheDocument();
    expect(axialSlider).toHaveAttribute('max', '63');

    // Step up button
    const stepUpBtn = screen.getByRole('button', { name: /Step up \(Superior\)/i });
    fireEvent.click(stepUpBtn);
  });

  it('renders structure mask overlay controls for tumor and vessel with opacity sliders', async () => {
    render(<SliceViewer caseDetail={mockCaseDetail} />);

    expect(screen.getByText(/Primary MRI \(t1\)/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/tumor/i)).toBeInTheDocument();
      expect(screen.getByText(/vessel/i)).toBeInTheDocument();
    });

    // Toggle tumor visibility
    const tumorToggleBtn = screen.getByRole('button', { name: /Toggle tumor mask visibility/i });
    expect(tumorToggleBtn).toBeInTheDocument();
    fireEvent.click(tumorToggleBtn);

    // Opacity slider
    const tumorOpacitySlider = screen.getByRole('slider', { name: /tumor opacity/i });
    expect(tumorOpacitySlider).toBeInTheDocument();
    fireEvent.change(tumorOpacitySlider, { target: { value: '0.85' } });
  });

  it('fires onPointSelected callback when user steps a slice slider', () => {
    const handlePointSelected = vi.fn();
    render(<SliceViewer caseDetail={mockCaseDetail} onPointSelected={handlePointSelected} />);

    const stepUpBtn = screen.getByRole('button', { name: /Step up \(Superior\)/i });
    fireEvent.click(stepUpBtn);

    expect(handlePointSelected).toHaveBeenCalled();
    const calledWith = handlePointSelected.mock.calls[0][0];
    expect(calledWith.length).toBe(3); // [x, y, z] in RAS mm
  });
});
