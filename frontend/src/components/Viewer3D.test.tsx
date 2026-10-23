import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Viewer3D } from './Viewer3D';
import { apiClient } from '../services/apiClient';
import type { CaseDetail, ClosestPointsRAS } from '../types/api';

const mockCaseDetail: CaseDetail = {
  id: 'test_case_01',
  name: 'test_case_01',
  primary_image_name: 't1',
  spacing_mm: [1.0, 1.0, 2.0],
  spacing_zyx_mm: [2.0, 1.0, 1.0],
  shape: [64, 64, 64],
  shape_zyx: [64, 64, 64],
  size: [64, 64, 64],
  origin_ras_mm: [0, 0, 0],
  direction_ras: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  structures: [
    { name: 'tumor', role: 'target' },
    { name: 'vessel', role: 'risk' },
  ],
  metadata: {},
  coordinate_system: 'RAS',
};

const mockClosestPoints: ClosestPointsRAS = {
  distance_mm: 6.0,
  point_target_ras_mm: [10.0, 20.0, 30.0],
  point_risk_ras_mm: [16.0, 20.0, 30.0],
  coordinate_system: 'RAS',
};

const sampleObjText = `
v 10.0 20.0 30.0
v 11.0 20.0 30.0
v 10.0 21.0 30.0
f 1 2 3
`;

describe('Viewer3D', () => {
  beforeEach(() => {
    vi.spyOn(apiClient, 'getMeshObj').mockResolvedValue(sampleObjText);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the 3D viewer canvas and RAS coordinate indicators', () => {
    render(<Viewer3D caseDetail={mockCaseDetail} closestPoints={null} />);

    expect(screen.getByText(/3D Anatomy Viewer \(RAS Space\)/i)).toBeInTheDocument();
    expect(screen.getByText(/SYNTHETIC PHANTOM — DEMO ONLY/i)).toBeInTheDocument();
    expect(screen.getByText(/\+X: Right \(R\)/i)).toBeInTheDocument();
    expect(screen.getByText(/\+Y: Anterior \(A\)/i)).toBeInTheDocument();
    expect(screen.getByText(/\+Z: Superior \(S\)/i)).toBeInTheDocument();
  });

  it('renders structure controls for toggling visibility and opacity', () => {
    render(<Viewer3D caseDetail={mockCaseDetail} closestPoints={null} />);

    expect(screen.getByText('tumor')).toBeInTheDocument();
    expect(screen.getByText('vessel')).toBeInTheDocument();
    expect(screen.getByTitle(/Reset camera viewpoint/i)).toBeInTheDocument();
  });

  it('renders closest pair button when closest points are provided', () => {
    render(<Viewer3D caseDetail={mockCaseDetail} closestPoints={mockClosestPoints} />);

    expect(screen.getByRole('button', { name: /Closest Pair/i })).toBeInTheDocument();
  });

  it('renders synchronized slice crosshair button when synchronizedPointRas is provided', () => {
    render(
      <Viewer3D
        caseDetail={mockCaseDetail}
        closestPoints={null}
        synchronizedPointRas={[10.0, 20.0, 30.0]}
      />
    );

    expect(screen.getByRole('button', { name: /Slice Crosshair/i })).toBeInTheDocument();
  });
});
