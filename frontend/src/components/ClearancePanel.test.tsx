import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ClearancePanel } from './ClearancePanel';
import { apiClient } from '../services/apiClient';
import type { CaseDetail, ClearanceResponse } from '../types/api';

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

const mockClearanceResponse: ClearanceResponse = {
  signed_clearance_mm: 4.876,
  clearance_mm: 4.876,
  convention: 'face_to_face_voxel_sets',
  closest_points_mm: {
    distance_mm: 6.25,
    point_target_ras_mm: [12.0, 34.0, 56.0],
    point_risk_ras_mm: [18.0, 34.0, 56.0],
    coordinate_system: 'RAS',
  },
  surface_distance_stats: {
    min: 4.876,
    mean: 7.5,
    hd95: 11.2,
    max: 14.0,
  },
  coordinate_system: 'RAS',
};

describe('ClearancePanel', () => {
  beforeEach(() => {
    vi.spyOn(apiClient, 'computeClearance').mockResolvedValue(mockClearanceResponse);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders target and risk structure options from case detail', () => {
    const handleClearance = vi.fn();
    render(
      <ClearancePanel
        caseDetail={mockCaseDetail}
        onClearanceComputed={handleClearance}
      />
    );

    expect(screen.getByLabelText(/Target Structure/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Risk Structure/i)).toBeInTheDocument();
    expect(screen.getAllByText(/tumor \(target\)/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/vessel \(risk\)/i).length).toBeGreaterThan(0);
  });

  it('computes and displays dynamic clearance without hardcoded values', async () => {
    const handleClearance = vi.fn();
    render(
      <ClearancePanel
        caseDetail={mockCaseDetail}
        onClearanceComputed={handleClearance}
      />
    );

    const computeBtn = screen.getByRole('button', { name: /Calculate Clearance/i });
    fireEvent.click(computeBtn);

    await waitFor(() => {
      expect(apiClient.computeClearance).toHaveBeenCalledWith('test_case_01', {
        target_name: 'tumor',
        risk_name: 'vessel',
      });
    });

    expect(await screen.findByText(/\+4\.876 mm/i)).toBeInTheDocument();
    expect(screen.getByText(/Convention: face_to_face_voxel_sets/i)).toBeInTheDocument();
    expect(screen.getByText(/6\.250 mm/i)).toBeInTheDocument();
    expect(screen.getByText(/\[12\.00, 34\.00, 56\.00\]/i)).toBeInTheDocument();
    expect(handleClearance).toHaveBeenCalledWith(mockClearanceResponse);
  });

  it('displays error message when clearance calculation fails', async () => {
    vi.spyOn(apiClient, 'computeClearance').mockRejectedValueOnce(
      new Error('Spatial geometry mismatch between masks')
    );

    render(
      <ClearancePanel
        caseDetail={mockCaseDetail}
        onClearanceComputed={vi.fn()}
      />
    );

    const computeBtn = screen.getByRole('button', { name: /Calculate Clearance/i });
    fireEvent.click(computeBtn);

    expect(
      await screen.findByText(/Spatial geometry mismatch between masks/i)
    ).toBeInTheDocument();
  });
});
