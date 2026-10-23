import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { UncertaintyPanel } from './UncertaintyPanel';
import { apiClient } from '../services/apiClient';
import type { CaseDetail, UncertaintyResponse } from '../types/api';

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

const mockUncertaintyResponse: UncertaintyResponse = {
  baseline_mm: 6.0,
  samples_mm: [5.8, 6.1, 5.9, 6.2, 5.7, 6.0, 6.3],
  mean: 6.0,
  std: 0.2,
  p05: 5.7,
  p50: 6.0,
  p95: 6.3,
  p_below_threshold: 0.05,
  p_below_ci95: [0.01, 0.12],
  p_overlap: 0.0,
  p_touching: 0.0,
  n_samples: 100,
  seed: 42,
  threshold_mm: 2.0,
  params: {
    sigma_mm: 1.0,
    corr_length_mm: 5.0,
    bias_sigma_mm: 0.0,
  },
  censored_fraction: 0.0,
  convention: 'face_to_face_voxel_sets',
  coordinate_system: 'RAS',
};

describe('UncertaintyPanel', () => {
  beforeEach(() => {
    vi.spyOn(apiClient, 'computeUncertainty').mockResolvedValue(mockUncertaintyResponse);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders uncertainty parameter controls with cap explanation', () => {
    render(<UncertaintyPanel caseDetail={mockCaseDetail} />);

    expect(screen.getByText(/Realizations \(N\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Cap: 300 samples \(backend enforced\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Perturbation Amplitude/i)).toBeInTheDocument();
    expect(screen.getByText(/Correlation Length/i)).toBeInTheDocument();
    expect(screen.getByText(/Critical Distance Threshold/i)).toBeInTheDocument();
  });

  it('triggers Monte Carlo simulation and renders statistics and confidence interval', async () => {
    render(<UncertaintyPanel caseDetail={mockCaseDetail} />);

    const runBtn = screen.getByRole('button', {
      name: /Run Monte Carlo Uncertainty Analysis/i,
    });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(apiClient.computeUncertainty).toHaveBeenCalledWith('test_case_01', {
        target_name: 'tumor',
        risk_name: 'vessel',
        sigma_mm: 1.0,
        corr_length_mm: 5.0,
        bias_sigma_mm: 0.0,
        n_samples: 100,
        seed: 42,
        threshold_mm: 2.0,
      });
    });

    expect(await screen.findByText(/6\.000 ± 0\.200 mm/i)).toBeInTheDocument();
    expect(screen.getByText(/95% Wilson CI: \[1\.0%, 12\.0%\]/i)).toBeInTheDocument();
    expect(screen.getByText(/P\(overlap\): 0\.0%/i)).toBeInTheDocument();
    expect(screen.getByText(/Model Assumption Disclaimer/i)).toBeInTheDocument();
  });
});
