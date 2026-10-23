import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient, ApiError, getBaseUrl } from './apiClient';

describe('apiClient', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses default localhost:8000 when environment variable is unset', () => {
    const url = getBaseUrl();
    expect(url).toBe('http://127.0.0.1:8000');
  });

  it('successfully fetches health status', async () => {
    const mockHealth = { status: 'ok' };
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockHealth,
    });

    const res = await apiClient.checkHealth();
    expect(res).toEqual(mockHealth);
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/health', expect.any(Object));
  });

  it('lists cases and their structures from the API', async () => {
    const mockCases = [
      {
        id: 'case_alpha',
        name: 'case_alpha',
        structures: [
          { name: 'tumor', role: 'target' },
          { name: 'vessel', role: 'risk' },
        ],
      },
    ];
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockCases,
    });

    const res = await apiClient.listCases();
    expect(res).toEqual(mockCases);
  });

  it('retrieves case detail with spatial geometry in RAS mm', async () => {
    const mockDetail = {
      id: 'case_alpha',
      name: 'case_alpha',
      primary_image_name: 't1',
      spacing_mm: [1.0, 1.0, 2.0],
      spacing_zyx_mm: [2.0, 1.0, 1.0],
      shape: [64, 64, 64],
      shape_zyx: [64, 64, 64],
      size: [64, 64, 64],
      origin_ras_mm: [-5.0, 20.0, 10.0],
      direction_ras: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      structures: [{ name: 'tumor', role: 'target' }],
      metadata: {},
      coordinate_system: 'RAS',
    };
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockDetail,
    });

    const res = await apiClient.getCaseDetail('case_alpha');
    expect(res).toEqual(mockDetail);
    expect(res.coordinate_system).toBe('RAS');
  });

  it('calculates clearance with custom structures', async () => {
    const mockClearance = {
      signed_clearance_mm: 5.432,
      clearance_mm: 5.432,
      convention: 'face_to_face_voxel_sets',
      closest_points_mm: {
        distance_mm: 7.123,
        point_target_ras_mm: [10.5, 20.2, 30.1],
        point_risk_ras_mm: [15.2, 22.1, 32.0],
        coordinate_system: 'RAS',
      },
      surface_distance_stats: {
        min: 5.432,
        mean: 8.91,
        hd95: 12.45,
        max: 15.2,
      },
      coordinate_system: 'RAS',
    };
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockClearance,
    });

    const res = await apiClient.computeClearance('case_alpha', {
      target_name: 'tumor',
      risk_name: 'vessel',
    });
    expect(res.signed_clearance_mm).toBe(5.432);
    expect(res.closest_points_mm.coordinate_system).toBe('RAS');
  });

  it('handles backend 404 error with detail message', async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Case 'unknown_case' not found." }),
    });

    const promise = apiClient.getCaseDetail('unknown_case');
    await expect(promise).rejects.toThrow(ApiError);
  });

  it('handles backend 422 validation error when sample count exceeds cap', async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{ msg: 'Input should be less than or equal to 300' }],
      }),
    });

    const promise = apiClient.computeUncertainty('case_alpha', {
      target_name: 'tumor',
      risk_name: 'vessel',
      sigma_mm: 1.0,
      corr_length_mm: 5.0,
      n_samples: 301,
    });
    await expect(promise).rejects.toThrow('Input should be less than or equal to 300');
  });

  it('handles network connection refusal when backend is offline', async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error('Failed to fetch')
    );

    const promise = apiClient.checkHealth();
    await expect(promise).rejects.toThrow(ApiError);
  });
});
