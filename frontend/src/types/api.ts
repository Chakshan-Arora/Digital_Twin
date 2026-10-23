/**
 * Data contracts and response interfaces mirroring the FastAPI HDT backend schemas.
 * Spatial coordinates are strictly in RAS millimetres.
 */

export interface HealthResponse {
  status: string;
}

export type StructureRole = 'target' | 'risk' | 'context';

export interface StructureSummary {
  name: string;
  role: StructureRole;
}

export interface CaseSummary {
  id: string;
  name: string;
  structures: StructureSummary[];
}

export interface CaseDetail {
  id: string;
  name: string;
  primary_image_name: string;
  spacing_mm: number[]; // [sx, sy, sz] under RAS convention
  spacing_zyx_mm: number[]; // [sz, sy, sx] internal
  shape: number[]; // [z, y, x]
  shape_zyx: number[];
  size: number[]; // [size_x, size_y, size_z]
  origin_ras_mm: number[]; // [x, y, z] in RAS mm
  direction_ras: number[][]; // 3x3 direction cosine matrix in RAS space
  structures: StructureSummary[];
  metadata: Record<string, unknown>;
  coordinate_system: 'RAS';
}

export interface ClearanceRequest {
  target_name: string;
  risk_name: string;
}

export interface ClosestPointsRAS {
  distance_mm: number;
  point_target_ras_mm: [number, number, number];
  point_risk_ras_mm: [number, number, number];
  point_a_ras_mm?: [number, number, number];
  point_b_ras_mm?: [number, number, number];
  coordinate_system: 'RAS';
}

export interface SurfaceDistanceStats {
  min: number;
  mean: number;
  hd95: number;
  max: number;
}

export interface ClearanceResponse {
  signed_clearance_mm: number;
  clearance_mm: number;
  convention: string;
  closest_points_mm: ClosestPointsRAS;
  surface_distance_stats: SurfaceDistanceStats;
  coordinate_system: 'RAS';
}

export interface UncertaintyRequest {
  target_name: string;
  risk_name: string;
  sigma_mm: number;
  corr_length_mm: number;
  bias_sigma_mm?: number;
  n_samples?: number; // 1 to 300
  seed?: number;
  threshold_mm?: number;
}

export interface UncertaintyResponse {
  baseline_mm: number;
  samples_mm: number[];
  mean: number;
  std: number;
  p05: number;
  p50: number;
  p95: number;
  p_below_threshold: number;
  p_below_ci95: [number, number];
  p_overlap: number;
  p_touching: number;
  n_samples: number;
  seed: number;
  threshold_mm: number;
  params: {
    sigma_mm?: number;
    corr_length_mm?: number;
    bias_sigma_mm?: number;
    [key: string]: unknown;
  };
  censored_fraction: number;
  convention: string;
  coordinate_system: 'RAS';
}
