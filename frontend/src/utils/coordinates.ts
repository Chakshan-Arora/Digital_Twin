/**
 * Pure coordinate conversion and verification utilities for HDT imaging data.
 *
 * All functions operate strictly in millimetres and obey the project's spatial conventions:
 * - Internal volume indexing: NumPy order (z, y, x)
 * - Physical wire & viewer convention: RAS (Right, Anterior, Superior) mm
 * - ITK / SimpleITK physical convention: LPS (Left, Posterior, Superior) mm
 */

import type { CaseDetail } from '../types/api';

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Convert continuous voxel index (z, y, x) to physical RAS coordinates (x, y, z) in mm.
 *
 * Math:
 *   scaled_xyz = (x * sx, y * sy, z * sz)
 *   p_LPS = origin_LPS + D_LPS @ scaled_xyz
 *   p_RAS = (-p_LPS.x, -p_LPS.y, p_LPS.z)
 */
export function voxelZyxToRasMm(
  voxelZyx: [number, number, number],
  caseDetail: CaseDetail
): [number, number, number] {
  const [vz, vy, vx] = voxelZyx;
  // CaseDetail spacing_mm is [sx, sy, sz] in RAS, spacing_zyx_mm is [sz, sy, sx]
  const [sz, sy, sx] = caseDetail.spacing_zyx_mm || [
    caseDetail.spacing_mm[2],
    caseDetail.spacing_mm[1],
    caseDetail.spacing_mm[0],
  ];

  // Continuous scaled vector in XYZ order
  const scaledX = vx * sx;
  const scaledY = vy * sy;
  const scaledZ = vz * sz;

  // CaseDetail direction_ras is 3x3 matrix where row i, col j is D_RAS[i][j]
  // In RAS space: p_RAS = origin_RAS + D_RAS @ scaled_xyz
  const d = caseDetail.direction_ras;
  const [ox, oy, oz] = caseDetail.origin_ras_mm;

  const rasX = ox + d[0][0] * scaledX + d[0][1] * scaledY + d[0][2] * scaledZ;
  const rasY = oy + d[1][0] * scaledX + d[1][1] * scaledY + d[1][2] * scaledZ;
  const rasZ = oz + d[2][0] * scaledX + d[2][1] * scaledY + d[2][2] * scaledZ;

  return [rasX, rasY, rasZ];
}

/**
 * Invert physical RAS coordinates (x, y, z) in mm back to continuous voxel indices (z, y, x).
 *
 * Since D_RAS is an orthogonal direction cosine matrix, its inverse is its transpose:
 *   (p_RAS - origin_RAS) = D_RAS @ scaled_xyz
 *   scaled_xyz = D_RAS^T @ (p_RAS - origin_RAS)
 *   vx = scaled_xyz.x / sx
 *   vy = scaled_xyz.y / sy
 *   vz = scaled_xyz.z / sz
 */
export function rasMmToVoxelZyx(
  rasMm: [number, number, number],
  caseDetail: CaseDetail
): [number, number, number] {
  const [rx, ry, rz] = rasMm;
  const [ox, oy, oz] = caseDetail.origin_ras_mm;
  const d = caseDetail.direction_ras;

  const dx = rx - ox;
  const dy = ry - oy;
  const dz = rz - oz;

  // D_RAS^T @ delta
  const scaledX = d[0][0] * dx + d[1][0] * dy + d[2][0] * dz;
  const scaledY = d[0][1] * dx + d[1][1] * dy + d[2][1] * dz;
  const scaledZ = d[0][2] * dx + d[1][2] * dy + d[2][2] * dz;

  const [sz, sy, sx] = caseDetail.spacing_zyx_mm || [
    caseDetail.spacing_mm[2],
    caseDetail.spacing_mm[1],
    caseDetail.spacing_mm[0],
  ];

  const vx = scaledX / sx;
  const vy = scaledY / sy;
  const vz = scaledZ / sz;

  return [vz, vy, vx];
}

/**
 * Compute the 4x4 affine matrix mapping continuous NIfTI voxel coordinates (x, y, z)
 * to physical RAS millimetres.
 *
 * Affine = [ D_RAS @ diag(sx, sy, sz) | origin_RAS ]
 *          [ 0, 0, 0                  | 1          ]
 */
export function computeNiftiAffine4x4(caseDetail: CaseDetail): number[][] {
  const [sz, sy, sx] = caseDetail.spacing_zyx_mm || [
    caseDetail.spacing_mm[2],
    caseDetail.spacing_mm[1],
    caseDetail.spacing_mm[0],
  ];
  const d = caseDetail.direction_ras;
  const [ox, oy, oz] = caseDetail.origin_ras_mm;

  return [
    [d[0][0] * sx, d[0][1] * sy, d[0][2] * sz, ox],
    [d[1][0] * sx, d[1][1] * sy, d[1][2] * sz, oy],
    [d[2][0] * sx, d[2][1] * sy, d[2][2] * sz, oz],
    [0, 0, 0, 1],
  ];
}

/**
 * Format physical RAS coordinates into human-readable anatomical text.
 */
export function formatRasMm(coords: [number, number, number] | null | undefined, decimals = 1): string {
  if (!coords) return '—';
  const [x, y, z] = coords;
  return `R: ${x.toFixed(decimals)}, A: ${y.toFixed(decimals)}, S: ${z.toFixed(decimals)} mm`;
}

/**
 * Format continuous voxel indices into human-readable text.
 */
export function formatVoxelZyx(indices: [number, number, number] | null | undefined, decimals = 1): string {
  if (!indices) return '—';
  const [z, y, x] = indices;
  return `(z=${z.toFixed(decimals)}, y=${y.toFixed(decimals)}, x=${x.toFixed(decimals)})`;
}
