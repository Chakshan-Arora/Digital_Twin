import { describe, expect, it } from 'vitest';
import type { CaseDetail } from '../types/api';
import {
  computeNiftiAffine4x4,
  formatRasMm,
  formatVoxelZyx,
  rasMmToVoxelZyx,
  voxelZyxToRasMm,
} from './coordinates';

// Mock case matching phantom_001 oblique anisotropic geometry
const mockObliqueCase: CaseDetail = {
  id: 'phantom_001',
  name: 'phantom_001',
  primary_image_name: 't1',
  spacing_mm: [1.0, 1.0, 2.0], // (sx, sy, sz) in RAS mm
  spacing_zyx_mm: [2.0, 1.0, 1.0], // (sz, sy, sx) in NumPy mm
  shape: [64, 64, 64],
  shape_zyx: [64, 64, 64],
  size: [64, 64, 64],
  origin_ras_mm: [-5.0, 20.0, 10.0],
  direction_ras: [
    [-0.9659258262890683, 0.0, -0.25881904510252074],
    [0.0, -1.0, 0.0],
    [-0.25881904510252074, 0.0, 0.9659258262890683],
  ],
  structures: [
    { name: 'tumor', role: 'target' },
    { name: 'vessel', role: 'risk' },
  ],
  metadata: {},
  coordinate_system: 'RAS',
};

// Simple isotropic identity case
const mockIsotropicCase: CaseDetail = {
  id: 'identity_case',
  name: 'identity_case',
  primary_image_name: 't1',
  spacing_mm: [1.0, 1.0, 1.0],
  spacing_zyx_mm: [1.0, 1.0, 1.0],
  shape: [32, 32, 32],
  shape_zyx: [32, 32, 32],
  size: [32, 32, 32],
  origin_ras_mm: [0.0, 0.0, 0.0],
  direction_ras: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ],
  structures: [],
  metadata: {},
  coordinate_system: 'RAS',
};

describe('Coordinate conversions', () => {
  it('maps voxel (0, 0, 0) to origin_ras_mm exactly', () => {
    const origin = voxelZyxToRasMm([0, 0, 0], mockObliqueCase);
    expect(origin[0]).toBeCloseTo(-5.0, 5);
    expect(origin[1]).toBeCloseTo(20.0, 5);
    expect(origin[2]).toBeCloseTo(10.0, 5);
  });

  it('inverts origin_ras_mm back to voxel (0, 0, 0) exactly', () => {
    const voxel = rasMmToVoxelZyx([-5.0, 20.0, 10.0], mockObliqueCase);
    expect(voxel[0]).toBeCloseTo(0.0, 5);
    expect(voxel[1]).toBeCloseTo(0.0, 5);
    expect(voxel[2]).toBeCloseTo(0.0, 5);
  });

  it('is a bijection: round-trip voxel -> RAS -> voxel preserves coordinates', () => {
    const testVoxels: [number, number, number][] = [
      [10, 20, 30],
      [32.5, 16.25, 48.75],
      [0, 63, 12],
      [63, 0, 63],
    ];

    for (const vox of testVoxels) {
      const ras = voxelZyxToRasMm(vox, mockObliqueCase);
      const recovered = rasMmToVoxelZyx(ras, mockObliqueCase);
      expect(recovered[0]).toBeCloseTo(vox[0], 5);
      expect(recovered[1]).toBeCloseTo(vox[1], 5);
      expect(recovered[2]).toBeCloseTo(vox[2], 5);
    }
  });

  it('handles identity case correctly', () => {
    const pt = voxelZyxToRasMm([10, 20, 5], mockIsotropicCase);
    // In NumPy (z, y, x): vz=10, vy=20, vx=5 -> in XYZ RAS: x=5, y=20, z=10
    expect(pt[0]).toBeCloseTo(5.0, 5);
    expect(pt[1]).toBeCloseTo(20.0, 5);
    expect(pt[2]).toBeCloseTo(10.0, 5);

    const vox = rasMmToVoxelZyx([5.0, 20.0, 10.0], mockIsotropicCase);
    expect(vox[0]).toBeCloseTo(10.0, 5);
    expect(vox[1]).toBeCloseTo(20.0, 5);
    expect(vox[2]).toBeCloseTo(5.0, 5);
  });

  it('computes 4x4 NIfTI affine matching sform', () => {
    const affine = computeNiftiAffine4x4(mockObliqueCase);
    expect(affine.length).toBe(4);
    expect(affine[3]).toEqual([0, 0, 0, 1]);
    expect(affine[0][3]).toBeCloseTo(-5.0, 5);
    expect(affine[1][3]).toBeCloseTo(20.0, 5);
    expect(affine[2][3]).toBeCloseTo(10.0, 5);
    // sz=2.0 along z column
    expect(affine[0][2]).toBeCloseTo(-0.25881904510252074 * 2.0, 5);
    expect(affine[2][2]).toBeCloseTo(0.9659258262890683 * 2.0, 5);
  });

  it('guarantees 1-voxel displacement along each axis strictly equals declared physical spacing', () => {
    const p0 = voxelZyxToRasMm([0, 0, 0], mockObliqueCase);
    const pz = voxelZyxToRasMm([1, 0, 0], mockObliqueCase);
    const py = voxelZyxToRasMm([0, 1, 0], mockObliqueCase);
    const px = voxelZyxToRasMm([0, 0, 1], mockObliqueCase);

    const distZ = Math.hypot(pz[0] - p0[0], pz[1] - p0[1], pz[2] - p0[2]);
    const distY = Math.hypot(py[0] - p0[0], py[1] - p0[1], py[2] - p0[2]);
    const distX = Math.hypot(px[0] - p0[0], px[1] - p0[1], px[2] - p0[2]);

    expect(distZ).toBeCloseTo(2.0, 5); // sz
    expect(distY).toBeCloseTo(1.0, 5); // sy
    expect(distX).toBeCloseTo(1.0, 5); // sx
  });

  it('is mathematically identical to NIfTI affine matrix multiplication', () => {
    const affine = computeNiftiAffine4x4(mockObliqueCase);
    const testPointsZyx: [number, number, number][] = [
      [0, 0, 0],
      [63, 63, 63],
      [32, 32, 32],
      [10.5, 20.3, 45.7],
      [0, 32, 63],
      [63, 0, 32],
    ];

    for (const [vz, vy, vx] of testPointsZyx) {
      const fromFunc = voxelZyxToRasMm([vz, vy, vx], mockObliqueCase);
      // NIfTI matrix affine applied to [vx, vy, vz, 1]
      const affX = affine[0][0] * vx + affine[0][1] * vy + affine[0][2] * vz + affine[0][3];
      const affY = affine[1][0] * vx + affine[1][1] * vy + affine[1][2] * vz + affine[1][3];
      const affZ = affine[2][0] * vx + affine[2][1] * vy + affine[2][2] * vz + affine[2][3];

      expect(fromFunc[0]).toBeCloseTo(affX, 5);
      expect(fromFunc[1]).toBeCloseTo(affY, 5);
      expect(fromFunc[2]).toBeCloseTo(affZ, 5);
    }
  });

  it('handles highly anisotropic (3.0, 0.5, 0.5) oblique geometry with exact displacement', () => {
    const theta = (30.0 * Math.PI) / 180.0;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    // Direction in RAS space with 30-degree rotation
    const anisotropicCase: CaseDetail = {
      id: 'aniso_30deg',
      name: 'aniso_30deg',
      primary_image_name: 't1',
      spacing_mm: [0.5, 0.5, 3.0], // (sx, sy, sz)
      spacing_zyx_mm: [3.0, 0.5, 0.5], // (sz, sy, sx)
      shape: [48, 64, 64],
      shape_zyx: [48, 64, 64],
      size: [64, 64, 48],
      origin_ras_mm: [-15.0, 35.0, -10.0],
      direction_ras: [
        [-cosT, 0.0, -sinT],
        [0.0, -1.0, 0.0],
        [-sinT, 0.0, cosT],
      ],
      structures: [],
      metadata: {},
      coordinate_system: 'RAS',
    };

    const p0 = voxelZyxToRasMm([0, 0, 0], anisotropicCase);
    const pz = voxelZyxToRasMm([1, 0, 0], anisotropicCase);
    const py = voxelZyxToRasMm([0, 1, 0], anisotropicCase);
    const px = voxelZyxToRasMm([0, 0, 1], anisotropicCase);

    expect(Math.hypot(pz[0] - p0[0], pz[1] - p0[1], pz[2] - p0[2])).toBeCloseTo(3.0, 5);
    expect(Math.hypot(py[0] - p0[0], py[1] - p0[1], py[2] - p0[2])).toBeCloseTo(0.5, 5);
    expect(Math.hypot(px[0] - p0[0], px[1] - p0[1], px[2] - p0[2])).toBeCloseTo(0.5, 5);

    // Round-trip bijection
    const testPoint: [number, number, number] = [24.2, 31.8, 45.1];
    const ras = voxelZyxToRasMm(testPoint, anisotropicCase);
    const recovered = rasMmToVoxelZyx(ras, anisotropicCase);
    expect(recovered[0]).toBeCloseTo(testPoint[0], 5);
    expect(recovered[1]).toBeCloseTo(testPoint[1], 5);
    expect(recovered[2]).toBeCloseTo(testPoint[2], 5);
  });

  it('formats text properly', () => {
    expect(formatRasMm([10.55, -20.44, 30.12])).toBe('R: 10.6, A: -20.4, S: 30.1 mm');
    expect(formatRasMm(null)).toBe('—');
    expect(formatVoxelZyx([10, 20, 30])).toBe('(z=10.0, y=20.0, x=30.0)');
    expect(formatVoxelZyx(null)).toBe('—');
  });
});
