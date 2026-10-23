import { describe, expect, it } from 'vitest';
import { BODY_PRESETS, getValidatedPreset, scaleBodyParams } from './presets';

describe('Body Presets and Stature Scaling', () => {
  it('defines the standard, tall, and compact presets', () => {
    expect(BODY_PRESETS.length).toBe(3);
    const ids = BODY_PRESETS.map((p) => p.id);
    expect(ids).toContain('standard-1750');
    expect(ids).toContain('tall-1850');
    expect(ids).toContain('compact-1600');
  });

  it('validates each preset against anatomical invariants', () => {
    for (const preset of BODY_PRESETS) {
      const result = getValidatedPreset(preset.id);
      expect(result.validation.ok).toBe(true);
      expect(result.validation.errors).toEqual([]);
      expect(result.params.height_mm).toBe(preset.baseHeightMm);
    }
  });

  it('scales reference joints to custom heights while maintaining validity', () => {
    const testHeights = [1500, 1650, 1750, 1850, 2000];
    for (const h of testHeights) {
      const result = getValidatedPreset('standard-1750', h);
      expect(result.validation.ok).toBe(true);
      expect(result.params.height_mm).toBe(h);

      // Invariants check
      const { head, l_shoulder, r_shoulder, l_foot, r_foot } = result.params.joints;
      expect(head[1]).toBeGreaterThan(1000);
      expect(l_shoulder[0]).toBeLessThan(0);
      expect(r_shoulder[0]).toBeGreaterThan(0);
      expect(l_foot[1]).toBeLessThan(100);
      expect(r_foot[1]).toBeLessThan(100);
    }
  });

  it('scaleBodyParams rounds coordinate numbers cleanly', () => {
    const standard = BODY_PRESETS[0];
    const scaled = scaleBodyParams(standard.baseJoints, 1800);
    for (const pt of Object.values(scaled.joints)) {
      expect(Number.isFinite(pt[0])).toBe(true);
      expect(Number.isFinite(pt[1])).toBe(true);
      expect(Number.isFinite(pt[2])).toBe(true);
    }
  });
});

