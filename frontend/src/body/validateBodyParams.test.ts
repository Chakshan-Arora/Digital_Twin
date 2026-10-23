import { describe, expect, it } from 'vitest';
import type { BodyParams } from '../types/body';
import { validateBodyParams } from './validateBodyParams';

function createValidBodyParams(): BodyParams {
  return {
    height_mm: 1750,
    source: 'preset',
    joints: {
      head: [0, 1720, 0],
      neck: [0, 1500, 0],
      chest: [0, 1300, 0],
      pelvis: [0, 950, 0],
      l_shoulder: [-190, 1420, 0],
      l_elbow: [-220, 1150, 0],
      l_wrist: [-230, 910, 0],
      l_hand: [-235, 760, 0],
      r_shoulder: [190, 1420, 0],
      r_elbow: [220, 1150, 0],
      r_wrist: [230, 910, 0],
      r_hand: [235, 760, 0],
      l_hip: [-100, 900, 0],
      l_knee: [-110, 480, 0],
      l_ankle: [-120, 80, 0],
      l_foot: [-120, 20, 30],
      r_hip: [100, 900, 0],
      r_knee: [110, 480, 0],
      r_ankle: [120, 80, 0],
      r_foot: [120, 20, 30],
    },
  };
}

describe('validateBodyParams', () => {
  it('passes on a valid canonical body model', () => {
    const valid = createValidBodyParams();
    const result = validateBodyParams(valid);
    expect(result.ok).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it('rejects null or non-object inputs', () => {
    // @ts-expect-error Testing invalid input
    const resNull = validateBodyParams(null);
    expect(resNull.ok).toBe(false);
    expect(resNull.errors.length).toBeGreaterThan(0);

    // @ts-expect-error Testing invalid input
    const resString = validateBodyParams('invalid');
    expect(resString.ok).toBe(false);
  });

  it('rejects missing joints', () => {
    const body = createValidBodyParams();
    // @ts-expect-error Simulating missing joint
    delete body.joints.l_knee;
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('Missing required joints') && e.includes('l_knee'))).toBe(true);
  });

  it('rejects non-finite coordinate values', () => {
    const body = createValidBodyParams();
    body.joints.r_shoulder = [NaN, 1420, 0];
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('Invalid non-finite') && e.includes('r_shoulder'))).toBe(true);

    const bodyInf = createValidBodyParams();
    bodyInf.joints.head = [0, Infinity, 0];
    const resultInf = validateBodyParams(bodyInf);
    expect(resultInf.ok).toBe(false);
  });

  it('rejects height_mm outside the [1000, 2300] mm range', () => {
    const tooShort = createValidBodyParams();
    tooShort.height_mm = 850;
    const resShort = validateBodyParams(tooShort);
    expect(resShort.ok).toBe(false);
    expect(resShort.errors.some((e) => e.includes('height_mm') && e.includes('850.0'))).toBe(true);

    const tooTall = createValidBodyParams();
    tooTall.height_mm = 2450;
    const resTall = validateBodyParams(tooTall);
    expect(resTall.ok).toBe(false);
    expect(resTall.errors.some((e) => e.includes('height_mm') && e.includes('2450.0'))).toBe(true);
  });

  it('rejects an invalid source type', () => {
    const body = createValidBodyParams();
    // @ts-expect-error Simulating unexpected source
    body.source = 'random_source';
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('source must be'))).toBe(true);
  });

  it('rejects when head is not the highest joint in Y', () => {
    const body = createValidBodyParams();
    // Move neck higher than head
    body.joints.neck = [0, 1800, 0];
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('Head') && e.includes('not the highest'))).toBe(true);
  });

  it('rejects when feet are not the lowest joints in Y', () => {
    const body = createValidBodyParams();
    // Move knee down below feet
    body.joints.l_knee = [-110, 10, 0];
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('Feet are not the lowest'))).toBe(true);
  });

  it('rejects when shoulders or hips are on the wrong side of x=0', () => {
    const bodyShoulder = createValidBodyParams();
    // Move left shoulder to +X (wrong side)
    bodyShoulder.joints.l_shoulder = [120, 1420, 0];
    const resShoulder = validateBodyParams(bodyShoulder);
    expect(resShoulder.ok).toBe(false);
    expect(resShoulder.errors.some((e) => e.includes('l_shoulder must be on the person\'s left side'))).toBe(true);

    const bodyHip = createValidBodyParams();
    // Move right hip to -X (wrong side)
    bodyHip.joints.r_hip = [-80, 900, 0];
    const resHip = validateBodyParams(bodyHip);
    expect(resHip.ok).toBe(false);
    expect(resHip.errors.some((e) => e.includes('r_hip must be on the person\'s right side'))).toBe(true);
  });

  it('rejects implausible bone lengths', () => {
    const body = createValidBodyParams();
    // Upper leg between 250 and 650 mm; set knee to y=100 mm (making upper leg ~800 mm)
    body.joints.l_knee = [-110, 100, 0];
    body.joints.l_ankle = [-120, 80, 0];
    const result = validateBodyParams(body);
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.includes('left upper leg') && e.includes('outside plausible range'))).toBe(true);
  });
});
