import { type BodyParams, type JointName, type JointPosition3D } from '../types/body';
import { validateBodyParams, type ValidationResult } from '../body/validateBodyParams';

export interface BodyPreset {
  id: string;
  name: string;
  description: string;
  baseHeightMm: number;
  baseJoints: Record<JointName, JointPosition3D>;
}

/** Canonical reference joints matching CONTRACTS.md conventions and passing validateBodyParams */
const CANONICAL_JOINTS: Record<JointName, JointPosition3D> = {
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
};

export const BODY_PRESETS: BodyPreset[] = [
  {
    id: 'standard-1750',
    name: 'Standard Adult (1750 mm)',
    description: 'Reference 50th percentile adult stature with standard limb proportions.',
    baseHeightMm: 1750,
    baseJoints: CANONICAL_JOINTS,
  },
  {
    id: 'tall-1850',
    name: 'Tall Adult (1850 mm)',
    description: 'Elongated stature with proportionally extended femoral and tibial bone lengths.',
    baseHeightMm: 1850,
    baseJoints: CANONICAL_JOINTS,
  },
  {
    id: 'compact-1600',
    name: 'Compact Adult (1600 mm)',
    description: 'Petite stature with compressed axial and appendicular skeletal segments.',
    baseHeightMm: 1600,
    baseJoints: CANONICAL_JOINTS,
  },
];

/**
 * Scale reference canonical joints to a target stature in millimetres.
 * Preserves the floor plane at Y=0 and bilateral symmetry.
 */
export function scaleBodyParams(baseJoints: Record<JointName, JointPosition3D>, targetHeightMm: number): BodyParams {
  const scale = targetHeightMm / 1750;
  const scaledJoints: Record<JointName, JointPosition3D> = {} as Record<JointName, JointPosition3D>;

  for (const [key, [x, y, z]] of Object.entries(baseJoints)) {
    scaledJoints[key as JointName] = [
      Math.round(x * scale * 10) / 10,
      Math.round(y * scale * 10) / 10,
      Math.round(z * scale * 10) / 10,
    ];
  }

  return {
    height_mm: targetHeightMm,
    source: 'preset',
    joints: scaledJoints,
  };
}

export function getValidatedPreset(presetId: string, customHeightMm?: number): {
  params: BodyParams;
  validation: ValidationResult;
} {
  const preset = BODY_PRESETS.find((p) => p.id === presetId) || BODY_PRESETS[0];
  const height = customHeightMm ?? preset.baseHeightMm;
  const params = scaleBodyParams(preset.baseJoints, height);
  const validation = validateBodyParams(params);
  return { params, validation };
}

