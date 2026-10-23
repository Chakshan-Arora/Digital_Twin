import { type BodyParams, JOINT_NAMES, type JointName, type JointPosition3D } from '../types/body';

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

export interface BoneSegmentRule {
  from: JointName;
  to: JointName;
  name: string;
  minMm: number;
  maxMm: number;
}

export const BONE_SEGMENT_RULES: BoneSegmentRule[] = [
  // Axial / spine
  { from: 'pelvis', to: 'chest', name: 'pelvis to chest', minMm: 150, maxMm: 500 },
  { from: 'chest', to: 'neck', name: 'chest to neck', minMm: 80, maxMm: 350 },
  { from: 'neck', to: 'head', name: 'neck to head', minMm: 80, maxMm: 350 },
  // Upper legs
  { from: 'l_hip', to: 'l_knee', name: 'left upper leg', minMm: 250, maxMm: 650 },
  { from: 'r_hip', to: 'r_knee', name: 'right upper leg', minMm: 250, maxMm: 650 },
  // Lower legs
  { from: 'l_knee', to: 'l_ankle', name: 'left lower leg', minMm: 250, maxMm: 600 },
  { from: 'r_knee', to: 'r_ankle', name: 'right lower leg', minMm: 250, maxMm: 600 },
  // Upper arms
  { from: 'l_shoulder', to: 'l_elbow', name: 'left upper arm', minMm: 180, maxMm: 500 },
  { from: 'r_shoulder', to: 'r_elbow', name: 'right upper arm', minMm: 180, maxMm: 500 },
  // Forearms
  { from: 'l_elbow', to: 'l_wrist', name: 'left forearm', minMm: 150, maxMm: 450 },
  { from: 'r_elbow', to: 'r_wrist', name: 'right forearm', minMm: 150, maxMm: 450 },
];

function euclideanDistance(a: JointPosition3D, b: JointPosition3D): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Validates a BodyParams object against geometric and anatomical plausibility rules.
 *
 * Checks:
 * 1. Object structure and presence of all 20 required joints.
 * 2. All coordinate values and height_mm are finite numbers.
 * 3. height_mm is within the plausible human stature range [1000mm, 2300mm].
 * 4. The head is strictly the highest joint in Y.
 * 5. The feet (l_foot, r_foot) are the lowest joints in Y.
 * 6. Left and right bilateral joints (shoulders, hips) sit on opposite sides of the x=0 sagittal midline.
 * 7. Key skeletal bone segments fall within anatomically plausible length bounds.
 */
export function validateBodyParams(params: BodyParams): ValidationResult {
  const errors: string[] = [];

  if (!params || typeof params !== 'object') {
    return { ok: false, errors: ['BodyParams must be an object'] };
  }

  // 1. Validate height_mm
  if (typeof params.height_mm !== 'number' || !Number.isFinite(params.height_mm)) {
    errors.push(`height_mm must be a finite number; got ${params.height_mm}`);
  } else if (params.height_mm < 1000 || params.height_mm > 2300) {
    errors.push(
      `height_mm (${params.height_mm.toFixed(1)} mm) is outside the plausible range [1000, 2300] mm`
    );
  }

  // 2. Validate source
  if (params.source !== 'scan' && params.source !== 'preset') {
    errors.push(`source must be 'scan' or 'preset'; got '${params.source}'`);
  }

  // 3. Check joints dictionary
  if (!params.joints || typeof params.joints !== 'object') {
    errors.push('joints property is missing or not an object');
    return { ok: false, errors };
  }

  // Check presence and finiteness of all 20 joints
  const missingJoints: string[] = [];
  const invalidCoordinateJoints: string[] = [];

  for (const jointName of JOINT_NAMES) {
    const pt = params.joints[jointName];
    if (!pt) {
      missingJoints.push(jointName);
      continue;
    }
    if (
      !Array.isArray(pt) ||
      pt.length !== 3 ||
      typeof pt[0] !== 'number' ||
      !Number.isFinite(pt[0]) ||
      typeof pt[1] !== 'number' ||
      !Number.isFinite(pt[1]) ||
      typeof pt[2] !== 'number' ||
      !Number.isFinite(pt[2])
    ) {
      invalidCoordinateJoints.push(jointName);
    }
  }

  if (missingJoints.length > 0) {
    errors.push(`Missing required joints: ${missingJoints.join(', ')}`);
  }
  if (invalidCoordinateJoints.length > 0) {
    errors.push(`Invalid non-finite or malformed coordinates for joints: ${invalidCoordinateJoints.join(', ')}`);
  }

  // If there are structural/coordinate errors, skip geometrical relationship checks
  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const { joints } = params;

  // 4. Head must be the highest joint in Y
  const headY = joints.head[1];
  for (const jName of JOINT_NAMES) {
    if (jName === 'head') continue;
    if (joints[jName][1] >= headY) {
      errors.push(
        `Head (y=${headY.toFixed(1)} mm) is not the highest joint; joint '${jName}' has y=${joints[jName][1].toFixed(1)} mm`
      );
    }
  }

  // 5. Feet must be the lowest joints in Y
  const footLowestY = Math.max(joints.l_foot[1], joints.r_foot[1]);
  for (const jName of JOINT_NAMES) {
    if (jName === 'l_foot' || jName === 'r_foot') continue;
    if (joints[jName][1] < footLowestY) {
      errors.push(
        `Feet are not the lowest joints; joint '${jName}' (y=${joints[jName][1].toFixed(1)} mm) is below foot (y=${footLowestY.toFixed(1)} mm)`
      );
    }
  }

  // 6. Left and right bilateral joints must be on opposite sides of x=0 midline
  // Person's left (l_) is -X, person's right (r_) is +X
  if (joints.l_shoulder[0] >= 0) {
    errors.push(
      `l_shoulder must be on the person's left side (x < 0); got x=${joints.l_shoulder[0].toFixed(1)} mm`
    );
  }
  if (joints.r_shoulder[0] <= 0) {
    errors.push(
      `r_shoulder must be on the person's right side (x > 0); got x=${joints.r_shoulder[0].toFixed(1)} mm`
    );
  }
  if (joints.l_hip[0] >= 0) {
    errors.push(
      `l_hip must be on the person's left side (x < 0); got x=${joints.l_hip[0].toFixed(1)} mm`
    );
  }
  if (joints.r_hip[0] <= 0) {
    errors.push(
      `r_hip must be on the person's right side (x > 0); got x=${joints.r_hip[0].toFixed(1)} mm`
    );
  }

  // 7. Plausible bone segment lengths
  for (const rule of BONE_SEGMENT_RULES) {
    const ptA = joints[rule.from];
    const ptB = joints[rule.to];
    const lengthMm = euclideanDistance(ptA, ptB);
    if (lengthMm < rule.minMm || lengthMm > rule.maxMm) {
      errors.push(
        `Bone segment '${rule.name}' length (${lengthMm.toFixed(1)} mm) is outside plausible range [${rule.minMm}, ${rule.maxMm}] mm`
      );
    }
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}
