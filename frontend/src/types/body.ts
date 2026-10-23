/**
 * Body Explorer Data Model and Geometry Contracts
 *
 * Coordinate System Conventions:
 * - Distance & position units: Millimetres (mm).
 * - Axis definitions:
 *   - +X: Person's left to right (from person's perspective: their own left is -X, their own right is +X).
 *   - +Y: Upward vertical axis (height above floor).
 *   - +Z: Forward direction (anterior).
 * - Origin (0, 0, 0): Situated on the floor plane directly midway between the left and right feet.
 * - Facing: The person stands erect and faces the +Z direction.
 * - Naming: The prefix "l_" refers to the person's own left side (anatomical left, -X).
 *           The prefix "r_" refers to the person's own right side (anatomical right, +X).
 */

export const JOINT_NAMES = [
  'head',
  'neck',
  'chest',
  'pelvis',
  'l_shoulder',
  'l_elbow',
  'l_wrist',
  'l_hand',
  'r_shoulder',
  'r_elbow',
  'r_wrist',
  'r_hand',
  'l_hip',
  'l_knee',
  'l_ankle',
  'l_foot',
  'r_hip',
  'r_knee',
  'r_ankle',
  'r_foot',
] as const;

export type JointName = (typeof JOINT_NAMES)[number];

export type JointPosition3D = [number, number, number];

export interface BodyParams {
  /** Total standing body stature in millimetres (mm). Plausible range: 1000mm to 2300mm. */
  height_mm: number;
  /** 3D coordinates [x, y, z] in mm for all standard skeletal joints. */
  joints: Record<JointName, JointPosition3D>;
  /** Origin source of the body parameters: browser vision camera scan vs preset model. */
  source: 'scan' | 'preset';
}
