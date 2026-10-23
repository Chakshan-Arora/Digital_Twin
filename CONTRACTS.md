# Human Digital Twin (HDT) — Body Explorer Contracts & Ownership

## Purpose
This document establishes the architecture, geometric data contracts, and folder ownership boundaries for the 4-person parallel hackathon build adding the **Body Explorer** mode to the Human Digital Twin (HDT) platform.

> [!IMPORTANT]
> **Preservation Rule:** The existing **Clearance Lab** (tumor/vessel clearance, NIfTI slice viewer, Monte Carlo uncertainty) must continue to function completely untouched. All existing tests must remain passing.

---

## 1. 3D Coordinate System & Geometry Conventions

All 3D procedural body, scanner, and nervous system modules strictly adhere to the following coordinate frame:

| Dimension / Property | Convention |
| :--- | :--- |
| **Units** | Millimetres (**mm**) |
| **+X Axis** | Person's anatomical left to right (Person's own left is $-X$, person's own right is $+X$) |
| **+Y Axis** | Upward vertical axis (height above the floor) |
| **+Z Axis** | Forward horizontal axis (anterior / facing direction) |
| **Origin `(0, 0, 0)`** | Situated on the floor plane directly midway between the left and right feet |
| **Facing** | The human figure stands erect facing $+Z$ |
| **Naming Prefix `l_`** | Person's own left side (anatomical left, $-X$) |
| **Naming Prefix `r_`** | Person's own right side (anatomical right, $+X$) |

---

## 2. The `BodyParams` Contract

Defined in [`frontend/src/types/body.ts`](file:///frontend/src/types/body.ts) and validated by [`frontend/src/body/validateBodyParams.ts`](file:///frontend/src/body/validateBodyParams.ts).

### TypeScript Interface
```typescript
export const JOINT_NAMES = [
  'head', 'neck', 'chest', 'pelvis',
  'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',
  'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand',
  'l_hip', 'l_knee', 'l_ankle', 'l_foot',
  'r_hip', 'r_knee', 'r_ankle', 'r_foot',
] as const;

export type JointName = (typeof JOINT_NAMES)[number];

export type JointPosition3D = [number, number, number];

export interface BodyParams {
  /** Standing height from floor to vertex in millimetres (1000 - 2300 mm). */
  height_mm: number;
  /** 3D spatial positions [x, y, z] in mm for each of the 20 canonical joints. */
  joints: Record<JointName, JointPosition3D>;
  /** Origin source: browser camera scan vs preset reference. */
  source: 'scan' | 'preset';
}
```

### Validation Invariants (`validateBodyParams`)
1. **Completeness**: All 20 joints must be present with finite 3D numeric coordinates.
2. **Stature**: `height_mm` must be finite and within `[1000, 2300]` mm.
3. **Head Height**: `joints.head[1]` must be strictly greater than all other joint $Y$ coordinates.
4. **Foot Floor Plane**: `joints.l_foot[1]` and `joints.r_foot[1]` must be the lowest joints in $Y$.
5. **Midline Symmetry**: `l_shoulder[0] < 0 < r_shoulder[0]` and `l_hip[0] < 0 < r_hip[0]`.
6. **Bone Length Limits**:
   - Spine: Pelvis-to-chest `[150, 500]` mm, chest-to-neck `[80, 350]` mm, neck-to-head `[80, 350]` mm.
   - Legs: Upper leg (hip-to-knee) `[250, 650]` mm, lower leg (knee-to-ankle) `[250, 600]` mm.
   - Arms: Upper arm (shoulder-to-elbow) `[180, 500]` mm, forearm (elbow-to-wrist) `[150, 450]` mm.

---

## 3. Team Ownership Boundaries & Contracts

```mermaid
flowchart TD
    Scanner["Person 2: Camera Scanner\n(frontend/src/scanner)"] -->|BodyParams| State["Application State / Mode Router\n(frontend/src/layout, App.tsx)"]
    State -->|BodyParams| Body["Person 3: 3D Procedural Body\n(frontend/src/body)"]
    Body -->|Skin Hit Point & Normal [x,y,z]| Atlas["Person 4: Atlas & Nervous System\n(frontend/src/atlas, nervous, backend)"]
    Atlas -->|Dermatome & Nerve Path Data| Body
    State --> UI["Person 1: Layout & UX\n(frontend/src/layout, App.tsx, components)"]
```

### Team Roster & Module Responsibilities

| Role | Assigned Directories & Files | Primary Responsibilities |
| :--- | :--- | :--- |
| **Person 1: Frontend & UX** | `frontend/src/layout/`, `frontend/src/App.tsx`, `frontend/src/components/` | Mode switcher (Clearance Lab vs Body Explorer), top-level workstation layout, UI controls, notifications. |
| **Person 2: Scanner** | `frontend/src/scanner/` | Browser webcam stream, MediaPipe Pose tracking, landmark-to-joint calibration, generating valid `BodyParams`. |
| **Person 3: 3D Body** | `frontend/src/body/` | Procedural 3D body mesh generator (Three.js), skeletal posing, raycasting/click-on-skin detection. |
| **Person 4: Atlas & Integrator** | `frontend/src/atlas/`, `frontend/src/nervous/`, backend endpoints in `src/hdt/` | Dermatome mapping, peripheral nerve & spinal tract paths, signal propagation animations, backend data services. |

---

## 4. Collaboration & Isolation Guidelines
- **Strict Boundary Rule**: Do not modify files outside your assigned module without consulting the assigned owner.
- **Dependency Rule**: Any new npm dependencies must be documented in the PR / task description.
- **Disclaimer Rule**: The Body Explorer is an educational and research prototype. Never add claims of diagnostic, clinical, or surgical validity.
