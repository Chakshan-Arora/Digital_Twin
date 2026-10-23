# Procedural 3D Body & Skeletal Mesh

## Owner
**Person 3: 3D Body Engineer**

## Scope
- `frontend/src/body/`
- Includes parameter validator: `validateBodyParams.ts` and `validateBodyParams.test.ts`.

## Contract

### Inputs
- `BodyParams` (validated 3D joint positions and body stature in mm).
- Optional visual configuration flags (wireframe, skin opacity, joint markers, skeleton lines).
- Highlighted skin regions / dermatome color maps from `frontend/src/atlas/`.

### Outputs
- Three.js procedural 3D surface mesh representing the human body.
- Joint markers and skeletal bones rendering.
- Raycasting hit events on the 3D skin: emits hit coordinate `[x, y, z]` in mm and surface normal for dermatome querying.
- OrbitControls and viewport synchronization.
