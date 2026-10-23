# Anatomical Dermatome Atlas

## Owner
**Person 4: Atlas & Integrator**

## Scope
- `frontend/src/atlas/`
- Integration with backend atlas routes in `src/hdt/`.

## Contract

### Inputs
- Click-on-skin 3D hit point `[x, y, z]` in mm from `frontend/src/body/`.
- Active `BodyParams` to scale dermatome boundary lookup to patient stature.

### Outputs
- Identified dermatome region (e.g. `C6`, `L4`, `S1`, etc.) and anatomical description.
- Corresponding spinal nerve root origin (`C1`-`C8`, `T1`-`T12`, `L1`-`L5`, `S1`-`S5`).
- Texture / vertex color map passed to `frontend/src/body/` to illuminate the clicked skin dermatome.
- Signal trigger dispatched to `frontend/src/nervous/` to animate nerve pathway transmission.
