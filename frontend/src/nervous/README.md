# Nervous System Overlay & Pathway Animation

## Owner
**Person 4: Atlas & Integrator**

## Scope
- `frontend/src/nervous/`
- Integration with backend nerve tract routes in `src/hdt/`.

## Contract

### Inputs
- Trigger event from `frontend/src/atlas/` containing the activated dermatome and target spinal root.
- Active `BodyParams` for 3D coordinate registration.

### Outputs
- 3D spline / tube geometries representing peripheral nerves, brachial/lumbosacral plexuses, and the spinal cord ascending to the brainstem/somatosensory cortex.
- Animated electrical action potential / impulse pulse traveling from the skin receptor through the peripheral nerve, into the spinal dorsal root ganglion, up the dorsal column/spinothalamic tract, to the thalamus/cortex.
- HUD telemetry showing pathway details (sensory modality, nerve name, spinal segment, conduction latency estimate).
