# Browser Camera Body Scanner

## Owner
**Person 2: Scanner Engineer**

## Scope
- `frontend/src/scanner/`

## Contract

### Inputs
- Browser `MediaStream` (live webcam video stream or uploaded video/frame).
- Calibration parameters (optional user height prompt in mm or standard reference height).

### Outputs
- Validated `BodyParams` matching the specification in `frontend/src/types/body.ts`.
- Landmark tracking confidence scores and camera status callbacks (`'idle' | 'scanning' | 'calibrated' | 'error'`).
- Must pass `validateBodyParams(bodyParams).ok === true` before dispatching to the 3D body viewer.
