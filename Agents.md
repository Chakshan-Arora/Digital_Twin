# Project rules: human-digital-twin (hdt)

## Purpose
Research prototype for uncertainty-aware 3D clearance analysis on medical scans.
NOT a clinical tool. Never write UI text, comments or docs claiming surgical
success prediction, safety or clinical validity.

## Conventions
- Python 3.11, type hints on all public functions, docstrings state units and axis order.
- Arrays are NumPy (z, y, x). Spacing and origin are (z, y, x) in mm.
- SimpleITK (x, y, z) is converted in ONE place: src/hdt/imaging/load_scan.py.
  Nothing else reads the direction matrix or imports SimpleITK. nibabel is tests-only.
- Physical space is LPS (ITK). NIfTI/Slicer/viewers use RAS. Convert only via
  the helpers in load_scan.py.
- All distances are in millimetres. Public functions never take voxel units.
- Clearance convention: FACE-TO-FACE distance between voxel sets
  (hdt.geometry.distances). Positive = gap, 0 = touching, negative = penetration.
  Do not re-implement distance code; call the geometry module.
- Randomness only via a passed-in numpy.random.Generator.
- Functions are pure: no global state, no in-place mutation of inputs.

## DO
- Ship every geometry function with tests on synthetic shapes with analytic answers.
- Handle edge cases explicitly: empty mask, touching or overlapping masks,
  anisotropic voxels, masks touching the array border.
- Run pytest and ruff before declaring a task done.
- Before coding: restate the task, list assumptions, list files you will touch.
- After coding: list what you tested, what you did NOT test, known limitations.
- If a requirement is ambiguous, ask before guessing.

## DON'T
- Don't download data, train models, or add dependencies without asking.
- Don't commit anything under data/ or logs/. Never log or print patient
  identifiers or DICOM headers.
- Don't resample, reorient, crop or normalise images silently.
- Don't use voxel counts as distances. Don't assume isotropic spacing.
- Don't build UI or dashboard code unless the task says so.
- Don't add features beyond the task. List ideas at the end instead.

## Test integrity
- Never edit an existing test's assertion, tolerance or expected value. If a test
  fails after a change, STOP: report the test name, old and new expected values,
  the measured value and your proposed fix. A human approves test changes.
  Adding new tests is always allowed.
- If a test was changed with approval, list it in the report with a before/after diff.
- Never retype or summarise test output. Run:
    pytest -v | Tee-Object -FilePath logs\pytest_last.txt
  and tell the human to open that file.
- Never state that a milestone is complete. List the manual checks a human must run.
- Commit after each approved prompt, with the prompt number in the message.