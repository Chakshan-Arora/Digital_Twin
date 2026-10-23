# Human Digital Twin (`hdt`)

> **DISCLAIMER: RESEARCH PROTOTYPE ONLY — NOT A CLINICAL TOOL**  
> This software is an experimental research prototype for uncertainty-aware 3D clearance analysis on medical scans. It is strictly not intended for clinical use, diagnosis, treatment planning, surgical navigation, safety assurance, or surgical success prediction. Never claim or rely on this codebase for clinical validity or patient care decisions.

---

## Overview

`hdt` is a Python library and research framework designed to model, represent, and analyze anatomical structures and geometric clearances in 3D medical image space with rigorous handling of spatial orientation, voxel anisotropy, and uncertainty.

### Core Conventions
- **Internal Array Order**: NumPy `(z, y, x)` order for all 3D volumetric data.
- **Physical Spacing & Coordinates**: SimpleITK reports `(x, y, z)` spatial axes; all conversions between SimpleITK `(x, y, z)` and internal `(z, y, x)` order occur exclusively within `src/hdt/imaging/load_scan.py`.
- **Units**: All distances, spacings, origins, and radii are strictly in millimeters (mm).
- **Function Purity**: Functions remain pure without global state or in-place mutations.

---

## Installation & Setup

Requirements:
- Python 3.11+

```bash
# Optional: create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install hdt in editable mode
pip install -e .
```

---

## Running Tests & Linting

Run unit tests via `pytest`:
```bash
pytest
```

Run code formatting and lint checks via `ruff`:
```bash
ruff check .
```
"# syntaxerror" 
"# DigitalTwin" 
