# Human Digital Twin (HDT) — Medical Digital Twin Frontend

> **RESEARCH PROTOTYPE — NOT A CLINICAL TOOL**  
> This demonstration application uses synthetic phantom data and is built strictly for research and paired engineering analysis. It does NOT predict surgical success, clinical safety, or surgical margins.

---

## Architecture Overview

The frontend is a dedicated single-page workstation built with:
- **Framework:** React 19 + TypeScript (strict typing with `noUnusedLocals` and `verbatimModuleSyntax`)
- **Build Tool:** Vite 8 (fast HMR and optimized production bundling)
- **3D Visualization:** [Three.js](https://threejs.org/) (`three` v0.186.x) with `OBJLoader` and `OrbitControls`
- **Charting:** Chart.js + `react-chartjs-2` for Monte Carlo clearance distribution histograms
- **Iconography:** `lucide-react`
- **Test Runner:** Vitest + `@testing-library/react` + `@testing-library/jest-dom` + `jsdom`

---

## 3D Anatomy Viewer & Coordinate Conventions

### Why Three.js?
1. **Direct Coordinate Fidelity:** The Python backend exports authoritative surface meshes via `GET /cases/{id}/structures/{name}/mesh.obj` in standard **RAS physical millimeters** (Right, Anterior, Superior). Three.js handles arbitrary Cartesian coordinates natively, preserving the exact spatial positions and outward triangle winding computed by the backend's marching cubes implementation without risk of hidden orientation flipping.
2. **Feature-Rich Medical Interaction:** Orbit, pan, zoom, reset-view, wireframe toggling, and per-structure opacity controls.
3. **Closest-Points Vector Rendering:** When clearance analysis is executed, Three.js renders 3D spheres at the closest points on the target and risk structures in RAS space, connected by a 3D distance vector.
4. **Orientation Helper:** Features an orientation axes indicator where:
   - **+X (Red):** Right ($R$)
   - **+Y (Green):** Anterior ($A$)
   - **+Z (Blue):** Superior ($S$)

### Volume Rendering Consideration & Roadmap
The backend provides endpoints for streaming raw NIfTI primary volumes (`/volume.nii.gz`) and binary structure masks (`/mask.nii.gz`). In this initial milestone, the viewer renders the backend's authoritative surface meshes in RAS space, and provides direct downloads for volume files. Direct volumetric ray-casting (or orthogonal MPR 2D slice planes) is planned for a subsequent stage; no fake or approximated volume rendering is used.

---

## Setup & Running on Windows

### 1. Prerequisites
- **Node.js:** v20+ (tested on Node v24.16.0)
- **Python:** 3.11 with the backend virtual environment active

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Default configuration:
```ini
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Install Dependencies
From the `frontend/` directory (use `npm.cmd` on Windows if PowerShell script execution policies restrict `npm.ps1`):
```powershell
cd frontend
npm.cmd install
```

### 4. Start the Application
First, ensure the Python backend is running in a terminal:
```powershell
# From the repository root
uvicorn hdt.api.app:app --host 127.0.0.1 --port 8000
```

Next, in a second terminal, start the Vite development server:
```powershell
cd frontend
npm.cmd run dev
```
Open [http://localhost:5173](http://localhost:5173) in your web browser.

---

## Testing & Quality Checks

Run the automated test suite (16 tests covering API client, clearance calculation, uncertainty simulation, and 3D viewer):
```powershell
cd frontend
npm.cmd test
```

Type-check and production build:
```powershell
npm.cmd run build
```

Run linter:
```powershell
npm.cmd run lint
```

---

## Features Implemented
- **Case Discovery & Status:** Automatic health-check, case discovery, and display of spatial metadata (spacing, shape, origin, direction matrix in RAS space).
- **Interactive 3D Mesh Viewer:** Smooth orbit/pan/zoom, structure visibility and opacity sliders, wireframe mode, reset camera, and RAS orientation indicator.
- **NIfTI Downloads:** Direct download links for raw volumetric scans and binary masks.
- **Pairwise Clearance Calculation:** Displays signed face-to-face clearance, Euclidean closest-point distance, closest-point coordinates in RAS space, and surface distance statistics (Min, Mean, HD95, Max).
- **Closest-Points 3D Vector:** Visualizes the closest voxel pair and connecting distance line directly inside the 3D viewer.
- **Monte Carlo Uncertainty Analysis:** Configurable sample count (1–300, capped at 300), amplitude $\sigma$, correlation length $L$, threshold, and seed. Displays baseline clearance, perturbed mean $\pm$ std, percentiles ($p_{05}, p_{50}, p_{95}$), overlap probability, Wilson 95% confidence intervals, elapsed execution timer, and interactive distribution histogram.
- **Safety Disclaimers:** Persistent headers and callouts emphasizing research demonstration purpose.
