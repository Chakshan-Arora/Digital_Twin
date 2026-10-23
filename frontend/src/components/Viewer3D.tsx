import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { Crosshair, Eye, EyeOff, RotateCcw, ShieldAlert, Sparkles, Volume2 } from 'lucide-react';
import type { CaseDetail, ClosestPointsRAS } from '../types/api';
import { apiClient } from '../services/apiClient';

interface Viewer3DProps {
  caseDetail: CaseDetail | null;
  closestPoints: ClosestPointsRAS | null;
  synchronizedPointRas?: [number, number, number] | null;
}

interface StructureMeshState {
  name: string;
  role: string;
  color: string;
  visible: boolean;
  opacity: number;
  loading: boolean;
  error: string | null;
  meshObject: THREE.Group | null;
}

const STRUCTURE_COLOR_PALETTE: Record<string, string> = {
  tumor: '#e11d48', // Crimson/coral for target
  vessel: '#0284c7', // Sky blue for risk
  target: '#e11d48',
  risk: '#0284c7',
  context: '#10b981',
};

export const Viewer3D: React.FC<Viewer3DProps> = ({
  caseDetail,
  closestPoints,
  synchronizedPointRas,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshesGroupRef = useRef<THREE.Group | null>(null);
  const closestPointsGroupRef = useRef<THREE.Group | null>(null);
  const syncCrosshairGroupRef = useRef<THREE.Group | null>(null);
  const defaultTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const defaultPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, -100, 50));

  const [structuresState, setStructuresState] = useState<StructureMeshState[]>([]);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const wireframeRef = useRef<boolean>(wireframe);
  useEffect(() => {
    wireframeRef.current = wireframe;
  }, [wireframe]);
  const [showAxes, setShowAxes] = useState<boolean>(true);
  const [showClosestPoints, setShowClosestPoints] = useState<boolean>(true);
  const [showSyncCrosshair, setShowSyncCrosshair] = useState<boolean>(true);
  const [loadingMeshes, setLoadingMeshes] = useState<boolean>(false);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 500;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0f17); // Dark medical workstation canvas
    sceneRef.current = scene;

    // Camera: Set up-vector to +Z (Superior in RAS space)
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -100, 50);
    cameraRef.current = camera;

    // Renderer with antialiasing
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      // Graceful fallback if WebGL is unavailable or in mock environment
      return;
    }
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true;
    controlsRef.current = controls;

    // Lighting (ambient + directional front, back, key)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(100, -100, 100);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa5b4fc, 0.5);
    dirLight2.position.set(-100, 100, -50);
    scene.add(dirLight2);

    // Groups
    const meshesGroup = new THREE.Group();
    meshesGroup.name = 'structures_meshes';
    scene.add(meshesGroup);
    meshesGroupRef.current = meshesGroup;

    const cpGroup = new THREE.Group();
    cpGroup.name = 'closest_points_visualizer';
    scene.add(cpGroup);
    closestPointsGroupRef.current = cpGroup;

    const syncGroup = new THREE.Group();
    syncGroup.name = 'sync_crosshair_group';
    scene.add(syncGroup);
    syncCrosshairGroupRef.current = syncGroup;

    // AxesHelper (RAS space: R=X red, A=Y green, S=Z blue)
    const axesHelper = new THREE.AxesHelper(30);
    axesHelper.name = 'axes_helper';
    scene.add(axesHelper);

    // Grid on axial plane (XY plane at Z=0 or local origin)
    const grid = new THREE.GridHelper(100, 20, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2; // Orient grid onto XY plane (Z up)
    grid.position.set(0, 0, 0);
    grid.name = 'spatial_grid';
    scene.add(grid);

    // Animation Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Observer
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      if (newW === 0 || newH === 0) return;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH, false);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Load structure meshes when caseDetail changes
  useEffect(() => {
    if (!caseDetail) {
      setStructuresState([]);
      return;
    }

    const initialStates: StructureMeshState[] = caseDetail.structures.map((s) => ({
      name: s.name,
      role: s.role,
      color: STRUCTURE_COLOR_PALETTE[s.name] || STRUCTURE_COLOR_PALETTE[s.role] || '#94a3b8',
      visible: true,
      opacity: 0.85,
      loading: !!sceneRef.current,
      error: null,
      meshObject: null,
    }));
    setStructuresState(initialStates);

    if (!sceneRef.current || !meshesGroupRef.current) return;

    const meshesGroup = meshesGroupRef.current;
    // Clear old meshes
    while (meshesGroup.children.length > 0) {
      const child = meshesGroup.children[0];
      meshesGroup.remove(child);
      if ('geometry' in child && child.geometry instanceof THREE.BufferGeometry) {
        child.geometry.dispose();
      }
    }

    const loader = new OBJLoader();
    let isCancelled = false;
    setLoadingMeshes(true);

    const loadAllMeshes = async () => {
      const updatedStates = [...initialStates];
      const bbox = new THREE.Box3();

      for (let i = 0; i < caseDetail.structures.length; i++) {
        const s = caseDetail.structures[i];
        try {
          const objText = await apiClient.getMeshObj(caseDetail.id, s.name);
          if (isCancelled) return;

          const obj = loader.parse(objText);
          const colorHex = updatedStates[i].color;

          // Apply medical workstation Phong material with smooth shading
          obj.traverse((child) => {
            if (child instanceof THREE.Mesh) {
              child.geometry.computeVertexNormals();
              child.material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(colorHex),
                transparent: true,
                opacity: 0.85,
                shininess: 40,
                specular: 0x222222,
                side: THREE.DoubleSide,
                wireframe: wireframeRef.current,
              });
            }
          });

          obj.name = `mesh_${s.name}`;
          meshesGroup.add(obj);

          // Update bounding box
          bbox.expandByObject(obj);

          updatedStates[i].meshObject = obj;
          updatedStates[i].loading = false;
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Mesh load error';
          updatedStates[i].loading = false;
          updatedStates[i].error = msg;
        }
      }

      if (!isCancelled) {
        setStructuresState([...updatedStates]);
        setLoadingMeshes(false);

        // Center camera and controls target on the anatomical center of structures in RAS mm
        if (!bbox.isEmpty() && cameraRef.current && controlsRef.current) {
          const center = new THREE.Vector3();
          bbox.getCenter(center);
          const size = new THREE.Vector3();
          bbox.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z, 20);

          defaultTargetRef.current.copy(center);
          defaultPosRef.current.set(center.x, center.y - maxDim * 3, center.z + maxDim * 1.5);

          controlsRef.current.target.copy(center);
          cameraRef.current.position.copy(defaultPosRef.current);
          cameraRef.current.lookAt(center);
          controlsRef.current.update();
        }
      }
    };

    loadAllMeshes();

    return () => {
      isCancelled = true;
    };
  }, [caseDetail]);

  // Update wireframe mode
  useEffect(() => {
    if (!meshesGroupRef.current) return;
    meshesGroupRef.current.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshPhongMaterial) {
        child.material.wireframe = wireframe;
      }
    });
  }, [wireframe]);

  // Update axes visibility
  useEffect(() => {
    if (!sceneRef.current) return;
    const axes = sceneRef.current.getObjectByName('axes_helper');
    if (axes) axes.visible = showAxes;
    const grid = sceneRef.current.getObjectByName('spatial_grid');
    if (grid) grid.visible = showAxes;
  }, [showAxes]);

  // Visualize closest points and connecting distance line in RAS coordinates
  useEffect(() => {
    if (!closestPointsGroupRef.current) return;
    const cpGroup = closestPointsGroupRef.current;

    while (cpGroup.children.length > 0) {
      const child = cpGroup.children[0];
      cpGroup.remove(child);
      if ('geometry' in child && child.geometry instanceof THREE.BufferGeometry) {
        child.geometry.dispose();
      }
    }

    if (!closestPoints || !showClosestPoints) return;

    const [tx, ty, tz] = closestPoints.point_target_ras_mm;
    const [rx, ry, rz] = closestPoints.point_risk_ras_mm;

    const pTarget = new THREE.Vector3(tx, ty, tz);
    const pRisk = new THREE.Vector3(rx, ry, rz);

    // Glowing spheres at closest points (radius 0.65 mm)
    const sphereGeo = new THREE.SphereGeometry(0.65, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 }); // Bright gold

    const sphereTarget = new THREE.Mesh(sphereGeo, sphereMat);
    sphereTarget.position.copy(pTarget);
    cpGroup.add(sphereTarget);

    const sphereRisk = new THREE.Mesh(sphereGeo, sphereMat);
    sphereRisk.position.copy(pRisk);
    cpGroup.add(sphereRisk);

    // Connecting line
    const linePoints = [pTarget, pRisk];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xfbbf24,
      linewidth: 3,
    });
    const line = new THREE.Line(lineGeo, lineMat);
    cpGroup.add(line);
  }, [closestPoints, showClosestPoints]);

  // Synchronized 3D crosshair cursor representing the 2D slice focal point in RAS space
  useEffect(() => {
    if (!syncCrosshairGroupRef.current) return;
    const group = syncCrosshairGroupRef.current;

    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if ('geometry' in child && child.geometry instanceof THREE.BufferGeometry) {
        child.geometry.dispose();
      }
    }

    if (!synchronizedPointRas || !showSyncCrosshair) return;

    const [px, py, pz] = synchronizedPointRas;
    const center = new THREE.Vector3(px, py, pz);

    // Glowing reticle central sphere
    const sphereGeo = new THREE.SphereGeometry(0.75, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 }); // Bright sky-blue
    const marker = new THREE.Mesh(sphereGeo, sphereMat);
    marker.position.copy(center);
    group.add(marker);

    // 3 orthogonal crosshair lines (length 14 mm) in RAS directions
    const armLen = 7;
    // R (X axis) arm - red
    const rGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(px - armLen, py, pz),
      new THREE.Vector3(px + armLen, py, pz),
    ]);
    const rLine = new THREE.Line(rGeo, new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 }));
    group.add(rLine);

    // A (Y axis) arm - green
    const aGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(px, py - armLen, pz),
      new THREE.Vector3(px, py + armLen, pz),
    ]);
    const aLine = new THREE.Line(aGeo, new THREE.LineBasicMaterial({ color: 0x22c55e, linewidth: 2 }));
    group.add(aLine);

    // S (Z axis) arm - blue
    const sGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(px, py, pz - armLen),
      new THREE.Vector3(px, py, pz + armLen),
    ]);
    const sLine = new THREE.Line(sGeo, new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 2 }));
    group.add(sLine);
  }, [synchronizedPointRas, showSyncCrosshair]);

  const toggleVisibility = (idx: number) => {
    const updated = [...structuresState];
    const s = updated[idx];
    s.visible = !s.visible;
    if (s.meshObject) {
      s.meshObject.visible = s.visible;
    }
    setStructuresState(updated);
  };

  const handleOpacityChange = (idx: number, opacity: number) => {
    const updated = [...structuresState];
    const s = updated[idx];
    s.opacity = opacity;
    if (s.meshObject) {
      s.meshObject.traverse((child) => {
        if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshPhongMaterial) {
          child.material.opacity = opacity;
        }
      });
    }
    setStructuresState(updated);
  };

  const resetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.copy(defaultPosRef.current);
    controlsRef.current.target.copy(defaultTargetRef.current);
    controlsRef.current.update();
  };

  return (
    <div className="card viewer-card">
      <div className="viewer-header">
        <div className="viewer-title">
          <Volume2 size={18} />
          <h2>3D Anatomy Viewer (RAS Space)</h2>
          <span className="badge badge-subtle">Three.js WebGL</span>
        </div>

        <div className="viewer-toolbar">
          <button
            className={`btn btn-toolbar ${showAxes ? 'active' : ''}`}
            onClick={() => setShowAxes(!showAxes)}
            title="Toggle RAS coordinate axes & grid"
          >
            <span>Axes</span>
          </button>

          <button
            className={`btn btn-toolbar ${wireframe ? 'active' : ''}`}
            onClick={() => setWireframe(!wireframe)}
            title="Toggle wireframe mode"
          >
            <span>Wireframe</span>
          </button>

          {closestPoints && (
            <button
              className={`btn btn-toolbar ${showClosestPoints ? 'active' : ''}`}
              onClick={() => setShowClosestPoints(!showClosestPoints)}
              title="Toggle closest points vector"
            >
              <Sparkles size={13} />
              <span>Closest Pair</span>
            </button>
          )}

          {synchronizedPointRas && (
            <button
              className={`btn btn-toolbar ${showSyncCrosshair ? 'active' : ''}`}
              onClick={() => setShowSyncCrosshair(!showSyncCrosshair)}
              title="Toggle synchronized 2D slice crosshair cursor in 3D"
            >
              <Crosshair size={13} />
              <span>Slice Crosshair</span>
            </button>
          )}

          <button
            className="btn btn-toolbar"
            onClick={resetCamera}
            title="Reset camera viewpoint"
          >
            <RotateCcw size={13} />
            <span>Reset View</span>
          </button>
        </div>
      </div>

      <div className="viewer-container">
        {/* WebGL Canvas Container */}
        <div ref={mountRef} className="webgl-canvas-mount" />

        {/* Loading overlay */}
        {loadingMeshes && (
          <div className="viewer-overlay-loading">
            <span className="spinner"></span>
            <span>Parsing & Rendering Surface Meshes...</span>
          </div>
        )}

        {/* Watermark & Safety Disclaimer */}
        <div className="viewer-watermark">
          <div className="watermark-pill">
            <ShieldAlert size={12} />
            <span>SYNTHETIC PHANTOM — DEMO ONLY</span>
          </div>
        </div>

        {/* Coordinate Space Legend */}
        <div className="viewer-legend">
          <span className="legend-title">RAS mm Axes:</span>
          <div className="legend-items">
            <span className="legend-item axis-r">
              <span className="legend-color-dot" style={{ background: '#ef4444' }}></span>
              +X: Right (R)
            </span>
            <span className="legend-item axis-a">
              <span className="legend-color-dot" style={{ background: '#22c55e' }}></span>
              +Y: Anterior (A)
            </span>
            <span className="legend-item axis-s">
              <span className="legend-color-dot" style={{ background: '#3b82f6' }}></span>
              +Z: Superior (S)
            </span>
          </div>
        </div>
      </div>

      {/* Structure Controls Bar */}
      <div className="viewer-controls-bar">
        <span className="controls-bar-title">Structure Controls:</span>
        <div className="structure-toggles">
          {structuresState.map((s, idx) => (
            <div key={s.name} className="structure-toggle-item">
              <button
                className={`btn-toggle-eye ${s.visible ? 'eye-visible' : 'eye-hidden'}`}
                onClick={() => toggleVisibility(idx)}
                title={s.visible ? `Hide ${s.name}` : `Show ${s.name}`}
              >
                {s.visible ? <Eye size={14} /> : <EyeOff size={14} />}
              </button>
              <span
                className="structure-color-badge"
                style={{ backgroundColor: s.color }}
              />
              <span className="structure-name-text">{s.name}</span>
              <span className="structure-role-tag font-mono">({s.role})</span>

              <div className="opacity-control" title="Adjust mesh opacity">
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={s.opacity}
                  onChange={(e) => handleOpacityChange(idx, parseFloat(e.target.value))}
                  disabled={!s.visible}
                />
                <span className="opacity-val">{Math.round(s.opacity * 100)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
