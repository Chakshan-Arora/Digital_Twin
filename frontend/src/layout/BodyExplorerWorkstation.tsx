import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import {
  Activity,
  AlertCircle,
  Bone,
  CheckCircle2,
  Crosshair,
  Heart,
  Layers,
  Maximize2,
  RotateCcw,
  Ruler,
  ShieldAlert,
  Sparkles,
  User,
  ZoomIn,
} from 'lucide-react';
import { type BodyParams } from '../types/body';
import { BODY_PRESETS, getValidatedPreset } from './presets';
import { createHologramFresnelMaterial, createMarginShellMaterial } from './hologramShaders';
import { createMriSliceTexture, type SliceOrientation } from './mriSliceTexture';
import { apiClient } from '../services/apiClient';
import type { SystemItem } from '../types/api';

export type CameraViewPreset = 'whole-body' | 'cranial-tumor';

/**
 * Creates a Three.js mesh or group from a procedural SystemItem.
 * Coordinates in HDT are RAS mm:
 *   rx: right (+x) / left (-x)
 *   ry: anterior (+y) / posterior (-y)
 *   rz: superior (+z) / inferior (-z)
 * In Three.js:
 *   three.x = rx (lateral)
 *   three.y = rz (height / superior)
 *   three.z = ry (depth / anterior)
 */
function createSystemItemMesh(item: SystemItem): THREE.Object3D | null {
  const { primitive, color_hint } = item;
  if (!primitive) return null;

  const hexColor = color_hint ? parseInt(color_hint.replace('#', '0x'), 16) : 0xe8e0d0;

  if (primitive.kind === 'tube' && primitive.polyline && primitive.polyline.length >= 2) {
    const pts = primitive.polyline.map(([rx, ry, rz]) => new THREE.Vector3(rx, rz, ry));

    const isArtery = item.system.includes('arterial');
    const isVein = item.system.includes('venous');
    const isCartilage = item.color_hint === '#D0E8D0';

    const mat = new THREE.MeshStandardMaterial({
      color: hexColor,
      roughness: isArtery || isVein ? 0.25 : 0.45,
      metalness: isArtery || isVein ? 0.2 : 0.1,
      emissive: isArtery ? 0x660000 : isVein ? 0x001855 : 0x000000,
      emissiveIntensity: isArtery || isVein ? 0.4 : 0,
      transparent: isCartilage,
      opacity: isCartilage ? 0.85 : 1.0,
    });

    if (pts.length === 2) {
      const p0 = pts[0];
      const p1 = pts[1];
      const dist = p0.distanceTo(p1);
      if (dist < 0.1) return null;

      const rBottom = primitive.radius_mm ?? 10;
      const rTop = primitive.radius_end_mm ?? rBottom;
      const geo = new THREE.CylinderGeometry(rTop, rBottom, dist, 12);
      const mesh = new THREE.Mesh(geo, mat);

      const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
      mesh.position.copy(mid);

      const dir = new THREE.Vector3().subVectors(p1, p0).normalize();
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      mesh.name = item.id;
      return mesh;
    } else {
      const curve = new THREE.CatmullRomCurve3(pts);
      const radius = primitive.radius_mm ?? 6;
      const geo = new THREE.TubeGeometry(curve, 24, radius, 8, false);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = item.id;
      return mesh;
    }
  }

  if (primitive.kind === 'ellipsoid' && primitive.center && primitive.radii_mm) {
    const [cx, cy, cz] = primitive.center;
    const [rx, ry, rz] = primitive.radii_mm;
    const geo = new THREE.SphereGeometry(1, 18, 14);
    // In Three.js: scale X=rx, Y=rz (superior), Z=ry (anterior)
    geo.scale(rx, rz, ry);

    const isOrgan = item.system === 'organs';
    const mat = new THREE.MeshStandardMaterial({
      color: hexColor,
      roughness: isOrgan ? 0.35 : 0.45,
      metalness: isOrgan ? 0.15 : 0.1,
      transparent: isOrgan,
      opacity: isOrgan ? 0.88 : 1.0,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(cx, cz, cy);

    if (primitive.rotation_deg) {
      const [rotX, rotY, rotZ] = primitive.rotation_deg;
      mesh.rotation.set(
        (rotX * Math.PI) / 180,
        (rotZ * Math.PI) / 180,
        (rotY * Math.PI) / 180
      );
    }
    mesh.name = item.id;
    return mesh;
  }

  if (primitive.kind === 'mesh_hint' && primitive.bounding_ellipsoid_center && primitive.bounding_ellipsoid_radii_mm) {
    const [cx, cy, cz] = primitive.bounding_ellipsoid_center;
    const [rx, ry, rz] = primitive.bounding_ellipsoid_radii_mm;
    const geo = new THREE.SphereGeometry(1, 24, 18);
    geo.scale(rx, rz, ry);
    const mat = new THREE.MeshStandardMaterial({
      color: hexColor,
      roughness: 0.45,
      metalness: 0.1,
      transparent: true,
      opacity: 0.8,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(cx, cz, cy);
    mesh.name = item.id;
    return mesh;
  }

  return null;
}

/** Disposes all mesh geometries and materials within a group */
function clearThreeGroup(group: THREE.Group | null) {
  if (!group) return;
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    if (child instanceof THREE.Mesh) {
      if (child.geometry) child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else if (child.material) {
        child.material.dispose();
      }
    }
  }
}

export const BodyExplorerWorkstation: React.FC = () => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('standard-1750');
  const [heightMm, setHeightMm] = useState<number>(1750);
  const [bodyParams, setBodyParams] = useState<BodyParams>(() => getValidatedPreset('standard-1750').params);
  const [validation, setValidation] = useState(() => getValidatedPreset('standard-1750').validation);

  // Anatomical Systems Toggles
  const [showSkeletonSystem, setShowSkeletonSystem] = useState<boolean>(true);
  const [showVascularSystem, setShowVascularSystem] = useState<boolean>(true);
  const [showOrgansSystem, setShowOrgansSystem] = useState<boolean>(true);
  const [showHologramHull, setShowHologramHull] = useState<boolean>(false);
  const [hullOpacity] = useState<number>(0.12);
  const [systemsLoading, setSystemsLoading] = useState<boolean>(false);
  const [itemCounts, setItemCounts] = useState<{ skeleton: number; vascular: number; organs: number }>({
    skeleton: 69,
    vascular: 36,
    organs: 16,
  });

  // Tumor & Margin Visualization Controls
  const [marginMm, setMarginMm] = useState<number>(3.0);
  const [showMarginShell] = useState<boolean>(true);

  // In-Situ Radiology Cut-Plane Controls
  const [showCutPlane, setShowCutPlane] = useState<boolean>(true);
  const [sliceOrientation, setSliceOrientation] = useState<SliceOrientation>('axial');
  const [cutPlaneOpacity, setCutPlaneOpacity] = useState<number>(0.75);

  const [cameraView, setCameraView] = useState<CameraViewPreset>('whole-body');

  // Tumor clearance telemetry
  const tumorVolumeMm3 = 885.0; // From tumor_mesh.obj header
  const measuredClearanceMm = 4.8; // Distance from cranial tumor to adjacent carotid risk vessel

  // Three.js Refs
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // Groups & Meshes
  const skeletonSystemGroupRef = useRef<THREE.Group | null>(null);
  const vascularSystemGroupRef = useRef<THREE.Group | null>(null);
  const organsSystemGroupRef = useRef<THREE.Group | null>(null);
  const hologramHullGroupRef = useRef<THREE.Group | null>(null);
  const pathologyGroupRef = useRef<THREE.Group | null>(null);
  const marginShellMeshRef = useRef<THREE.Mesh | null>(null);
  const cutPlaneMeshRef = useRef<THREE.Mesh | null>(null);
  const rawTumorMeshRef = useRef<THREE.BufferGeometry | null>(null);

  // Handle Preset or Stature change
  const handlePresetSelect = (presetId: string) => {
    setSelectedPresetId(presetId);
    const preset = BODY_PRESETS.find((p) => p.id === presetId) || BODY_PRESETS[0];
    setHeightMm(preset.baseHeightMm);
    const result = getValidatedPreset(presetId, preset.baseHeightMm);
    setBodyParams(result.params);
    setValidation(result.validation);
  };

  const handleHeightChange = (newHeight: number) => {
    setHeightMm(newHeight);
    const result = getValidatedPreset(selectedPresetId, newHeight);
    setBodyParams(result.params);
    setValidation(result.validation);
  };

  // Camera transition helper
  const transitionCamera = (view: CameraViewPreset, headPos: [number, number, number]) => {
    if (!cameraRef.current || !controlsRef.current) return;
    setCameraView(view);
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    if (view === 'whole-body') {
      camera.position.set(0, 1000, 2600);
      controls.target.set(0, 950, 0);
    } else {
      // Zoom into cranial cavity around head
      const [hx, hy, hz] = headPos;
      camera.position.set(hx - 25, hy - 20, hz + 220);
      controls.target.set(hx - 25, hy - 40, hz + 15);
    }
    controls.update();
  };

  // 1. Initialize Three.js viewport
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b12);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 5, 12000);
    camera.position.set(0, 1000, 2600);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 950, 0);
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xe0f2fe, 1.3);
    dirLight1.position.set(800, 2400, 1800);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf43f5e, 0.7);
    dirLight2.position.set(-800, 1200, -800);
    scene.add(dirLight2);

    const floorLight = new THREE.DirectionalLight(0x38bdf8, 0.4);
    floorLight.position.set(0, -500, 500);
    scene.add(floorLight);

    // Floor Grid
    const grid = new THREE.GridHelper(3000, 30, 0x1e293b, 0x0f172a);
    grid.position.y = 0;
    scene.add(grid);

    // Node groups for Body Systems
    const skeletonSysGroup = new THREE.Group();
    const vascularSysGroup = new THREE.Group();
    const organsSysGroup = new THREE.Group();
    const hologramHullGroup = new THREE.Group();
    const pathologyGroup = new THREE.Group();

    scene.add(skeletonSysGroup);
    scene.add(vascularSysGroup);
    scene.add(organsSysGroup);
    scene.add(hologramHullGroup);
    scene.add(pathologyGroup);

    skeletonSystemGroupRef.current = skeletonSysGroup;
    vascularSystemGroupRef.current = vascularSysGroup;
    organsSystemGroupRef.current = organsSysGroup;
    hologramHullGroupRef.current = hologramHullGroup;
    pathologyGroupRef.current = pathologyGroup;

    // Load actual cranial tumor mesh (.obj)
    const loader = new OBJLoader();
    loader.load(
      '/tumor_mesh.obj',
      (obj) => {
        let tumorGeo: THREE.BufferGeometry | null = null;
        obj.traverse((child) => {
          if (child instanceof THREE.Mesh && !tumorGeo) {
            const cloned = child.geometry.clone();
            cloned.center();
            cloned.computeVertexNormals();
            tumorGeo = cloned;
          }
        });

        if (tumorGeo) {
          rawTumorMeshRef.current = tumorGeo;
          buildPathologyScene(tumorGeo);
        }
      },
      undefined,
      (err) => {
        console.warn('Fallback tumor geometry used:', err);
        const fallbackGeo = new THREE.DodecahedronGeometry(8, 2);
        rawTumorMeshRef.current = fallbackGeo;
        buildPathologyScene(fallbackGeo);
      }
    );

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // 2. Fetch and render procedural body systems (Skeleton, Vascular, Organs)
  useEffect(() => {
    let isCancelled = false;

    async function loadSystems() {
      const skelGroup = skeletonSystemGroupRef.current;
      const vascGroup = vascularSystemGroupRef.current;
      const orgGroup = organsSystemGroupRef.current;
      if (!skelGroup || !vascGroup || !orgGroup) return;

      setSystemsLoading(true);

      // Convert frontend bodyParams to backend RAS convention: [x, z_anterior, y_superior]
      const rasJoints: Record<string, [number, number, number]> = {};
      for (const [k, [x, y, z]] of Object.entries(bodyParams.joints)) {
        rasJoints[k] = [x, z, y];
      }
      const rasBodyParams = {
        height_mm: bodyParams.height_mm,
        source: bodyParams.source,
        joints: rasJoints,
      };

      try {
        const [skelRes, vascRes, orgRes] = await Promise.all([
          apiClient.getBodySystemItems('skeleton', rasBodyParams),
          apiClient.getBodySystemItems('vascular', rasBodyParams),
          apiClient.getBodySystemItems('organs', rasBodyParams),
        ]);

        if (isCancelled) return;

        // Clear previous meshes
        clearThreeGroup(skelGroup);
        clearThreeGroup(vascGroup);
        clearThreeGroup(orgGroup);

        // Populate Skeleton
        if (skelRes?.items) {
          for (const item of skelRes.items) {
            const mesh = createSystemItemMesh(item);
            if (mesh) skelGroup.add(mesh);
          }
        }

        // Populate Vascular
        if (vascRes?.items) {
          for (const item of vascRes.items) {
            const mesh = createSystemItemMesh(item);
            if (mesh) vascGroup.add(mesh);
          }
        }

        // Populate Organs
        if (orgRes?.items) {
          for (const item of orgRes.items) {
            const mesh = createSystemItemMesh(item);
            if (mesh) orgGroup.add(mesh);
          }
        }

        setItemCounts({
          skeleton: skelRes?.item_count || skelGroup.children.length,
          vascular: vascRes?.item_count || vascGroup.children.length,
          organs: orgRes?.item_count || orgGroup.children.length,
        });
      } catch (err) {
        console.warn('Backend body systems fetch failed, checking local state:', err);
      } finally {
        if (!isCancelled) setSystemsLoading(false);
      }
    }

    loadSystems();

    return () => {
      isCancelled = true;
    };
  }, [bodyParams]);

  // Toggle Visibility of System Groups
  useEffect(() => {
    if (skeletonSystemGroupRef.current) {
      skeletonSystemGroupRef.current.visible = showSkeletonSystem;
    }
  }, [showSkeletonSystem]);

  useEffect(() => {
    if (vascularSystemGroupRef.current) {
      vascularSystemGroupRef.current.visible = showVascularSystem;
    }
  }, [showVascularSystem]);

  useEffect(() => {
    if (organsSystemGroupRef.current) {
      organsSystemGroupRef.current.visible = showOrgansSystem;
    }
  }, [showOrgansSystem]);

  // 3. Render Subtle Hologram Glass Hull
  useEffect(() => {
    const hullGroup = hologramHullGroupRef.current;
    if (!hullGroup) return;

    clearThreeGroup(hullGroup);
    hullGroup.visible = showHologramHull;

    if (!showHologramHull) return;

    const { joints } = bodyParams;
    const fresnelMat = createHologramFresnelMaterial(0x38bdf8, 3.2);

    // Subtle cranial glass sphere
    const headPt = joints.head;
    const headGeo = new THREE.SphereGeometry(105, 24, 24);
    headGeo.scale(1.0, 1.25, 1.15);
    const headMesh = new THREE.Mesh(headGeo, fresnelMat);
    headMesh.position.set(headPt[0], headPt[1] - 30, headPt[2]);
    hullGroup.add(headMesh);

    // Torso Glass Silhouette
    const chestPt = joints.chest;
    const pelvisPt = joints.pelvis;
    const torsoHeight = chestPt[1] - pelvisPt[1] + 160;
    const torsoGeo = new THREE.CylinderGeometry(155, 140, torsoHeight, 20);
    torsoGeo.scale(1.15, 1.0, 0.7);
    const torsoMesh = new THREE.Mesh(torsoGeo, fresnelMat);
    torsoMesh.position.set(0, (chestPt[1] + pelvisPt[1]) / 2 + 10, 0);
    hullGroup.add(torsoMesh);
  }, [bodyParams, showHologramHull, hullOpacity]);

  // 4. Build Pathology Scene (Cranial Tumor Mesh, Risk Vessel, Margin Shell & In-Situ Cut-Plane)
  const buildPathologyScene = (baseGeo: THREE.BufferGeometry) => {
    const group = pathologyGroupRef.current;
    if (!group) return;

    clearThreeGroup(group);

    const headJoint = bodyParams.joints.head;
    const tumorCenter = new THREE.Vector3(headJoint[0] - 25, headJoint[1] - 40, headJoint[2] + 15);

    // 1. Core Tumor Surface Mesh (crimson #e11d48)
    const tumorMat = new THREE.MeshStandardMaterial({
      color: 0xe11d48,
      emissive: 0x9f1239,
      emissiveIntensity: 0.5,
      roughness: 0.25,
      metalness: 0.15,
      side: THREE.DoubleSide,
    });
    const tumorMesh = new THREE.Mesh(baseGeo, tumorMat);
    tumorMesh.position.copy(tumorCenter);
    group.add(tumorMesh);

    // 2. Translucent Margin Expansion Shell
    const shellMat = createMarginShellMaterial(0xfbbf24, 0.32);
    const shellGeo = baseGeo.clone();
    const shellMesh = new THREE.Mesh(shellGeo, shellMat);
    shellMesh.position.copy(tumorCenter);
    group.add(shellMesh);
    marginShellMeshRef.current = shellMesh;
    updateMarginShellGeometry(baseGeo, shellMesh, marginMm);

    // 3. In-Situ Radiology Cut-Plane (Pinned strictly at tumorCenter)
    const planeGeo = new THREE.PlaneGeometry(165, 165);
    const sliceTex = createMriSliceTexture(sliceOrientation);
    const planeMat = new THREE.MeshBasicMaterial({
      map: sliceTex,
      transparent: true,
      opacity: cutPlaneOpacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const cutPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
    cutPlaneMesh.position.copy(tumorCenter);
    applyPlaneOrientation(cutPlaneMesh, sliceOrientation);
    cutPlaneMesh.visible = showCutPlane;
    group.add(cutPlaneMesh);
    cutPlaneMeshRef.current = cutPlaneMesh;

    // 4. Adjacent Critical Risk Structure (Carotid/Cranial Vessel in Sky Blue #0284c7)
    const vesselX = tumorCenter.x - 11.3;
    const vesselCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(vesselX, tumorCenter.y - 30, tumorCenter.z - 8),
      new THREE.Vector3(vesselX, tumorCenter.y, tumorCenter.z),
      new THREE.Vector3(vesselX, tumorCenter.y + 30, tumorCenter.z + 8),
    ]);
    const vesselGeo = new THREE.TubeGeometry(vesselCurve, 24, 2.5, 12, false);
    const vesselMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      emissive: 0x0369a1,
      emissiveIntensity: 0.4,
      roughness: 0.3,
      metalness: 0.2,
    });
    const vesselMesh = new THREE.Mesh(vesselGeo, vesselMat);
    group.add(vesselMesh);

    // 5. Closest-Points Distance Vector
    const ptTumor = new THREE.Vector3(tumorCenter.x - 6.5, tumorCenter.y, tumorCenter.z);
    const ptVessel = new THREE.Vector3(tumorCenter.x - 11.3, tumorCenter.y, tumorCenter.z);

    const vectorGeo = new THREE.BufferGeometry().setFromPoints([ptTumor, ptVessel]);
    const vectorMat = new THREE.LineDashedMaterial({
      color: 0xfacc15,
      dashSize: 1.5,
      gapSize: 0.8,
      linewidth: 2,
    });
    const line = new THREE.Line(vectorGeo, vectorMat);
    line.computeLineDistances();
    group.add(line);

    // Endpoint Spheres
    const dotGeo = new THREE.SphereGeometry(1.6, 12, 12);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    const dot1 = new THREE.Mesh(dotGeo, dotMat);
    dot1.position.copy(ptTumor);
    const dot2 = new THREE.Mesh(dotGeo, dotMat);
    dot2.position.copy(ptVessel);
    group.add(dot1);
    group.add(dot2);
  };

  // Helper to rotate cut plane according to orthogonal orientation
  const applyPlaneOrientation = (plane: THREE.Mesh, orientation: SliceOrientation) => {
    if (orientation === 'axial') {
      plane.rotation.set(-Math.PI / 2, 0, 0);
    } else if (orientation === 'coronal') {
      plane.rotation.set(0, 0, 0);
    } else {
      plane.rotation.set(0, Math.PI / 2, 0);
    }
  };

  // Update In-Situ Cut-Plane when orientation, opacity, or visibility change
  useEffect(() => {
    const plane = cutPlaneMeshRef.current;
    if (!plane) return;

    plane.visible = showCutPlane;

    if (plane.material instanceof THREE.MeshBasicMaterial) {
      plane.material.opacity = cutPlaneOpacity;

      if (plane.material.map) {
        plane.material.map.dispose();
      }
      const newTex = createMriSliceTexture(sliceOrientation);
      plane.material.map = newTex;
      plane.material.needsUpdate = true;
    }

    applyPlaneOrientation(plane, sliceOrientation);
  }, [sliceOrientation, cutPlaneOpacity, showCutPlane]);

  // Expand Margin Shell vertices along their normals by marginMm
  const updateMarginShellGeometry = (baseGeo: THREE.BufferGeometry, targetMesh: THREE.Mesh, margin: number) => {
    const pos = baseGeo.attributes.position;
    const norm = baseGeo.attributes.normal;
    if (!pos || !norm) return;

    const newGeo = targetMesh.geometry;
    const targetPos = newGeo.attributes.position;

    for (let i = 0; i < pos.count; i++) {
      const px = pos.getX(i);
      const py = pos.getY(i);
      const pz = pos.getZ(i);

      const nx = norm.getX(i);
      const ny = norm.getY(i);
      const nz = norm.getZ(i);

      targetPos.setXYZ(i, px + nx * margin, py + ny * margin, pz + nz * margin);
    }
    targetPos.needsUpdate = true;
    newGeo.computeVertexNormals();
  };

  // Update margin shell when slider changes
  useEffect(() => {
    if (marginShellMeshRef.current && rawTumorMeshRef.current) {
      marginShellMeshRef.current.visible = showMarginShell;
      updateMarginShellGeometry(rawTumorMeshRef.current, marginShellMeshRef.current, marginMm);
    }
  }, [marginMm, showMarginShell]);

  // Margin collision check
  const isMarginBreached = marginMm > measuredClearanceMm;

  return (
    <main className="body-explorer-grid" aria-label="Body Explorer Workstation">
      {/* Left Column: Preset, Stature, Anatomy Layers & Cut-Plane Controls */}
      <section className="col-sidebar" aria-label="Preset and Calibration Controls">
        <div className="card preset-card">
          <div className="card-header">
            <div className="card-title-row">
              <User size={18} className="text-cyan-400" />
              <h2 className="card-title">Anatomical Digital Twin</h2>
            </div>
            <span className="badge badge-preset">Procedural 3D</span>
          </div>

          <div className="card-body">
            <p className="preset-description">
              Select an anatomical stature preset or scale height to resize skeletal bones, vascular trees, and organs.
            </p>

            {/* Presets List */}
            <div className="preset-options">
              {BODY_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  className={`btn-preset-card ${selectedPresetId === preset.id ? 'active' : ''}`}
                  onClick={() => handlePresetSelect(preset.id)}
                >
                  <div className="preset-card-head">
                    <span className="preset-name">{preset.name}</span>
                    <span className="preset-tag">{preset.baseHeightMm} mm</span>
                  </div>
                  <span className="preset-summary">{preset.description}</span>
                </button>
              ))}
            </div>

            {/* Height Slider */}
            <div className="stature-slider-box">
              <div className="slider-label-row">
                <span className="slider-title">Standing Stature:</span>
                <span className="slider-value font-mono">{heightMm} mm</span>
              </div>
              <input
                type="range"
                min={1400}
                max={2100}
                step={10}
                value={heightMm}
                onChange={(e) => handleHeightChange(Number(e.target.value))}
                className="slider-range"
              />
              <div className="slider-bounds font-mono">
                <span>1400 mm</span>
                <span>1750 mm (ref)</span>
                <span>2100 mm</span>
              </div>
            </div>

            {/* Anatomical Systems Toggles */}
            <div className="viewer-toggles-box">
              <div className="flex items-center justify-between toggles-header">
                <span>Anatomical Systems:</span>
                {systemsLoading && (
                  <span className="text-xs text-cyan-400 animate-pulse font-mono">Generating...</span>
                )}
              </div>

              {/* Skeletal System Toggle */}
              <div className="toggle-row justify-between py-1">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showSkeletonSystem}
                    onChange={(e) => setShowSkeletonSystem(e.target.checked)}
                  />
                  <div className="flex items-center gap-1.5 font-medium text-slate-200">
                    <Bone size={14} className="text-amber-200" />
                    <span>Skeletal System</span>
                  </div>
                </label>
                <span className="badge badge-dim font-mono text-xs">{itemCounts.skeleton} bones</span>
              </div>

              {/* Vascular System Toggle */}
              <div className="toggle-row justify-between py-1">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showVascularSystem}
                    onChange={(e) => setShowVascularSystem(e.target.checked)}
                  />
                  <div className="flex items-center gap-1.5 font-medium text-slate-200">
                    <Activity size={14} className="text-rose-400" />
                    <span>Vascular System</span>
                  </div>
                </label>
                <span className="badge badge-dim font-mono text-xs">{itemCounts.vascular} vessels</span>
              </div>

              {/* Organs System Toggle */}
              <div className="toggle-row justify-between py-1">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showOrgansSystem}
                    onChange={(e) => setShowOrgansSystem(e.target.checked)}
                  />
                  <div className="flex items-center gap-1.5 font-medium text-slate-200">
                    <Heart size={14} className="text-emerald-400" />
                    <span>Major Organs</span>
                  </div>
                </label>
                <span className="badge badge-dim font-mono text-xs">{itemCounts.organs} organs</span>
              </div>

              {/* Ghost Cyber Hull Toggle */}
              <div className="toggle-row justify-between py-1 border-t border-slate-800/80 mt-1 pt-1.5">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showHologramHull}
                    onChange={(e) => setShowHologramHull(e.target.checked)}
                  />
                  <span className="text-slate-400">Cyber Hull Envelope</span>
                </label>
                <span className="text-xs text-slate-500 font-mono">Translucent</span>
              </div>

              {/* In-Situ Radiology Cut-Plane Toggle */}
              <div className="toggle-row border-t-subtle pt-2 mt-2">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showCutPlane}
                    onChange={(e) => setShowCutPlane(e.target.checked)}
                  />
                  <span className="font-semibold text-cyan-300">In-Situ 2D MRI Cut-Plane</span>
                </label>
              </div>

              {/* Cut-Plane Orientation & Opacity Subcontrols */}
              {showCutPlane && (
                <div className="cutplane-subcontrols">
                  <div className="cutplane-mode-row">
                    <span className="subcontrol-title">Orientation:</span>
                    <div className="btn-group-toggle-sm" role="group" aria-label="Cut-plane orientation">
                      <button
                        className={`btn-submode ${sliceOrientation === 'axial' ? 'active' : ''}`}
                        onClick={() => setSliceOrientation('axial')}
                        title="Axial: Transverse horizontal cross-section"
                      >
                        Axial
                      </button>
                      <button
                        className={`btn-submode ${sliceOrientation === 'coronal' ? 'active' : ''}`}
                        onClick={() => setSliceOrientation('coronal')}
                        title="Coronal: Front-to-back cross-section"
                      >
                        Coronal
                      </button>
                      <button
                        className={`btn-submode ${sliceOrientation === 'sagittal' ? 'active' : ''}`}
                        onClick={() => setSliceOrientation('sagittal')}
                        title="Sagittal: Side-to-side cross-section"
                      >
                        Sagittal
                      </button>
                    </div>
                  </div>

                  <div className="opacity-slider-row">
                    <span className="opacity-label">Plane Opacity:</span>
                    <input
                      type="range"
                      min={0.2}
                      max={1.0}
                      step={0.05}
                      value={cutPlaneOpacity}
                      onChange={(e) => setCutPlaneOpacity(Number(e.target.value))}
                      className="slider-range slider-range-sm"
                    />
                    <span className="opacity-val font-mono">{Math.round(cutPlaneOpacity * 100)}%</span>
                  </div>
                  <span className="pinned-notice font-mono">Pinned strictly at cranial lesion center</span>
                </div>
              )}
            </div>

            {/* Validation Feedback */}
            <div className={`validation-status-box ${validation.ok ? 'val-pass' : 'val-fail'}`}>
              <div className="val-header">
                {validation.ok ? (
                  <>
                    <CheckCircle2 size={16} className="text-emerald-400" />
                    <span>Anatomical Validity: Verified</span>
                  </>
                ) : (
                  <>
                    <AlertCircle size={16} className="text-rose-400" />
                    <span>Anatomical Discrepancy</span>
                  </>
                )}
              </div>
              {validation.ok ? (
                <p className="val-text">All 20 canonical joints conform to bone length and symmetry invariants.</p>
              ) : (
                <ul className="val-errors">
                  {validation.errors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Center Column: 3D Holographic Avatar & Tumor Viewport */}
      <section className="col-viewer" aria-label="3D Body Viewport">
        <div className="viewer-layout-bar">
          <div className="viewer-title-row">
            <span className="layout-bar-title">3D Anatomical Digital Twin</span>
            <span className="badge badge-status">Skeleton + Vascular + Organs + Pathology</span>
          </div>

          <div className="viewer-actions btn-group-toggle">
            <button
              className={`btn btn-sm btn-layout-mode ${cameraView === 'whole-body' ? 'active' : ''}`}
              onClick={() => transitionCamera('whole-body', bodyParams.joints.head)}
              title="View full standing stature"
            >
              <Maximize2 size={14} />
              <span>Full Stature</span>
            </button>
            <button
              className={`btn btn-sm btn-layout-mode ${cameraView === 'cranial-tumor' ? 'active' : ''}`}
              onClick={() => transitionCamera('cranial-tumor', bodyParams.joints.head)}
              title="Zoom into cranial vault to inspect tumor, in-situ MRI plane, and vessel clearance"
            >
              <ZoomIn size={14} className="text-rose-400" />
              <span>Zoom to Tumor</span>
            </button>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => transitionCamera('whole-body', bodyParams.joints.head)}
              title="Reset View"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>

        <div className="avatar-canvas-wrapper" ref={mountRef}>
          {/* Overlay Floating Badges */}
          <div className="canvas-overlay-hud">
            {showSkeletonSystem && (
              <div className="hud-pill">
                <span className="hud-indicator text-amber-200">●</span>
                <span>Skeleton ({itemCounts.skeleton})</span>
              </div>
            )}
            {showVascularSystem && (
              <div className="hud-pill">
                <span className="hud-indicator text-rose-500">●</span>
                <span>Vascular Tree ({itemCounts.vascular})</span>
              </div>
            )}
            {showOrgansSystem && (
              <div className="hud-pill">
                <span className="hud-indicator text-emerald-400">●</span>
                <span>Organs ({itemCounts.organs})</span>
              </div>
            )}
            <div className="hud-pill">
              <span className="hud-indicator text-rose-500">●</span>
              <span>Tumor Mesh (CPA)</span>
            </div>
            {showCutPlane && (
              <div className="hud-pill hud-pill-highlight">
                <Layers size={13} className="text-cyan-400" />
                <span>MRI In-Situ: {sliceOrientation.toUpperCase()}</span>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Right Column: Surgical Margin Simulator & Lesion Telemetry */}
      <section className="col-analysis" aria-label="Lesion Telemetry and Margin Analysis">
        {/* Margin Expansion Simulator Card */}
        <div className="card margin-simulator-card">
          <div className="card-header">
            <div className="card-title-row">
              <Ruler size={18} className="text-amber-400" />
              <h2 className="card-title">Surgical Margin Shell</h2>
            </div>
            <span className="badge badge-dim font-mono">{marginMm.toFixed(1)} mm</span>
          </div>

          <div className="card-body">
            <p className="margin-description">
              Simulate an outward resection margin boundary expanded along surface normals from the 3D tumor mesh.
            </p>

            <div className="stature-slider-box margin-slider-box">
              <div className="slider-label-row">
                <span className="slider-title">Margin Expansion:</span>
                <span className="slider-value font-mono text-amber-400">+{marginMm.toFixed(1)} mm</span>
              </div>
              <input
                type="range"
                min={0.0}
                max={10.0}
                step={0.5}
                value={marginMm}
                onChange={(e) => setMarginMm(Number(e.target.value))}
                className="slider-range slider-range-amber"
              />
              <div className="slider-bounds font-mono">
                <span>0 mm (Tangency)</span>
                <span>5 mm</span>
                <span>10 mm</span>
              </div>
            </div>

            {/* Margin vs Clearance Status Alert */}
            <div className={`margin-alert-card ${isMarginBreached ? 'alert-breach' : 'alert-safe'}`}>
              <div className="alert-header">
                {isMarginBreached ? (
                  <>
                    <ShieldAlert size={18} className="text-rose-400" />
                    <span className="alert-title font-semibold text-rose-400">Margin Boundary Compromised</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={18} className="text-emerald-400" />
                    <span className="alert-title font-semibold text-emerald-400">Clearance Preserved</span>
                  </>
                )}
              </div>
              <p className="alert-text">
                {isMarginBreached
                  ? `An expansion shell of ${marginMm.toFixed(1)} mm exceeds the measured closest distance of ${measuredClearanceMm.toFixed(1)} mm, intersecting adjacent vascular structures.`
                  : `Margin shell (${marginMm.toFixed(1)} mm) remains clear of critical vessel boundary (Distance: ${measuredClearanceMm.toFixed(1)} mm).`}
              </p>
            </div>
          </div>
        </div>

        {/* Cranial Lesion Metadata Card */}
        <div className="card joint-inspector-card">
          <div className="card-header">
            <div className="card-title-row">
              <Crosshair size={18} className="text-rose-400" />
              <h2 className="card-title">Lesion Telemetry</h2>
            </div>
            <span className="badge badge-synthetic">Vestibular / CPA</span>
          </div>

          <div className="card-body">
            <div className="telemetry-grid">
              <div className="telemetry-item">
                <span className="telemetry-label">Structure Name:</span>
                <span className="telemetry-val text-rose-400 font-mono">Cranial Tumor</span>
              </div>
              <div className="telemetry-item">
                <span className="telemetry-label">Authoritative Volume:</span>
                <span className="telemetry-val font-mono">{tumorVolumeMm3.toFixed(1)} mm³</span>
              </div>
              <div className="telemetry-item">
                <span className="telemetry-label">Adjacent Risk Structure:</span>
                <span className="telemetry-val text-sky-400 font-mono">Carotid / CPA Vessel</span>
              </div>
              <div className="telemetry-item">
                <span className="telemetry-label">Closest-Point Clearance:</span>
                <span className="telemetry-val font-mono font-bold text-amber-400">
                  {measuredClearanceMm.toFixed(1)} mm
                </span>
              </div>
            </div>

            <div className="quick-zoom-callout">
              <Sparkles size={16} className="text-cyan-400" />
              <p className="callout-text">
                The 2D in-situ MRI slice is pinned to the tumor's center. Toggle between <strong>Axial, Coronal, and Sagittal</strong> to verify cross-sectional alignment with the 3D mesh.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
};
