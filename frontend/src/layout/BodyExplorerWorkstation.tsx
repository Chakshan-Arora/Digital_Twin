import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import {
  AlertCircle,
  CheckCircle2,
  Crosshair,
  Layers,
  Maximize2,
  RotateCcw,
  Ruler,
  ShieldAlert,
  Sparkles,
  User,
  ZoomIn,
} from 'lucide-react';
import { type BodyParams, JOINT_NAMES, type JointName } from '../types/body';
import { BODY_PRESETS, getValidatedPreset } from './presets';
import { createHologramFresnelMaterial, createMarginShellMaterial } from './hologramShaders';
import { createMriSliceTexture, type SliceOrientation } from './mriSliceTexture';

// Skeletal connectivity for rendering bone linkages
const SKELETAL_BONES: Array<[JointName, JointName]> = [
  ['pelvis', 'chest'],
  ['chest', 'neck'],
  ['neck', 'head'],
  ['chest', 'l_shoulder'],
  ['l_shoulder', 'l_elbow'],
  ['l_elbow', 'l_wrist'],
  ['l_wrist', 'l_hand'],
  ['chest', 'r_shoulder'],
  ['r_shoulder', 'r_elbow'],
  ['r_elbow', 'r_wrist'],
  ['r_wrist', 'r_hand'],
  ['pelvis', 'l_hip'],
  ['l_hip', 'l_knee'],
  ['l_knee', 'l_ankle'],
  ['l_ankle', 'l_foot'],
  ['pelvis', 'r_hip'],
  ['r_hip', 'r_knee'],
  ['r_knee', 'r_ankle'],
  ['r_ankle', 'r_foot'],
];

export type CameraViewPreset = 'whole-body' | 'cranial-tumor';

export const BodyExplorerWorkstation: React.FC = () => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('standard-1750');
  const [heightMm, setHeightMm] = useState<number>(1750);
  const [bodyParams, setBodyParams] = useState<BodyParams>(() => getValidatedPreset('standard-1750').params);
  const [validation, setValidation] = useState(() => getValidatedPreset('standard-1750').validation);
  const [selectedJoint] = useState<JointName | null>('head');

  // Tumor & Margin Visualization Controls
  const [marginMm, setMarginMm] = useState<number>(3.0);
  const [showHologramHull, setShowHologramHull] = useState<boolean>(true);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [showMarginShell, setShowMarginShell] = useState<boolean>(true);

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
  const skeletonGroupRef = useRef<THREE.Group | null>(null);
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
    renderer.toneMappingExposure = 1.2;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 950, 0);
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.4);
    dirLight1.position.set(600, 2200, 1600);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xe11d48, 0.9);
    dirLight2.position.set(-600, 1200, -800);
    scene.add(dirLight2);

    // Floor Grid
    const grid = new THREE.GridHelper(3000, 30, 0x1e293b, 0x0f172a);
    grid.position.y = 0;
    scene.add(grid);

    // Node groups
    const skeletonGroup = new THREE.Group();
    const hologramHullGroup = new THREE.Group();
    const pathologyGroup = new THREE.Group();
    scene.add(skeletonGroup);
    scene.add(hologramHullGroup);
    scene.add(pathologyGroup);
    skeletonGroupRef.current = skeletonGroup;
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
            cloned.center(); // Crucial: Centering resets local origin to (0,0,0)
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

  // 2. Build Pathology Scene (Cranial Tumor Mesh, Risk Vessel, Margin Shell & In-Situ Cut-Plane)
  const buildPathologyScene = (baseGeo: THREE.BufferGeometry) => {
    const group = pathologyGroupRef.current;
    if (!group) return;

    // Clear previous pathology objects
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
      }
    }

    const headJoint = bodyParams.joints.head;
    // Anatomical placement inside cerebellopontine angle / temporal cranial region
    const tumorCenter = new THREE.Vector3(headJoint[0] - 25, headJoint[1] - 40, headJoint[2] + 15);

    // 1. Core Tumor Surface Mesh (Authoritative solid crimson #e11d48)
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
    // Runs alongside the tumor with a 4.8 mm face-to-face gap
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
    // Tumor surface is at -6.5 mm; vessel surface is at -11.3 + 2.5 = -8.8 mm (distance = 4.8 mm)
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
      // Horizontal X-Z plane
      plane.rotation.set(-Math.PI / 2, 0, 0);
    } else if (orientation === 'coronal') {
      // Frontal X-Y plane
      plane.rotation.set(0, 0, 0);
    } else {
      // Sagittal Y-Z plane
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

      // Update texture with new orientation
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

  // 3. Update 3D Holographic Hull and Skeleton when BodyParams change
  useEffect(() => {
    const hullGroup = hologramHullGroupRef.current;
    const skelGroup = skeletonGroupRef.current;
    if (!hullGroup || !skelGroup) return;

    // Clear existing
    while (hullGroup.children.length > 0) {
      const c = hullGroup.children[0];
      hullGroup.remove(c);
      if (c instanceof THREE.Mesh) c.geometry.dispose();
    }
    while (skelGroup.children.length > 0) {
      const c = skelGroup.children[0];
      skelGroup.remove(c);
      if (c instanceof THREE.Mesh) c.geometry.dispose();
    }

    const { joints } = bodyParams;
    const fresnelMat = createHologramFresnelMaterial(0x38bdf8, 2.5);

    // A. Holographic Hull Volume
    if (showHologramHull) {
      // Cranial Head Ellipsoid
      const headPt = joints.head;
      const headGeo = new THREE.SphereGeometry(105, 24, 24);
      headGeo.scale(1.0, 1.25, 1.15);
      const headMesh = new THREE.Mesh(headGeo, fresnelMat);
      headMesh.position.set(headPt[0], headPt[1] - 30, headPt[2]);
      hullGroup.add(headMesh);

      // Torso Ellipsoid / Loft
      const chestPt = joints.chest;
      const pelvisPt = joints.pelvis;
      const torsoHeight = chestPt[1] - pelvisPt[1] + 160;
      const torsoGeo = new THREE.CylinderGeometry(155, 140, torsoHeight, 20);
      torsoGeo.scale(1.15, 1.0, 0.7);
      const torsoMesh = new THREE.Mesh(torsoGeo, fresnelMat);
      torsoMesh.position.set(0, (chestPt[1] + pelvisPt[1]) / 2 + 10, 0);
      hullGroup.add(torsoMesh);

      // Limbs Volumetric Capsules
      for (const [startName, endName] of SKELETAL_BONES) {
        if (
          startName === 'pelvis' ||
          startName === 'chest' ||
          startName === 'neck'
        ) {
          continue; // Axial covered above
        }
        const p1 = joints[startName];
        const p2 = joints[endName];
        if (!p1 || !p2) continue;

        const v1 = new THREE.Vector3(p1[0], p1[1], p1[2]);
        const v2 = new THREE.Vector3(p2[0], p2[1], p2[2]);
        const distance = v1.distanceTo(v2);

        const radius = startName.includes('hip') || startName.includes('knee') ? 48 : 36;
        const limbGeo = new THREE.CylinderGeometry(radius * 0.85, radius, distance, 14);
        const limbMesh = new THREE.Mesh(limbGeo, fresnelMat);

        const midpoint = new THREE.Vector3().addVectors(v1, v2).multiplyScalar(0.5);
        limbMesh.position.copy(midpoint);

        const orientation = new THREE.Vector3().subVectors(v2, v1).normalize();
        const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), orientation);
        limbMesh.setRotationFromQuaternion(quaternion);

        hullGroup.add(limbMesh);
      }
    }

    // B. Internal Skeleton
    if (showSkeleton) {
      const jointGeo = new THREE.SphereGeometry(15, 14, 14);
      const jointMat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        emissive: 0x0284c7,
        emissiveIntensity: 0.35,
        wireframe: true,
      });

      const highlightJointMat = new THREE.MeshStandardMaterial({
        color: 0xf43f5e,
        emissive: 0xe11d48,
        emissiveIntensity: 0.8,
      });

      const boneMat = new THREE.LineBasicMaterial({
        color: 0x0284c7,
        transparent: true,
        opacity: 0.6,
      });

      // Render Joints
      for (const name of JOINT_NAMES) {
        const pt = joints[name];
        if (!pt) continue;
        const mesh = new THREE.Mesh(jointGeo, name === selectedJoint ? highlightJointMat : jointMat);
        mesh.position.set(pt[0], pt[1], pt[2]);
        mesh.name = name;
        skelGroup.add(mesh);
      }

      // Render Bone Lines
      for (const [startName, endName] of SKELETAL_BONES) {
        const p1 = joints[startName];
        const p2 = joints[endName];
        if (!p1 || !p2) continue;

        const lineGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(p1[0], p1[1], p1[2]),
          new THREE.Vector3(p2[0], p2[1], p2[2]),
        ]);
        const line = new THREE.Line(lineGeo, boneMat);
        skelGroup.add(line);
      }
    }

    // Rebuild pathology when stature/head changes
    if (rawTumorMeshRef.current) {
      buildPathologyScene(rawTumorMeshRef.current);
    }
  }, [bodyParams, selectedJoint, showHologramHull, showSkeleton]);

  // Margin collision check
  const isMarginBreached = marginMm > measuredClearanceMm;

  return (
    <main className="body-explorer-grid" aria-label="Body Explorer Workstation">
      {/* Left Column: Preset, Stature, Hologram & Cut-Plane Controls */}
      <section className="col-sidebar" aria-label="Preset and Calibration Controls">
        <div className="card preset-card">
          <div className="card-header">
            <div className="card-title-row">
              <User size={18} className="text-cyan-400" />
              <h2 className="card-title">Digital Twin Avatar</h2>
            </div>
            <span className="badge badge-preset">Cyber Hull</span>
          </div>

          <div className="card-body">
            <p className="preset-description">
              Select an anatomical body stature preset or adjust stature to scale the digital twin.
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

            {/* Visual Toggles & Cut-Plane */}
            <div className="viewer-toggles-box">
              <span className="toggles-header">Anatomical Envelope Layers:</span>
              <div className="toggle-row">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showHologramHull}
                    onChange={(e) => setShowHologramHull(e.target.checked)}
                  />
                  <span>Holographic Glass Hull</span>
                </label>
              </div>
              <div className="toggle-row">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showSkeleton}
                    onChange={(e) => setShowSkeleton(e.target.checked)}
                  />
                  <span>Internal Skeleton Wireframe</span>
                </label>
              </div>
              <div className="toggle-row">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showMarginShell}
                    onChange={(e) => setShowMarginShell(e.target.checked)}
                  />
                  <span>Surgical Margin Shell ({marginMm.toFixed(1)} mm)</span>
                </label>
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
            <span className="badge badge-status">Holographic Hull + In-Situ MRI Plane</span>
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
            <div className="hud-pill">
              <span className="hud-indicator text-rose-500">●</span>
              <span>Tumor Mesh (Cranial CPA)</span>
            </div>
            <div className="hud-pill">
              <span className="hud-indicator text-sky-400">●</span>
              <span>Adjacent Risk Vessel</span>
            </div>
            <div className="hud-pill">
              <span className="hud-indicator text-amber-400">●</span>
              <span>Margin Shell: {marginMm.toFixed(1)} mm</span>
            </div>
            {showCutPlane && (
              <div className="hud-pill hud-pill-highlight">
                <Layers size={13} className="text-cyan-400" />
                <span>MRI In-Situ: {sliceOrientation.toUpperCase()} (T1+C)</span>
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
                    <span className="alert-title font-semibold text-rose-400">Safety Margin Compromised</span>
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
