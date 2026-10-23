import * as THREE from 'three';

export type SliceOrientation = 'axial' | 'coronal' | 'sagittal';

/**
 * Procedurally generates an authentic high-resolution medical MRI slice texture
 * displaying anatomical cranial contours, ventricles, and tumor hyperintensity.
 */
export function createMriSliceTexture(
  orientation: SliceOrientation,
  size = 512
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  if (!ctx) {
    return texture;
  }

  const cx = size / 2;
  const cy = size / 2;

  // In-situ plane is centered on the tumor:
  // Tumor sits directly at plane center (cx, cy)
  const tumorX = cx;
  const tumorY = cy;

  // Head center is anatomically shifted relative to lateral CPA tumor
  const headCenterX = cx + size * (25 / 165);
  const headCenterY = cy - size * (15 / 165);

  // 1. Dark background
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, size, size);

  // 2. Head / Skull Contour (Soft gradient cortical outline centered on head)
  ctx.save();
  ctx.beginPath();
  if (orientation === 'axial') {
    ctx.ellipse(headCenterX, headCenterY, size * 0.46, size * 0.50, 0, 0, Math.PI * 2);
  } else if (orientation === 'coronal') {
    ctx.ellipse(headCenterX, headCenterY + size * 0.04, size * 0.45, size * 0.48, 0, 0, Math.PI * 2);
  } else {
    // Sagittal: slightly elongated anterior-posterior
    ctx.ellipse(headCenterX, headCenterY + size * 0.02, size * 0.48, size * 0.46, 0, 0, Math.PI * 2);
  }

  const brainGrad = ctx.createRadialGradient(headCenterX, headCenterY, size * 0.1, headCenterX, headCenterY, size * 0.48);
  brainGrad.addColorStop(0, '#2d3748');
  brainGrad.addColorStop(0.7, '#1a202c');
  brainGrad.addColorStop(0.9, '#10141d');
  brainGrad.addColorStop(0.97, '#3b4252'); // Skull bone cortex
  brainGrad.addColorStop(1, 'transparent');
  ctx.fillStyle = brainGrad;
  ctx.fill();
  ctx.restore();

  // 3. Ventricular System (Hypointense cerebrospinal fluid)
  ctx.save();
  ctx.fillStyle = '#0a0d14';
  if (orientation === 'axial') {
    // Lateral ventricles (butterfly wings)
    ctx.beginPath();
    ctx.ellipse(headCenterX - size * 0.06, headCenterY - size * 0.02, size * 0.035, size * 0.12, 0.2, 0, Math.PI * 2);
    ctx.ellipse(headCenterX + size * 0.06, headCenterY - size * 0.02, size * 0.035, size * 0.12, -0.2, 0, Math.PI * 2);
    ctx.fill();
  } else if (orientation === 'coronal') {
    ctx.beginPath();
    ctx.ellipse(headCenterX - size * 0.05, headCenterY - size * 0.04, size * 0.03, size * 0.08, 0.15, 0, Math.PI * 2);
    ctx.ellipse(headCenterX + size * 0.05, headCenterY - size * 0.04, size * 0.03, size * 0.08, -0.15, 0, Math.PI * 2);
    ctx.fill();
  } else {
    // Sagittal midline lateral ventricle arch
    ctx.beginPath();
    ctx.arc(headCenterX - size * 0.04, headCenterY - size * 0.02, size * 0.08, 0, Math.PI, true);
    ctx.lineWidth = size * 0.025;
    ctx.strokeStyle = '#0a0d14';
    ctx.stroke();
  }
  ctx.restore();

  // 4. Subtle Sulcal / Gyral Cortical Folds
  ctx.save();
  ctx.strokeStyle = 'rgba(74, 85, 104, 0.35)';
  ctx.lineWidth = 1.5;
  for (let r = size * 0.18; r < size * 0.42; r += size * 0.06) {
    ctx.beginPath();
    ctx.arc(headCenterX, headCenterY, r, 0, Math.PI * 2);
    ctx.setLineDash([8, 14]);
    ctx.stroke();
  }
  ctx.restore();

  // 5. Cranial CPA Tumor Hyperintensity (Centered exactly at tumor center)
  ctx.save();
  const tumorGrad = ctx.createRadialGradient(tumorX, tumorY, size * 0.01, tumorX, tumorY, size * 0.075);
  tumorGrad.addColorStop(0, '#ffffff'); // Intense core
  tumorGrad.addColorStop(0.35, '#f43f5e'); // Crimson contrast edge
  tumorGrad.addColorStop(0.75, 'rgba(225, 29, 72, 0.45)');
  tumorGrad.addColorStop(1, 'transparent');
  ctx.fillStyle = tumorGrad;
  ctx.beginPath();
  ctx.ellipse(tumorX, tumorY, size * 0.065, size * 0.058, 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 6. Adjacent Internal Carotid / Basilar Vessel (Signal void / hypointense dot)
  const vesselOffsetX = size * (11.3 / 165);
  ctx.save();
  ctx.beginPath();
  ctx.arc(tumorX - vesselOffsetX, tumorY, size * 0.022, 0, Math.PI * 2);
  ctx.fillStyle = '#0284c7';
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#38bdf8';
  ctx.stroke();
  ctx.restore();

  // 7. Orientation Border & Medical Caliper Grid
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
  ctx.lineWidth = 2;
  ctx.strokeRect(4, 4, size - 8, size - 8);

  // Corner Crosshairs
  const crossSize = 12;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
  ctx.lineWidth = 1.5;
  // Top-left
  ctx.beginPath();
  ctx.moveTo(12, 12);
  ctx.lineTo(12 + crossSize, 12);
  ctx.moveTo(12, 12);
  ctx.lineTo(12, 12 + crossSize);
  // Bottom-right
  ctx.moveTo(size - 12, size - 12);
  ctx.lineTo(size - 12 - crossSize, size - 12);
  ctx.moveTo(size - 12, size - 12);
  ctx.lineTo(size - 12, size - 12 - crossSize);
  ctx.stroke();

  // Text Tag
  ctx.font = '600 13px monospace';
  ctx.fillStyle = '#38bdf8';
  ctx.fillText(`MRI ${orientation.toUpperCase()} T1+C`, 14, size - 16);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('IN-SITU CUT-PLANE', size - 148, size - 16);
  ctx.restore();

  return texture;
}
