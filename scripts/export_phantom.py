"""Export a two-sphere phantom with non-trivial geometry to data/processed/.

Geometry specs:
- Spacing: (3.0, 0.5, 0.5) mm in (z, y, x) order.
- Origin: non-zero, e.g. (-30.0, 50.0, -20.0) mm.
- Direction: 30-degree oblique rotation around LPS-z axis.
- Centers: sphere shifted toward patient's left (LPS +x) so orientation errors are visible.
- Exports: image.nii.gz, target_mask.nii.gz, risk_mask.nii.gz.
"""

from pathlib import Path

import numpy as np

from hdt.core.structure import Structure
from hdt.geometry.distances import signed_clearance_mm
from hdt.imaging.load_scan import ImageGeometry, save_nifti_image, save_nifti_mask
from hdt.imaging.phantoms import make_sphere


def export_phantom() -> None:
    shape = (50, 100, 100)
    spacing_zyx = (3.0, 0.5, 0.5)
    origin_zyx = (25.0, -40.0, 30.0)

    # 30-degree rotation around LPS z-axis (axis 2 in LPS xyz)
    theta = np.deg2rad(30.0)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Direction matrix in LPS (x, y, z):
    direction_lps = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    geom = ImageGeometry(
        shape=shape,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction_lps,
    )

    # Physical coordinates in volume frame (index * spacing)
    # Volume center along y is 100 * 0.5 / 2 = 25.0 mm, along z is 50 * 3.0 / 2 = 75.0 mm.
    # To shift toward patient's left (LPS +x), place spheres in positive half of x
    c_target = (75.0, 25.0, 32.0)
    r_target = 8.0

    c_risk = (75.0, 25.0, 14.0)
    r_risk = 6.0

    target_mask = make_sphere(shape, spacing_zyx, c_target, r_target)
    risk_mask = make_sphere(shape, spacing_zyx, c_risk, r_risk)

    # Analytic clearance: distance along x is 32.0 - 14.0 = 18.0 mm
    # Clearance = 18.0 - 8.0 - 6.0 = 4.0 mm
    center_dist = float(np.linalg.norm(np.array(c_target) - np.array(c_risk)))
    expected_clearance = center_dist - r_target - r_risk

    # Compute measured clearance
    measured_clearance = signed_clearance_mm(target_mask, risk_mask, spacing_zyx)

    # Synthetic background image with structure contrast
    image = np.full(shape, 100.0, dtype=np.float32)
    image[target_mask] = 400.0
    image[risk_mask] = 800.0

    # Ensure output directory exists
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "phantom_image.nii.gz"
    target_path = out_dir / "target_mask.nii.gz"
    risk_path = out_dir / "risk_mask.nii.gz"

    save_nifti_image(image, geom, img_path)
    save_nifti_mask(target_mask, geom, target_path)
    save_nifti_mask(risk_mask, geom, risk_path)

    # Validate Structure creation
    _ = Structure(name="tumor", mask=target_mask, role="target")
    _ = Structure(name="vessel", mask=risk_mask, role="risk")

    print("=== Two-Sphere Phantom Exported Successfully ===")
    print(f"Output Directory:      {out_dir.resolve()}")
    print(f"Files written:         {img_path.name}, {target_path.name}, {risk_path.name}")
    print(f"Array Shape (z, y, x): {shape}")
    print(f"Spacing (z, y, x) mm:  {spacing_zyx}")
    print(f"Origin (z, y, x) mm:   {origin_zyx}")
    print(
        f"Target Sphere:         center={c_target} mm, radius={r_target} mm "
        "(shifted to patient left)"
    )
    print(f"Risk Sphere:           center={c_risk} mm, radius={r_risk} mm")
    print(f"Expected Analytic Gap: {expected_clearance:.4f} mm")
    print(f"Measured Clearance:    {measured_clearance:.4f} mm")


if __name__ == "__main__":
    export_phantom()
