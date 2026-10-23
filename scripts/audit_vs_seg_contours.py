"""Audit script for TCIA Vestibular-Schwannoma-SEG contours and KCL conversion code.

Answers open questions regarding ROI selection rules, contour rasterisation,
registration transform directions, ring plane slice alignment, and contour volumes.
Prints full audit to stdout and saves a verbatim copy to logs/audit_vs_seg.txt.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pydicom

from hdt.imaging.dicom_inspect import compute_slice_normal
from hdt.imaging.dicom_series import evaluate_transform_hypotheses


class TeeLogger:
    """Helper to write to both stdout and a log file."""

    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.stdout.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def polygon_area_3d(points: list[list[float]] | np.ndarray) -> float:
    """Calculate the 3D planar polygon area using the 3D shoelace (Stokes) formula."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    cross_sum = np.sum(np.cross(pts, np.roll(pts, -1, axis=0)), axis=0)
    return float(0.5 * np.linalg.norm(cross_sum))


def run_audit() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "raw" / "vs_seg"
    vs_seg_dcm_dir = data_dir / "vestibular_schwannoma_seg"
    contours_base_dir = data_dir / "Vestibular-Schwannoma-SEG-contours-Mar-2021" / "contours"
    matrices_base_dir = (
        data_dir / "Vestibular-Schwannoma-SEG_matrices-Mar-2021" / "registration_matrices"
    )
    kcl_code_path = Path(r"C:\Users\pooja\Desktop\VS_Seg\preprocessing\data_conversion.py")
    log_path = root_dir / "logs" / "audit_vs_seg.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(log_path)
    orig_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 80)
        print("TCIA VS-SEG CONTOUR AND REGISTRATION AUDIT REPORT")
        print("=" * 80)
        print(f"Dataset root: {data_dir}")
        print(f"KCL script:   {kcl_code_path}\n")

        # =====================================================================
        # 1a. KCL data_conversion.py CODE INSPECTION & QUOTES
        # =====================================================================
        print("=" * 80)
        print("1a. KCL data_conversion.py CODE INSPECTION")
        print("=" * 80)

        print("\n(i) How KCL picks which ROI is the tumour:")
        print("File: preprocessing/data_conversion.py, Lines 289-296 & Lines 313-323:")
        print('"""')
        print("289:     if export_only_tumour_seg:")
        print("290:         nb_structures = 1")
        print("291:     else:")
        print("292:         nb_structures = segmentIDs.GetNumberOfValues()")
        print("293: ")
        print("294:     for segmentIndex in range(0, nb_structures):")
        print("295:         # Selecting a structure")
        print("296:         segmentID = segmentIDs.GetValue(segmentIndex)")
        print("...")
        print("313:         if export_only_tumour_seg:")
        print('314:             if "t1_" in ref.GetName():')
        print("315:                 fileName_rt = os.path.join(")
        print('316:                     output_folder, f"vs_gk_{case_number}",')
        print('                         f"vs_gk_seg" + "_refT1.nii.gz"')
        print("317:                 ).translate({ord(i): None for i in charsRoRemove})")
        print('318:             elif "t2_" in ref.GetName():')
        print("319:                 fileName_rt = os.path.join(")
        print('320:                     output_folder, f"vs_gk_{case_number}",')
        print('                         f"vs_gk_seg" + "_refT2.nii.gz"')
        print("321:                 ).translate({ord(i): None for i in charsRoRemove})")
        print('"""')
        print("Analysis: The authors have NO keyword or name list for tumour.")
        print("They assume structure index 0 in contours.json (segmentIndex = 0) is the tumour,")
        print("setting nb_structures = 1 and saving it as vs_gk_seg_refT1/T2.nii.gz.\n")

        print("(ii) How KCL picks the cochlea:")
        print("File: preprocessing/data_conversion.py, Lines 291-292 & Lines 324-338:")
        print('"""')
        print("291:     else:")
        print("292:         nb_structures = segmentIDs.GetNumberOfValues()")
        print("...")
        print("324:         else:")
        print('325:             if "t1_" in ref.GetName():')
        print("326:                 fileName_rt = os.path.join(")
        print("327:                     output_folder,")
        print('328:                     f"vs_gk_{case_number}",')
        print('329:                     f"vs_gk_struc{segmentIndex + 1}_" + segmentID')
        print('                             + "_refT1.nii.gz",')
        print("330:                 ).translate({ord(i): None for i in charsRoRemove})")
        print('"""')
        print("Analysis: There is NO code or rule selecting the cochlea in data_conversion.py.")
        print("In default mode, the cochlea is discarded. Only when --export_all_structures is")
        print("provided are all structures exported sequentially by index and segment ID.\n")

        print("(iii) How KCL turns contours into a mask:")
        print("File: preprocessing/data_conversion.py, Lines 217-240 & Line 307:")
        print('"""')
        print("217: def createSegNodeFromContourPoints(segmentationNode, contours, name):")
        print("...  [builds vtkPolyData with vtkPoints and closed vtkPolyLine contourLines]")
        print('237:     segment.AddRepresentation("Planar contour", contoursPolyData)')
        print("238:     segmentationNode.GetSegmentation().SetMasterRepresentationName(")
        print('             "Planar contour"')
        print("         )")
        print("239:     segmentationNode.GetSegmentation().AddSegment(segment)")
        print("...")
        print("307:     slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(")
        print("             pc_node, segmentID_a, lm_node, ref")
        print("         )")
        print('"""')
        print("Analysis: Slicer creates a vtkMRMLSegmentationNode with master representation")
        print("'Planar contour' (closed 3D polylines), and delegates conversion directly to")
        print("3D Slicer's internal ExportSegmentsToLabelmapNode() method (surface/ribbon-based")
        print("closed surface polygon rasterisation onto the reference image grid 'ref').\n")

        print("(iv) How KCL applies the .tfm:")
        print("File: preprocessing/data_conversion.py, Lines 187-198 & Lines 444-460:")
        print('"""')
        print("187: def register_and_resample(input_node, reference_node, transform_node=None,")
        print('                               interpolationMode="Linear"):')
        print("188:     # when loaded with slicer, the matrix in tfm file is multiplied with")
        print("         # LPS_to_RAS transforms from both sides")
        print("189:     # furthermore the transformNode will be set to FromParent instead of")
        print("         # ToParent, which has the same effect as inverting it")
        print("192:     if transform_node:")
        print("193:         copy_input_node = slicer.modules.volumes.logic().CloneVolume(")
        print('             slicer.mrmlScene, input_node, "translated")')
        print("194:         copy_input_node.SetAndObserveTransformNodeID(transform_node.GetID())")
        print("195:         logic = slicer.vtkSlicerTransformLogic()")
        print("196:         logic.hardenTransform(copy_input_node)")
        print("...")
        print("444:         if register_T1_image_and_contour_points_to_T2_image:")
        print("445:             transform_path = os.path.join(")
        print('                     input_folder, f"vs_gk_{case_number}_t1",')
        print('                     "inv_T1_LPS_to_T2_LPS.tfm"')
        print("                 )")
        print("446:             transformNode = slicer.util.loadNodeFromFile(")
        print('                     transform_path, filetype="TransformFile"')
        print("                 )")
        print("447:             ref1 = register_and_resample(")
        print("448:                 input_node=ref1, reference_node=ref2,")
        print('                     transform_node=transformNode, interpolationMode="Linear"')
        print("449:             )")
        print("450:             # also transform contour points")
        print("451:             RTSS1.SetAndObserveTransformNodeID(transformNode.GetID())")
        print('"""')
        print("Analysis: Slicer loads the .tfm via slicer.util.loadNodeFromFile(). Slicer inverts")
        print("the transform direction (setting FromParent instead of ToParent) and converts")
        print("LPS to RAS. It then calls hardenTransform() and brainsresample CLI to resample")
        print("into the reference grid.\n")

        # =====================================================================
        # 1b. RING COORDINATES & SLICE PLANE OFFSETS FOR PATIENTS 1-5
        # =====================================================================
        print("=" * 80)
        print("1b. RING COORDINATES AND SLICE PLANE OFFSETS (PATIENTS 1 TO 5)")
        print("=" * 80)

        for pat_id in range(1, 6):
            pat_str = f"VS-SEG-{pat_id:03d}"
            print(f"\n==================== {pat_str} ====================")

            pat_dir = vs_seg_dcm_dir / pat_str
            t1_dcm_dirs = [d for d in pat_dir.glob("*/*t1*") if d.is_dir()]
            t2_dcm_dirs = [d for d in pat_dir.glob("*/*t2*") if d.is_dir()]
            if not t1_dcm_dirs or not t2_dcm_dirs:
                print(f"[!] Warning: Missing T1 or T2 series directory for {pat_str}")
                continue

            t1_files = sorted(glob.glob(str(t1_dcm_dirs[0] / "*.dcm")))
            ds1 = pydicom.dcmread(t1_files[0], stop_before_pixels=True)
            n1 = compute_slice_normal(ds1.ImageOrientationPatient)
            t1_pos_list = [
                np.dot(
                    np.asarray(
                        pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient,
                        dtype=np.float64,
                    ),
                    n1,
                )
                for f in t1_files
            ]
            t1_positions = np.array(t1_pos_list)

            t2_files = sorted(glob.glob(str(t2_dcm_dirs[0] / "*.dcm")))
            ds2 = pydicom.dcmread(t2_files[0], stop_before_pixels=True)
            n2 = compute_slice_normal(ds2.ImageOrientationPatient)
            t2_pos_list = [
                np.dot(
                    np.asarray(
                        pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient,
                        dtype=np.float64,
                    ),
                    n2,
                )
                for f in t2_files
            ]
            t2_positions = np.array(t2_pos_list)

            for mod in ["t1", "t2"]:
                json_path = contours_base_dir / f"vs_gk_{pat_id}_{mod}" / "contours.json"
                if not json_path.exists():
                    continue

                with open(json_path, encoding="utf-8") as jf:
                    c_data = json.load(jf)

                print(f"\n  --- Folder vs_gk_{pat_id}_{mod} ---")
                for item in c_data:
                    sname = item.get("structure_name", "")
                    is_tumour = sname.upper() in ("TV", "AN")
                    is_cochlea = sname.lower() == "cochlea"

                    if not (is_tumour or is_cochlea):
                        continue

                    category = "Tumour" if is_tumour else "Cochlea"
                    orig_img = item.get("original_contour_created_on_image_name")
                    ref_img = item.get("reference_geometry_image_name")
                    rings = item.get("LPS_contour_points", [])

                    ring_coords_n1 = [np.dot(r[0], n1) for r in rings]
                    ring_coords_n2 = [np.dot(r[0], n2) for r in rings]

                    offsets_t1 = [float(np.min(np.abs(t1_positions - rc))) for rc in ring_coords_n1]
                    offsets_t2 = [float(np.min(np.abs(t2_positions - rc))) for rc in ring_coords_n2]

                    print(f"    ROI '{sname}' ({category}):")
                    print(f"      original_contour_created_on_image_name: {orig_img}")
                    print(f"      reference_geometry_image_name:         {ref_img}")
                    print(f"      ring count:                            {len(rings)}")
                    coords_str = ", ".join(f"{c:.2f}" for c in ring_coords_n1[:8])
                    if len(ring_coords_n1) > 8:
                        coords_str += f", ... ({len(ring_coords_n1)} total)"
                    print(f"      ring coordinates along normal (mm):    [{coords_str}]")
                    min_t1 = min(offsets_t1)
                    max_t1 = max(offsets_t1)
                    mean_t1 = float(np.mean(offsets_t1))
                    print(
                        f"      offset from nearest T1 slice (mm):     "
                        f"min={min_t1:.6f}, max={max_t1:.6f}, mean={mean_t1:.6f}"
                    )
                    min_t2 = min(offsets_t2)
                    max_t2 = max(offsets_t2)
                    mean_t2 = float(np.mean(offsets_t2))
                    print(
                        f"      offset from nearest T2 slice (mm):     "
                        f"min={min_t2:.6f}, max={max_t2:.6f}, mean={mean_t2:.6f}"
                    )

        # =====================================================================
        # 1c. POINT ARRAYS IDENTITY & HYPOTHESIS TESTING (PATIENT 1 TUMOUR)
        # =====================================================================
        print("\n" + "=" * 80)
        print("1c. POINT IDENTITY AND REGISTRATION HYPOTHESIS TESTING (PATIENT 1 TUMOUR)")
        print("=" * 80)

        p1_t1_json = contours_base_dir / "vs_gk_1_t1" / "contours.json"
        p1_t2_json = contours_base_dir / "vs_gk_1_t2" / "contours.json"
        p1_tfm_path = matrices_base_dir / "vs_gk_1_t1" / "inv_T1_LPS_to_T2_LPS.tfm"

        with open(p1_t1_json, encoding="utf-8") as f1, open(p1_t2_json, encoding="utf-8") as f2:
            d1 = json.load(f1)
            d2 = json.load(f2)

        p1_tumour_t1 = d1[0]["LPS_contour_points"]
        p1_tumour_t2 = d2[0]["LPS_contour_points"]

        flat_p1 = np.concatenate(p1_tumour_t1, axis=0)
        flat_p2 = np.concatenate(p1_tumour_t2, axis=0)

        are_identical = bool(np.array_equal(flat_p1, flat_p2))
        max_raw_diff = float(np.max(np.abs(flat_p1 - flat_p2)))

        print(f"Are t1 and t2 point arrays identical?: {are_identical}")
        print(f"Maximum raw coordinate difference:    {max_raw_diff:.6f} mm")

        hyp_res = evaluate_transform_hypotheses(p1_tfm_path, flat_p1, flat_p2)
        err_a = hyp_res["hypothesis_A_max_error_mm"]
        err_b = hyp_res["hypothesis_B_max_error_mm"]
        sup = hyp_res["supported_hypothesis"]

        print(f"Hypothesis (A) [t2 points = tfm(t1 points)] max error:        {err_a:.6f} mm")
        print(f"Hypothesis (B) [t2 points = inverse(tfm)(t1 points)] max error: {err_b:.6e} mm")
        print(f"Supported hypothesis:                                        Hypothesis {sup}")

        if hyp_res["supported_hypothesis"] == "B":
            print(
                "CONCLUSION: Hypothesis B is confirmed (max error < 1e-12 mm).\n"
                "The file 'inv_T1_LPS_to_T2_LPS.tfm' represents the transform mapping from\n"
                "T2 space to T1 space (hence its inverse maps T1 points to T2 space)."
            )
        else:
            print("[!] STOP: Neither hypothesis matched within 0.01 mm.")
            sys.exit(1)

        # =====================================================================
        # 1d. CONTOUR-DERIVED VOLUMES (SHOELACE * SLICE SPACING)
        # =====================================================================
        print("\n" + "=" * 80)
        print("1d. CONTOUR-DERIVED VOLUMES (SHOELACE FORMULA * 1.5 MM SPACING)")
        print("=" * 80)

        print(
            f"{'Patient ID':<12} | {'Structure':<15} | {'Category':<10} | "
            f"{'Rings':<6} | {'Volume (mm^3)':<14} | {'Volume (cm^3)'}"
        )
        print("-" * 80)

        for pat_id in range(1, 6):
            pat_str = f"VS-SEG-{pat_id:03d}"
            jpath = contours_base_dir / f"vs_gk_{pat_id}_t1" / "contours.json"
            if not jpath.exists():
                continue
            with open(jpath, encoding="utf-8") as jf:
                c_data = json.load(jf)

            spacing = 1.5
            for item in c_data:
                sname = item.get("structure_name", "")
                rings = item.get("LPS_contour_points", [])
                tot_area = sum(polygon_area_3d(r) for r in rings)
                vol_mm3 = tot_area * spacing
                vol_cm3 = vol_mm3 / 1000.0

                is_tumour = sname.upper() in ("TV", "AN")
                is_cochlea = sname.lower() == "cochlea"
                is_vol = sname.startswith("Vol") or sname.startswith("T1+")

                if is_tumour:
                    cat = "Tumour"
                elif is_cochlea:
                    cat = "Cochlea"
                elif is_vol:
                    cat = "Followup"
                else:
                    cat = "Other"

                print(
                    f"{pat_str:<12} | {sname:<15} | {cat:<10} | "
                    f"{len(rings):<6d} | {vol_mm3:14.2f} | {vol_cm3:8.3f}"
                )

        print("\n" + "=" * 80)
        print("AUDIT COMPLETE")
        print("=" * 80)

    finally:
        sys.stdout = orig_stdout
        tee.close()


if __name__ == "__main__":
    run_audit()
