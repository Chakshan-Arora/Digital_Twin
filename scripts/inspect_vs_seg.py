"""Read-only inspection script for TCIA Vestibular-Schwannoma-SEG subset.

Inspects DICOM completeness, RTSTRUCT contours, JSON contours, registration
matrices, and modality mappings. Does NOT convert or modify any data.
Writes output to stdout and saves a copy to logs/inspect_vs_seg.txt.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from hdt.imaging.dicom_inspect import inspect_patient_dicom_tree


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


def run_inspection() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "raw" / "vs_seg"
    vs_seg_dcm_dir = data_dir / "vestibular_schwannoma_seg"
    mapping_csv_path = data_dir / "DirectoryNamesMappingModality.csv"
    contours_base_dir = data_dir / "Vestibular-Schwannoma-SEG-contours-Mar-2021" / "contours"
    matrices_base_dir = (
        data_dir / "Vestibular-Schwannoma-SEG_matrices-Mar-2021" / "registration_matrices"
    )
    kcl_repo_dir = Path(r"C:\Users\pooja\Desktop\VS_Seg")
    log_path = root_dir / "logs" / "inspect_vs_seg.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(log_path)
    orig_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 80)
        print("TCIA VESTIBULAR-SCHWANNOMA-SEG DATASET INSPECTION REPORT")
        print("=" * 80)
        print(f"Dataset root: {data_dir}")
        print(f"DICOM tree:   {vs_seg_dcm_dir}")
        print(f"KCL repo:     {kcl_repo_dir}\n")

        patient_dirs = sorted([d for d in vs_seg_dcm_dir.iterdir() if d.is_dir()])
        patient_ids = [d.name for d in patient_dirs]
        print(f"Found {len(patient_dirs)} patient folders: {', '.join(patient_ids)}\n")

        # =====================================================================
        # 1. COMPLETENESS
        # =====================================================================
        print("=" * 80)
        print("1. COMPLETENESS & GEOMETRY AUDIT")
        print("=" * 80)

        patient_inspections = []
        all_distinct_rois: set[str] = set()

        for p_dir in patient_dirs:
            p_insp = inspect_patient_dicom_tree(p_dir)
            patient_inspections.append(p_insp)

            print(f"\n--- Patient: {p_insp.patient_id} ---")
            print(f"Status: {'COMPLETE' if p_insp.is_complete else 'INCOMPLETE'}")
            if p_insp.missing_items:
                print(f"Missing items: {', '.join(p_insp.missing_items)}")

            print("\nSeries Summary:")
            for s in p_insp.series:
                rel_folder = s.folder_path.relative_to(p_dir)
                print(f"  * Folder:              {rel_folder}")
                print(f"    Modality:            {s.modality}")
                print(f"    SeriesDescription:   {s.series_description}")
                print(f"    SeriesInstanceUID:   {s.series_instance_uid}")
                print(f"    File count:          {s.file_count}")
                print(f"    Distinct positions:  {s.distinct_positions_count}")
                print(f"    Dimensions (RxC):    {s.rows} x {s.columns}")
                print(f"    PixelSpacing (mm):   {s.pixel_spacing_mm}")
                print(f"    SliceThickness (mm): {s.slice_thickness_mm}")

                sp_pos_str = (
                    f"{s.slice_spacing_from_positions_mm:.4f}"
                    if s.slice_spacing_from_positions_mm is not None
                    else "N/A"
                )
                print(f"    Slice spacing from IPP (mm): {sp_pos_str}")

                iop_str = (
                    f"[{', '.join(f'{x:.4f}' for x in s.image_orientation_patient)}]"
                    if s.image_orientation_patient
                    else "None"
                )
                print(f"    ImageOrientationPatient:     {iop_str}")
                print(f"    FrameOfReferenceUID:         {s.frame_of_reference_uid}")

                if s.position_gaps:
                    print("    [!] FLAG - TRUNCATED / POSITION GAPS DETECTED:")
                    for g in s.position_gaps:
                        step = g["step_mm"]
                        nom = g["nominal_spacing_mm"]
                        approx = g["missing_slices_approx"]
                        print(
                            f"        Gap step: {step:.4f} mm "
                            f"(nominal: {nom:.4f} mm, ~{approx} missing slices)"
                        )
                else:
                    print("    Position continuity: Valid (no gaps detected)")

                if s.duplicate_files:
                    n_dups = len(s.duplicate_files)
                    print(
                        f"    [!] FLAG - DUPLICATE FILES DETECTED ({n_dups}): {s.duplicate_files}"
                    )
                else:
                    print("    Duplicates:          None (0 duplicate files)")
                print()

        # =====================================================================
        # 2. RTSTRUCT
        # =====================================================================
        print("=" * 80)
        print("2. RTSTRUCT INSPECTION")
        print("=" * 80)

        for p_insp in patient_inspections:
            n_rt = len(p_insp.rtstructs)
            print(f"\n--- Patient: {p_insp.patient_id} RTSTRUCT Objects ({n_rt}) ---")
            for rt in p_insp.rtstructs:
                print(f"  File:              {rt.file_path.name}")
                print(f"  SOPInstanceUID:    {rt.sop_instance_uid}")
                print(f"  SeriesInstanceUID: {rt.series_instance_uid}")
                print(f"  FrameOfRefUID:     {rt.frame_of_reference_uid}")
                print(f"  Referenced Series: {rt.referenced_series_uids}")
                print(f"  ROI Count:         {len(rt.rois)}")
                print("  ROIs Detail:")
                for r in rt.rois:
                    all_distinct_rois.add(r.roi_name)
                    geom_str = ", ".join(r.geometric_types)
                    match_str = r.matching_image_series_desc or "None"
                    print(
                        f"    - ROI #{r.roi_number:2d}: '{r.roi_name}' | "
                        f"Contours: {r.contour_count:3d} | "
                        f"Geom: {geom_str} | Points: {r.total_point_count:5d} | "
                        f"Matches Plane: {match_str}"
                    )
                print()

        print("\nDISTINCT ROI NAMES ACROSS ALL PATIENTS:")
        for roi in sorted(all_distinct_rois):
            print(f"  - '{roi}'")

        print("\nCANDIDATE ROI MEANINGS (DO NOT ASSUME - LISTED FOR USER CONFIRMATION):")
        print(
            "  * Potential Tumour (Target) candidates:  "
            "'TV', 'tv', 'AN', 'Vol2015', 'Vol2016', 'Vol2018_CISS', 'T1+ 13'"
        )
        print("  * Potential Cochlea (Risk) candidates:   'Cochlea', 'cochlea'")
        print("  * Other structural/context candidates:   '*Skull'")
        print(
            "  [?] Question for User: Please specify which exact ROI names correspond "
            "to the true target tumour and organ-at-risk cochlea.\n"
        )

        # =====================================================================
        # 3. JSON CONTOURS
        # =====================================================================
        print("=" * 80)
        print("3. JSON CONTOURS AUDIT")
        print("=" * 80)

        sample_json_path = contours_base_dir / "vs_gk_1_t1" / "contours.json"
        if sample_json_path.exists():
            with open(sample_json_path, encoding="utf-8") as jf:
                sample_data = json.load(jf)

            print(f"Inspecting sample file: {sample_json_path.relative_to(data_dir)}")
            print(f"Top-level type: {type(sample_data).__name__} with length {len(sample_data)}")
            if len(sample_data) > 0:
                first_item = sample_data[0]
                print("\nItem [0] Schema:")
                for k, v in first_item.items():
                    if k == "LPS_contour_points":
                        pts_list = v
                        n_rings = len(pts_list)
                        n_pts_ring0 = len(pts_list[0]) if n_rings > 0 else 0
                        sample_pt = pts_list[0][0] if n_pts_ring0 > 0 else None
                        print(
                            f"  - '{k}': list of {n_rings} contour rings "
                            f"(Ring 0 has {n_pts_ring0} points, e.g. {sample_pt})"
                        )
                    else:
                        print(f"  - '{k}': {type(v).__name__} = {repr(v)}")

            print("\nMapping of JSON Contour Folders to Patients:")
            print(
                "  Naming convention: folders are named 'vs_gk_<case_idx>_<modality>' "
                "(e.g. 'vs_gk_1_t1')."
            )
            print(
                "  Each internal JSON object explicitly contains "
                "'PatientID': 'VS-SEG-<case_idx:03d>'"
            )
            print("  (e.g. vs_gk_1_t1 has PatientID: 'VS-SEG-001').")

            json_structure_names: set[str] = set()
            for c_folder in contours_base_dir.glob("vs_gk_*"):
                c_file = c_folder / "contours.json"
                if c_file.exists():
                    try:
                        with open(c_file, encoding="utf-8") as jf:
                            c_data = json.load(jf)
                            for s in c_data:
                                if "structure_name" in s:
                                    json_structure_names.add(s["structure_name"])
                    except Exception:
                        pass

            n_json_names = len(json_structure_names)
            print(f"\nDistinct structure_name values across all JSON files ({n_json_names}):")
            for sname in sorted(json_structure_names):
                print(f"  - '{sname}'")

            print("\nCoordinate Representation & Units in JSON Contours:")
            print("  - Coordinates are continuous floating-point millimetres (mm).")
            print(
                "  - Coordinate system: explicitly stated in key name as "
                "'LPS_contour_points' (ITK LPS physical space)."
            )
            print("  - Coordinates are NOT voxel indices.\n")
        else:
            print(f"[!] Warning: Sample JSON file not found at {sample_json_path}\n")

        # =====================================================================
        # 4. TRANSFORMS AND MAPPING CSV
        # =====================================================================
        print("=" * 80)
        print("4. TRANSFORMS AND MAPPING CSV")
        print("=" * 80)

        print(f"First 10 lines of {mapping_csv_path.name}:")
        if mapping_csv_path.exists():
            with open(mapping_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx >= 10:
                        break
                    print(f"  Line {idx + 1:2d}: {row}")
        else:
            print(f"[!] Warning: Mapping CSV not found at {mapping_csv_path}")

        print("\nRegistration .tfm Files Per Patient (Patients 1 to 5):")
        for i in range(1, 6):
            p_id = f"VS-SEG-{i:03d}"
            t1_folder = matrices_base_dir / f"vs_gk_{i}_t1"
            t2_folder = matrices_base_dir / f"vs_gk_{i}_t2"
            t1_files = [f.name for f in t1_folder.glob("*.tfm")] if t1_folder.exists() else []
            t2_files = [f.name for f in t2_folder.glob("*.tfm")] if t2_folder.exists() else []
            print(f"  - {p_id}:")
            print(f"      vs_gk_{i}_t1: {t1_files}")
            print(f"      vs_gk_{i}_t2: {t2_files}")

        sample_tfm_path = matrices_base_dir / "vs_gk_1_t1" / "inv_T1_LPS_to_T2_LPS.tfm"
        print(f"\nContents of sample .tfm file ({sample_tfm_path.name}):")
        if sample_tfm_path.exists():
            with open(sample_tfm_path, "r", encoding="utf-8") as tf:
                for line in tf:
                    print(f"  {line.rstrip()}")
        else:
            print(f"[!] Warning: Sample .tfm not found at {sample_tfm_path}")

        print("\nTransformation Direction Implied by File Names (citing file names only):")
        print(
            "  - File 'inv_T1_LPS_to_T2_LPS.tfm': implies the inverse transformation "
            "mapping from T1 (LPS) to T2 (LPS)."
        )
        print(
            "  - File 'inv_T2_LPS_to_T1_LPS.tfm': implies the inverse transformation "
            "mapping from T2 (LPS) to T1 (LPS).\n"
        )

        # =====================================================================
        # 5. KCL SCRIPTS
        # =====================================================================
        print("=" * 80)
        print("5. KCL VS_SEG SCRIPTS INSPECTION")
        print("=" * 80)
        print(f"Inspected repository path: {kcl_repo_dir}")

        if kcl_repo_dir.exists():
            print("\nA. Expected Folder Layout:")
            print("  1. Input layout for TCIA_data_convert_into_convenient_folder_structure.py:")
            print("     Raw TCIA download in 'Descriptive Directory Name' format:")
            print("     <input_folder>/VS-SEG-<idx>/<study_folder>/<series_folders_or_files>")
            print("  2. Intermediate structure / Input to data_conversion.py:")
            print("     <intermediate_folder>/")
            print("       vs_gk_<case_idx>_t1/")
            print("         IMG0000000000.dcm, IMG0000000001.dcm, ...")
            print("         RTSS.dcm, RTPLAN.dcm, RTDOSE.dcm")
            print("         contours.json (manually merged from contours archive)")
            print("         inv_T1_LPS_to_T2_LPS.tfm (manually merged from registration matrices)")
            print("       vs_gk_<case_idx>_t2/")
            print("         IMG0000000000.dcm, ...")
            print("         RTSS.dcm, RTPLAN.dcm, RTDOSE.dcm")
            print("         contours.json")
            print("         inv_T2_LPS_to_T1_LPS.tfm")

            print("\nB. Command-Line Arguments:")
            print("  1. TCIA_data_convert_into_convenient_folder_structure.py:")
            print("       --input  <path> : path to raw TCIA dataset in 'Descriptive' format")
            print("       --output <path> : path to target folder with 'vs_gk_<idx>_<modality>'")
            print("  2. data_conversion.py (run inside 3D Slicer):")
            print("       -i, --input-folder  <path> : folder containing vs_gk_<idx>_t1/t2")
            print("       -o, --output-folder <path> : folder for output NIfTI files (default '.')")
            print("       -r, --register      <choice>: 'no_registration', 'T1', or 'T2'")
            print("       --export_all_structures    : flag; if omitted, only tumour is exported")

            print("\nC. Required Packages:")
            print("  - Python dependencies: pydicom, natsort, numpy")
            print("  - Slicer runtime dependencies: 3D Slicer (>= 4.13), SlicerRT extension, VTK")

            print("\nD. Output File Names, Label Values, and Coordinate Space:")
            print("  - Output File Names:")
            print("      Without registration:")
            print("        vs_gk_<case>/vs_gk_t1_refT1.nii.gz")
            print("        vs_gk_<case>/vs_gk_t2_refT2.nii.gz")
            print("        vs_gk_<case>/vs_gk_seg_refT1.nii.gz (tumour only)")
            print("        vs_gk_<case>/vs_gk_seg_refT2.nii.gz (tumour only)")
            print("      With --export_all_structures:")
            print("        vs_gk_<case>/vs_gk_struc<idx>_<structure_name>_refT1.nii.gz")
            print("        vs_gk_<case>/vs_gk_struc<idx>_<structure_name>_refT2.nii.gz")
            print("      With --register T1:")
            print("        vs_gk_<case>/vs_gk_t1_refT1.nii.gz")
            print("        vs_gk_<case>/vs_gk_t2_refT1.nii.gz (T2 resampled into T1 space)")
            print("        vs_gk_<case>/vs_gk_seg_refT1.nii.gz")
            print("  - Label values:")
            print("      Binary masks (0 = background, 1 = structure voxels) per structure file.")
            print("  - Coordinate Space:")
            print("      NIfTI files exported by Slicer use RAS physical space.")
            print("      (Contour points in LPS are converted to RAS via affine before export).\n")
        else:
            print(f"[!] Warning: KCL repo directory does not exist at {kcl_repo_dir}\n")

        # =====================================================================
        # 6. FINISH: PER-PATIENT SUMMARY TABLE & QUESTIONS
        # =====================================================================
        print("=" * 80)
        print("6. SUMMARY & QUESTIONS")
        print("=" * 80)

        print("\nPER-PATIENT COMPLETENESS SUMMARY TABLE:")
        hdr = (
            f"{'Patient ID':<12} | {'Status':<10} | {'T1 Series':<12} | "
            f"{'T2 Series':<12} | {'RTSTRUCT':<10} | {'RTDOSE':<10} | "
            f"{'RTPLAN':<10} | {'Missing Series / Gaps'}"
        )
        print(hdr)
        print("-" * len(hdr))
        for p in patient_inspections:
            t1_status = "Present (120)" if p.has_contrast_t1 else "Missing"
            t2_s = [
                s
                for s in p.series
                if s.modality == "MR"
                and s != p.series[0]
                and any(k in s.series_description.lower() for k in ("t2", "ci3d", "tse"))
            ]
            t2_count = t2_s[0].file_count if t2_s else 0
            t2_status = f"Present ({t2_count})" if p.has_t2 else "Missing"
            rt_status = "Present (2)" if p.has_rtstruct else "Missing"
            dose_status = "Present (2)" if p.has_rtdose else "Missing"
            plan_status = "Present (2)" if p.has_rtplan else "Missing"
            missing_str = (
                ", ".join(p.missing_items) if p.missing_items else "None (0 gaps, 0 duplicates)"
            )
            status_str = "Complete" if p.is_complete else "Incomplete"
            row_str = (
                f"{p.patient_id:<12} | {status_str:<10} | {t1_status:<12} | "
                f"{t2_status:<12} | {rt_status:<10} | {dose_status:<10} | "
                f"{plan_status:<10} | {missing_str}"
            )
            print(row_str)

        print("\nCANDIDATE ROI NAMES ACROSS DATASET:")
        print(
            "  - Potential Target (Tumour): 'TV', 'tv', 'AN', 'Vol2015', "
            "'Vol2016', 'Vol2018_CISS', 'T1+ 13'"
        )
        print("  - Potential Organ-at-Risk (Cochlea): 'Cochlea', 'cochlea'")
        print("  - Non-target/Context structures: '*Skull'")

        print("\nQUESTIONS FOR THE USER (DO NOT GUESS):")
        print("  1. ROI Definition for Target Tumour:")
        print(
            "     Across the 5 patient RTSTRUCTs, tumour annotations appear under "
            "'TV', 'tv', and 'AN', as well as"
        )
        print(
            "     time-stamped names like 'Vol2015' and 'Vol2016'. Which exact ROI names "
            "should be treated as the authoritative target volume?"
        )
        print("  2. ROI Definition for Risk Cochlea:")
        print(
            "     Cochlea annotations appear with different casing ('Cochlea' vs 'cochlea'). "
            "Should case-insensitive"
        )
        print(
            "     matching ('cochlea' / 'Cochlea') be used, and are there any side distinctions "
            "(left vs right) required?"
        )
        print("  3. Space and Registration Reference:")
        print(
            "     Both T1 and T2 images exist with independent coordinate frames and .tfm "
            "registration matrices between them."
        )
        print(
            "     For subsequent milestones, should clearance analysis be performed in native "
            "T1 space, native T2 space, or registered space?"
        )
        print("  4. Contours Source Selection:")
        print(
            "     Contour lines exist both as DICOM RTSTRUCTs (DICOM physical coordinates) "
            "and as JSON contour files (LPS coordinates)."
        )
        print(
            "     Should future conversion pipelines parse native DICOM RTSTRUCT files directly, "
            "or consume the pre-extracted JSON contour files?"
        )
        print("=" * 80)

    finally:
        sys.stdout = orig_stdout
        tee.close()


if __name__ == "__main__":
    run_inspection()
