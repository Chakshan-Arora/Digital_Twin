"""Export VS-SEG cases 1 through 5 and produce the verification report.

Generates:
- data/processed/cases/VS-SEG-00N/
  - t2.nii.gz
  - structures/tumour.nii.gz
  - structures/cochlea.nii.gz
  - structures_alt/tumour_capped.nii.gz
  - structures_alt/cochlea_capped.nii.gz
  - case.json
- logs/vs_export_report.txt
"""

from __future__ import annotations

from pathlib import Path

from hdt.domains.vs_seg.export import CaseExportReport, export_case

PATIENTS = ["VS-SEG-001", "VS-SEG-002", "VS-SEG-003", "VS-SEG-004", "VS-SEG-005"]
VS_SERIES_DIR = Path("data/processed/vs_series")
CONTOURS_DIR = Path("data/raw/vs_seg/Vestibular-Schwannoma-SEG-contours-Mar-2021/contours")
TFM_DIR = Path("data/raw/vs_seg/Vestibular-Schwannoma-SEG_matrices-Mar-2021/registration_matrices")
CASES_DIR = Path("data/processed/cases")
REPORT_LOG = Path("logs/vs_export_report.txt")


def format_case_report(r: CaseExportReport) -> str:
    sep = "=" * 80
    sub_sep = "-" * 80

    t_reduct_mm3 = r.tumour_slab.volume_mm3 - r.tumour_capped.volume_mm3
    t_reduct_pct = (
        (t_reduct_mm3 / r.tumour_slab.volume_mm3 * 100) if r.tumour_slab.volume_mm3 > 0 else 0.0
    )

    c_reduct_mm3 = r.cochlea_slab.volume_mm3 - r.cochlea_capped.volume_mm3
    c_reduct_pct = (
        (c_reduct_mm3 / r.cochlea_slab.volume_mm3 * 100) if r.cochlea_slab.volume_mm3 > 0 else 0.0
    )

    t_multi = r.tumour_slab.multi_ring_planes if r.tumour_slab.multi_ring_planes else "None"
    c_multi = r.cochlea_slab.multi_ring_planes if r.cochlea_slab.multi_ring_planes else "None"

    t_bb = r.tumour_slab.bbox_ras_mm
    c_bb = r.cochlea_slab.bbox_ras_mm

    t_bb_str = (
        f"X=[{t_bb[0][0]:.2f}, {t_bb[0][1]:.2f}], "
        f"Y=[{t_bb[1][0]:.2f}, {t_bb[1][1]:.2f}], "
        f"Z=[{t_bb[2][0]:.2f}, {t_bb[2][1]:.2f}]"
    )
    c_bb_str = (
        f"X=[{c_bb[0][0]:.2f}, {c_bb[0][1]:.2f}], "
        f"Y=[{c_bb[1][0]:.2f}, {c_bb[1][1]:.2f}], "
        f"Z=[{c_bb[2][0]:.2f}, {c_bb[2][1]:.2f}]"
    )

    lines = [
        sep,
        f"CASE REPORT: {r.case_id}",
        sep,
        "1. STRUCTURE: TUMOUR (Role: target)",
        f"   ROI Name: {r.tumour_slab.name}",
        "   [Slab Variant - default end extent]",
        f"     Voxel count:              {r.tumour_slab.voxel_count}",
        f"     Volume:                   {r.tumour_slab.volume_mm3:.3f} mm3",
        f"     Shoelace volume:          {r.tumour_slab.shoelace_volume_mm3:.3f} mm3",
        f"     Volume / Shoelace ratio:  {r.tumour_slab.shoelace_ratio:.4f}",
        f"     Connected components (6): {r.tumour_slab.connected_components}",
        f"     Bounding box (RAS mm):    {t_bb_str}",
        f"     Planes with >1 ring:      {t_multi}",
        "   [Capped Variant - 0.0 mm end extent]",
        f"     Voxel count:              {r.tumour_capped.voxel_count}",
        f"     Volume:                   {r.tumour_capped.volume_mm3:.3f} mm3",
        f"     Volume reduction:         {t_reduct_mm3:.3f} mm3 ({t_reduct_pct:.2f}%)",
        "",
        "2. STRUCTURE: COCHLEA (Role: risk)",
        f"   ROI Name: {r.cochlea_slab.name}",
        "   [Slab Variant - default end extent]",
        f"     Voxel count:              {r.cochlea_slab.voxel_count}",
        f"     Volume:                   {r.cochlea_slab.volume_mm3:.3f} mm3",
        f"     Shoelace volume:          {r.cochlea_slab.shoelace_volume_mm3:.3f} mm3",
        f"     Volume / Shoelace ratio:  {r.cochlea_slab.shoelace_ratio:.4f}",
        f"     Connected components (6): {r.cochlea_slab.connected_components}",
        f"     Bounding box (RAS mm):    {c_bb_str}",
        f"     Planes with >1 ring:      {c_multi}",
        "   [Capped Variant - 0.0 mm end extent]",
        f"     Voxel count:              {r.cochlea_capped.voxel_count}",
        f"     Volume:                   {r.cochlea_capped.volume_mm3:.3f} mm3",
        f"     Volume reduction:         {c_reduct_mm3:.3f} mm3 ({c_reduct_pct:.2f}%)",
        "",
        "3. CLEARANCE METRICS (Tumour <-> Cochlea)",
        "   [Slab Variant]",
        f"     signed_clearance_mm:      {r.slab_signed_clearance_mm:.4f} mm",
        f"     mesh_clearance_mm:        {r.slab_mesh_clearance_mm:.4f} mm",
        "   [Capped Variant]",
        f"     signed_clearance_mm:      {r.capped_signed_clearance_mm:.4f} mm",
        f"     mesh_clearance_mm:        {r.capped_mesh_clearance_mm:.4f} mm",
        "   [Difference: Slab - Capped]",
        f"     signed_clearance diff:    {r.signed_clearance_diff_mm:+.4f} mm",
        f"     mesh_clearance diff:      {r.mesh_clearance_diff_mm:+.4f} mm",
        sub_sep + "\n",
    ]
    return "\n".join(lines)


def run_prepare() -> list[CaseExportReport]:
    REPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
    all_reports: list[CaseExportReport] = []
    output_text_blocks: list[str] = [
        "TCIA VESTIBULAR-SCHWANNOMA-SEG CASE EXPORT VERIFICATION REPORT",
        "Generated using point-in-prism continuous structure rasteriser on native T2 grid.",
        "Clearance conventions: face-to-face voxel distances (hdt.geometry).\n",
    ]

    for pat in PATIENTS:
        print(f"Exporting case {pat}...")
        report = export_case(
            patient_id=pat,
            vs_series_dir=VS_SERIES_DIR,
            contours_dir=CONTOURS_DIR,
            tfm_matrices_dir=TFM_DIR,
            output_cases_dir=CASES_DIR,
        )
        all_reports.append(report)
        formatted = format_case_report(report)
        output_text_blocks.append(formatted)
        print(formatted)

    full_report_text = "\n".join(output_text_blocks)
    with open(REPORT_LOG, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\nVerification report written to {REPORT_LOG}")
    return all_reports


if __name__ == "__main__":
    run_prepare()
