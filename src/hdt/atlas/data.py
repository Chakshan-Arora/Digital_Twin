"""Standard anatomical atlas dataset for Body Explorer.

Anchored to textbook ASIA and Netter dermatome standards.
Simplified educational research model - NOT a clinical diagnostic tool.
"""

from hdt.atlas.models import (
    SPINAL_CORD_LEVELS,
    Atlas,
    BodyZone,
    Condition,
    Dermatome,
    NodeName,
)


def _make_path(peripheral_joints: list[NodeName], cord_root: NodeName) -> list[NodeName]:
    """Generate ascending neural transmission pathway from joints through cord to S1 cortex."""
    start_idx = SPINAL_CORD_LEVELS.index(cord_root.value)
    ascending_cord = [NodeName(lvl) for lvl in SPINAL_CORD_LEVELS[start_idx:]]
    return [
        *peripheral_joints,
        *ascending_cord,
        NodeName.BRAINSTEM,
        NodeName.THALAMUS,
        NodeName.CORTEX_S1,
    ]


DERMATOMES: list[Dermatome] = [
    # -----------------------------------------------------------------------
    # Cervical Dermatomes (C2 - C8)
    # -----------------------------------------------------------------------
    Dermatome(
        id="C2",
        root="C2",
        region="Occiput and posterior scalp",
        body_zone=BodyZone(
            kind="head_neck",
            description="Posterior skull and occipital region",
        ),
        peripheral_nerve="Greater occipital nerve",
        path=_make_path([NodeName.HEAD, NodeName.NECK], NodeName.CORD_C2),
        tract="dorsal column",
        notes="Anatomical landmark: occipital protuberance.",
    ),
    Dermatome(
        id="C3",
        root="C3",
        region="High neck and supraclavicular fossa",
        body_zone=BodyZone(
            kind="head_neck",
            description="Lateral and anterior neck down to clavicle",
        ),
        peripheral_nerve="Supraclavicular nerves",
        path=_make_path([NodeName.NECK], NodeName.CORD_C3),
        tract="spinothalamic",
        notes="Anatomical landmark: supraclavicular fossa.",
    ),
    Dermatome(
        id="C4",
        root="C4",
        region="Shoulder top and clavicular line",
        body_zone=BodyZone(
            kind="head_neck",
            description="Acromioclavicular summit and upper shoulder mantle",
        ),
        peripheral_nerve="Supraclavicular nerves (C3-C4)",
        path=_make_path([NodeName.NECK, NodeName.L_SHOULDER], NodeName.CORD_C4),
        tract="spinothalamic",
        notes="Anatomical landmark: acromioclavicular joint.",
    ),
    Dermatome(
        id="C5",
        root="C5",
        region="Lateral upper arm",
        body_zone=BodyZone(
            kind="arm",
            description="Lateral aspect of arm over deltoid muscle",
            limb_part="lateral upper arm",
        ),
        peripheral_nerve="Axillary nerve (superior lateral cutaneous nerve of arm)",
        path=_make_path([NodeName.L_SHOULDER], NodeName.CORD_C5),
        tract="spinothalamic",
        notes="Anatomical landmark: lateral antecubital fossa.",
    ),
    Dermatome(
        id="C6",
        root="C6",
        region="Thumb and lateral forearm",
        body_zone=BodyZone(
            kind="arm",
            description="Radial forearm and dorsal thumb",
            limb_part="thumb and lateral forearm",
        ),
        peripheral_nerve="Lateral antebrachial cutaneous and radial/median digital nerves",
        path=_make_path(
            [NodeName.L_HAND, NodeName.L_WRIST, NodeName.L_ELBOW, NodeName.L_SHOULDER],
            NodeName.CORD_C6,
        ),
        tract="dorsal column",
        notes="Anatomical landmark: dorsal surface of thumb proximal phalanx.",
    ),
    Dermatome(
        id="C7",
        root="C7",
        region="Middle finger and palm center",
        body_zone=BodyZone(
            kind="arm",
            description="Dorsal and palmar aspect of third digit",
            limb_part="middle finger and palmar center",
        ),
        peripheral_nerve="Median nerve (proper digital branches)",
        path=_make_path(
            [NodeName.L_HAND, NodeName.L_WRIST, NodeName.L_ELBOW, NodeName.L_SHOULDER],
            NodeName.CORD_C7,
        ),
        tract="dorsal column",
        notes="Anatomical landmark: dorsal proximal phalanx of middle finger.",
    ),
    Dermatome(
        id="C8",
        root="C8",
        region="Little finger and medial forearm",
        body_zone=BodyZone(
            kind="arm",
            description="Ulnar forearm and hypothenar/fifth digit",
            limb_part="little finger and medial forearm",
        ),
        peripheral_nerve="Medial antebrachial cutaneous and ulnar nerve",
        path=_make_path(
            [NodeName.L_HAND, NodeName.L_WRIST, NodeName.L_ELBOW, NodeName.L_SHOULDER],
            NodeName.CORD_C8,
        ),
        tract="spinothalamic",
        notes="Anatomical landmark: dorsal proximal phalanx of little finger.",
    ),
    # -----------------------------------------------------------------------
    # Thoracic Dermatomes (T1 - T12)
    # -----------------------------------------------------------------------
    Dermatome(
        id="T1",
        root="T1",
        region="Medial upper arm",
        body_zone=BodyZone(
            kind="arm",
            description="Medial side of the arm proximal to medial epicondyle",
            limb_part="medial arm",
        ),
        peripheral_nerve="Medial brachial cutaneous nerve",
        path=_make_path([NodeName.L_ELBOW, NodeName.L_SHOULDER], NodeName.CORD_T1),
        tract="spinothalamic",
        notes="Anatomical landmark: medial side of antecubital fossa.",
    ),
    Dermatome(
        id="T2",
        root="T2",
        region="Axilla and upper chest",
        body_zone=BodyZone(
            kind="torso_band",
            description="Apex of axilla and upper pectoral line",
            y_fraction=(0.80, 0.83),
        ),
        peripheral_nerve="Intercostobrachial and second intercostal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T2),
        tract="spinothalamic",
        notes="Anatomical landmark: apex of axilla.",
    ),
    Dermatome(
        id="T3",
        root="T3",
        region="Third intercostal space upper thorax",
        body_zone=BodyZone(
            kind="torso_band",
            description="Subclavicular upper rib cage band",
            y_fraction=(0.78, 0.80),
        ),
        peripheral_nerve="Third intercostal nerve anterior cutaneous branch",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T3),
        tract="spinothalamic",
        notes="Anatomical landmark: midclavicular third intercostal space.",
    ),
    Dermatome(
        id="T4",
        root="T4",
        region="Nipple line across thorax",
        body_zone=BodyZone(
            kind="torso_band",
            description="Fourth intercostal space at the nipple level",
            y_fraction=(0.75, 0.78),
        ),
        peripheral_nerve="Fourth intercostal nerve anterior cutaneous branch",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T4),
        tract="spinothalamic",
        notes="Classic key sensory landmark: nipple line (T4).",
    ),
    Dermatome(
        id="T5",
        root="T5",
        region="Inframammary chest band",
        body_zone=BodyZone(
            kind="torso_band",
            description="Fifth intercostal space below the breast",
            y_fraction=(0.72, 0.75),
        ),
        peripheral_nerve="Fifth intercostal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T5),
        tract="spinothalamic",
        notes="Anatomical landmark: midway between nipple line and xiphoid.",
    ),
    Dermatome(
        id="T6",
        root="T6",
        region="Xiphoid process level",
        body_zone=BodyZone(
            kind="torso_band",
            description="Lower sternal border and epigastric entry",
            y_fraction=(0.69, 0.72),
        ),
        peripheral_nerve="Sixth intercostal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T6),
        tract="spinothalamic",
        notes="Key sensory landmark: xiphisternal junction (T6).",
    ),
    Dermatome(
        id="T7",
        root="T7",
        region="Subxiphoid epigastric band",
        body_zone=BodyZone(
            kind="torso_band",
            description="Upper epigastric abdominal wall",
            y_fraction=(0.66, 0.69),
        ),
        peripheral_nerve="Seventh intercostal thoracoabdominal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T7),
        tract="spinothalamic",
        notes="Anatomical landmark: subxiphoid abdominal margin.",
    ),
    Dermatome(
        id="T8",
        root="T8",
        region="Upper abdomen at costal arch",
        body_zone=BodyZone(
            kind="torso_band",
            description="Midway between xiphoid process and umbilicus",
            y_fraction=(0.63, 0.66),
        ),
        peripheral_nerve="Eighth intercostal thoracoabdominal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T8),
        tract="spinothalamic",
        notes="Anatomical landmark: lower costal margin.",
    ),
    Dermatome(
        id="T9",
        root="T9",
        region="Mid-abdominal band above umbilicus",
        body_zone=BodyZone(
            kind="torso_band",
            description="Supraumbilical anterior abdominal wall",
            y_fraction=(0.60, 0.63),
        ),
        peripheral_nerve="Ninth intercostal thoracoabdominal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T9),
        tract="spinothalamic",
        notes="Anatomical landmark: immediately supraumbilical margin.",
    ),
    Dermatome(
        id="T10",
        root="T10",
        region="Umbilicus (belly button) level",
        body_zone=BodyZone(
            kind="torso_band",
            description="Transverse abdominal band passing directly through umbilicus",
            y_fraction=(0.57, 0.60),
        ),
        peripheral_nerve="Tenth intercostal thoracoabdominal nerve",
        path=_make_path([NodeName.CHEST], NodeName.CORD_T10),
        tract="spinothalamic",
        notes="Classic key sensory landmark: umbilicus (T10).",
    ),
    Dermatome(
        id="T11",
        root="T11",
        region="Infraumbilical lower abdomen",
        body_zone=BodyZone(
            kind="torso_band",
            description="Abdominal wall between umbilicus and groin crease",
            y_fraction=(0.54, 0.57),
        ),
        peripheral_nerve="Eleventh intercostal thoracoabdominal nerve",
        path=_make_path([NodeName.PELVIS], NodeName.CORD_T11),
        tract="spinothalamic",
        notes="Anatomical landmark: midway between umbilicus and inguinal ligament.",
    ),
    Dermatome(
        id="T12",
        root="T12",
        region="Inguinal ligament suprapubic crease",
        body_zone=BodyZone(
            kind="torso_band",
            description="Lowest torso band directly overlying inguinal ligament and pubis",
            y_fraction=(0.52, 0.54),
        ),
        peripheral_nerve="Subcostal nerve (T12 anterior ramus)",
        path=_make_path([NodeName.PELVIS], NodeName.CORD_T12),
        tract="spinothalamic",
        notes="Anatomical landmark: midpoint of inguinal ligament crease.",
    ),
    # -----------------------------------------------------------------------
    # Lumbar Dermatomes (L1 - L5)
    # -----------------------------------------------------------------------
    Dermatome(
        id="L1",
        root="L1",
        region="Groin and upper anterior thigh",
        body_zone=BodyZone(
            kind="leg",
            description="Inguinal crease and high anterior-medial thigh",
            limb_part="groin and upper anterior thigh",
        ),
        peripheral_nerve="Ilioinguinal and genitofemoral nerves",
        path=_make_path([NodeName.PELVIS, NodeName.L_HIP], NodeName.CORD_L1),
        tract="spinothalamic",
        notes="Key sensory landmark: upper medial thigh below inguinal crease.",
    ),
    Dermatome(
        id="L2",
        root="L2",
        region="Anterior upper thigh",
        body_zone=BodyZone(
            kind="leg",
            description="Midway down the anterior thigh surface",
            limb_part="anterior mid-thigh",
        ),
        peripheral_nerve="Anterior cutaneous branches of femoral nerve",
        path=_make_path([NodeName.L_HIP, NodeName.PELVIS], NodeName.CORD_L2),
        tract="spinothalamic",
        notes="Key sensory landmark: mid-anterior thigh.",
    ),
    Dermatome(
        id="L3",
        root="L3",
        region="Knee and lower anterior thigh",
        body_zone=BodyZone(
            kind="leg",
            description="Distal anterior thigh across medial patella and joint line",
            limb_part="anterior lower thigh and knee",
        ),
        peripheral_nerve="Femoral nerve (medial cutaneous branch)",
        path=_make_path([NodeName.L_KNEE, NodeName.L_HIP, NodeName.PELVIS], NodeName.CORD_L3),
        tract="spinothalamic",
        notes="Key sensory landmark: medial femoral condyle.",
    ),
    Dermatome(
        id="L4",
        root="L4",
        region="Medial lower leg and medial ankle",
        body_zone=BodyZone(
            kind="leg",
            description="Medial tibia extending down to the medial malleolus",
            limb_part="medial lower leg and medial malleolus",
        ),
        peripheral_nerve="Saphenous nerve (terminal sensory branch of femoral nerve)",
        path=_make_path(
            [NodeName.L_ANKLE, NodeName.L_KNEE, NodeName.L_HIP, NodeName.PELVIS],
            NodeName.CORD_L4,
        ),
        tract="dorsal column",
        notes="Key sensory landmark: medial malleolus.",
    ),
    Dermatome(
        id="L5",
        root="L5",
        region="Lateral lower leg and dorsum of foot",
        body_zone=BodyZone(
            kind="leg",
            description="Anterolateral calf and dorsal aspect of foot to great toe",
            limb_part="lateral lower leg and dorsum of foot",
        ),
        peripheral_nerve="Superficial and deep fibular (peroneal) nerves",
        path=_make_path(
            [NodeName.L_FOOT, NodeName.L_ANKLE, NodeName.L_KNEE, NodeName.L_HIP, NodeName.PELVIS],
            NodeName.CORD_L5,
        ),
        tract="spinothalamic",
        notes="Key sensory landmark: dorsum of third metatarsophalangeal joint and web space.",
    ),
    # -----------------------------------------------------------------------
    # Sacral Dermatomes (S1 - S5)
    # -----------------------------------------------------------------------
    Dermatome(
        id="S1",
        root="S1",
        region="Lateral foot and sole",
        body_zone=BodyZone(
            kind="leg",
            description="Lateral heel, lateral malleolus, and plantar aspect of foot",
            limb_part="lateral foot, heel and sole",
        ),
        peripheral_nerve="Sural nerve and tibial nerve (plantar branches)",
        path=_make_path(
            [NodeName.L_FOOT, NodeName.L_ANKLE, NodeName.L_KNEE, NodeName.L_HIP, NodeName.PELVIS],
            NodeName.CORD_S1,
        ),
        tract="dorsal column",
        notes="Key sensory landmark: lateral calcaneus / lateral malleolus.",
    ),
    Dermatome(
        id="S2",
        root="S2",
        region="Posterior thigh and popliteal fossa",
        body_zone=BodyZone(
            kind="leg",
            description="Midline posterior thigh extending into popliteal space",
            limb_part="posterior thigh and popliteal fossa",
        ),
        peripheral_nerve="Posterior femoral cutaneous nerve",
        path=_make_path([NodeName.L_KNEE, NodeName.L_HIP, NodeName.PELVIS], NodeName.CORD_S2),
        tract="spinothalamic",
        notes="Key sensory landmark: midpoint of popliteal fossa.",
    ),
    Dermatome(
        id="S3",
        root="S3",
        region="Medial buttock and ischial region",
        body_zone=BodyZone(
            kind="leg",
            description="Medial infragluteal crease and ischial tuberosity",
            limb_part="medial buttock and ischial region",
        ),
        peripheral_nerve="Pudendal nerve and cluneal nerves",
        path=_make_path([NodeName.PELVIS], NodeName.CORD_S3),
        tract="spinothalamic",
        notes="Key sensory landmark: ischial tuberosity.",
    ),
    Dermatome(
        id="S4",
        root="S4",
        region="Perineal and perianal saddle region",
        body_zone=BodyZone(
            kind="leg",
            description="Perineum and outer perianal ring",
            limb_part="perineal saddle region",
        ),
        peripheral_nerve="Pudendal nerve (perineal and inferior anal branches)",
        path=_make_path([NodeName.PELVIS], NodeName.CORD_S4),
        tract="spinothalamic",
        notes="Key sensory landmark: perianal mucocutaneous junction.",
    ),
    Dermatome(
        id="S5",
        root="S5",
        region="Innermost perianal region",
        body_zone=BodyZone(
            kind="leg",
            description="Anal verge and immediate perianal skin",
            limb_part="perianal anal verge",
        ),
        peripheral_nerve="Coccygeal and lowest pudendal branches",
        path=_make_path([NodeName.PELVIS], NodeName.CORD_S5),
        tract="spinothalamic",
        notes="Key sensory landmark: anal verge / mucocutaneous ring.",
    ),
]


CONDITIONS: list[Condition] = [
    Condition(
        id="sciatica",
        name="Sciatica (L5/S1 Radiculopathy)",
        description=(
            "Lumbosacral nerve root compression (typically from an L4-L5 or L5-S1 "
            "herniated disc) producing radiating pain down the posterolateral leg."
        ),
        compressed_roots=["L5", "S1"],
        affected_dermatomes=["L5", "S1"],
        symptoms=[
            "Sharp radiating pain from lower back/buttock down posterior calf to sole",
            "Numbness and tingling across the lateral lower leg and foot dorsum",
            "Depressed Achilles tendon jerk (S1) or weakness in foot dorsiflexion (L5)",
        ],
    ),
    Condition(
        id="carpal_tunnel",
        name="Carpal Tunnel Syndrome",
        description=(
            "Median nerve entrapment under the transverse carpal ligament at the wrist, "
            "affecting cutaneous sensory fibers derived primarily from C6 and C7 nerve roots."
        ),
        compressed_roots=["C6", "C7"],
        affected_dermatomes=["C6", "C7"],
        symptoms=[
            "Nocturnal tingling and dysesthesia in thumb, index, and middle fingers",
            "Pain awakening patient from sleep relieved by hand shaking",
            "Thenar eminence muscle wasting and weakness in thumb opposition",
        ],
    ),
    Condition(
        id="cervical_radiculopathy",
        name="Cervical Radiculopathy (C6/C7)",
        description=(
            "Cervical disc herniation or spondylosis compressing the C6 or C7 spinal nerve root, "
            "radiating neuropathic pain and paresthesias down the upper extremity."
        ),
        compressed_roots=["C6", "C7"],
        affected_dermatomes=["C6", "C7"],
        symptoms=[
            "Cervicobrachial burning pain shooting down the arm into hand and fingers",
            "Numbness in the lateral forearm and thumb (C6) or middle finger (C7)",
            "Depressed biceps/brachioradialis (C6) or triceps (C7) reflex",
        ],
    ),
    Condition(
        id="l4_l5_herniation",
        name="L4-L5 Disc Herniation",
        description=(
            "Herniation of the L4-L5 lumbar intervertebral disc impinging upon the traversing "
            "L5 nerve root and occasionally the exiting L4 nerve root."
        ),
        compressed_roots=["L4", "L5"],
        affected_dermatomes=["L4", "L5"],
        symptoms=[
            "Low back pain radiating over the lateral hip, thigh, and anterolateral calf",
            "Sensory deficits over the great toe dorsum and first web space",
            "Weakness in great toe dorsiflexion (extensor hallucis longus)",
        ],
    ),
    Condition(
        id="sci_t10",
        name="Complete Spinal Cord Injury at T10",
        description=(
            "Complete myelopathic lesion or traumatic cord transection at the T10 spinal level, "
            "causing total loss of sensory and motor transmission below the umbilicus."
        ),
        compressed_roots=["T10"],
        affected_dermatomes=[
            "T10",
            "T11",
            "T12",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
        ],
        symptoms=[
            "Bilateral complete loss of light touch, pinprick, and vibration below umbilicus",
            "Paraplegia with loss of voluntary lower limb motor control",
            "Neurogenic bladder and bowel dysfunction",
        ],
    ),
    Condition(
        id="meralgia_paresthetica",
        name="Meralgia Paresthetica",
        description=(
            "Entrapment neuropathy of the lateral femoral cutaneous nerve (L2-L3 posterior "
            "divisions) beneath the inguinal ligament near the anterior superior iliac spine."
        ),
        compressed_roots=["L2", "L3"],
        affected_dermatomes=["L2", "L3"],
        symptoms=[
            "Burning dysesthesia, stinging, and numbness over anterolateral thigh",
            "Hypersensitivity to clothing touching the outer thigh skin",
            "Absence of muscular weakness or reflex impairment (pure sensory neuropathy)",
        ],
    ),
]


ATLAS_DISCLAIMER = (
    "Research prototype educational map. NOT a clinical reference tool, "
    "diagnostic aid, or surgical planning instrument. Never use for clinical "
    "decision-making or patient safety assessment."
)


def get_default_atlas() -> Atlas:
    """Instantiate and return the canonical educational neural atlas."""
    return Atlas(
        disclaimer=ATLAS_DISCLAIMER,
        dermatomes=DERMATOMES,
        conditions=CONDITIONS,
    )
