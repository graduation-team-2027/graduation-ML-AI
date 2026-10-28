from pathlib import Path
from datetime import datetime, timezone
import csv
import cv2
import numpy as np

"""
Actor Gallery Face Recognition - Development Prototype.

- YuNet: face detection
- SFace: face embeddings and matching
- Data augmentation: gallery images only
- Probe and unregistered images remain independent test inputs

Note: This augments reference images. It does not retrain the SFace model.
"""

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PACKAGE_DIR = BASE_DIR / "actor_gallery_test_package"

GALLERY_DIR = PACKAGE_DIR / "gallery"
PROBES_DIR = PACKAGE_DIR / "probes"
UNREGISTERED_DIR = PACKAGE_DIR / "unregistered"
AUGMENTED_GALLERY_DIR = PACKAGE_DIR / "augmented_gallery"

DETECTION_MODEL = BASE_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = BASE_DIR / "face_recognition_sface_2021dec.onnx"

OUTPUT_CSV = BASE_DIR / "actor_gallery_results.csv"
REPORT_FILE = BASE_DIR / "actor_gallery_run_report.txt"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"
}

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MIN_DETECTION_CONFIDENCE = 0.60
MIN_FACE_SIZE = 50

# Initial blur threshold. It may need calibration using your images.
MIN_BLUR_SCORE = 15.0

# Initial SFace cosine similarity settings.
SIMILARITY_THRESHOLD = 0.363
MATCH_MARGIN = 0.05

# Generate six augmented versions for each usable gallery image.
USE_AUGMENTATION = True

CSV_FIELDS = [
    "timestamp_utc",
    "split",
    "image",
    "relative_path",
    "expected_identity",
    "predicted_identity",
    "result",
    "detection_status",
    "detection_confidence",
    "blur_score",
    "best_similarity",
    "match_margin",
    "notes",
]


# ============================================================
# GENERAL UTILITIES
# ============================================================

def timestamp_utc():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def relpath(path):
    if path is None:
        return ""

    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def image_files(folder):
    """
    Find images recursively.

    Generated augmented images are stored outside gallery, but this
    function also skips common generated-output subfolders if present.
    """
    if not folder.exists() or not folder.is_dir():
        return []

    ignored = {
        "augmented",
        "augmented_gallery",
        ".git",
        "__pycache__",
    }

    found = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            relative_parts = path.relative_to(folder).parts[:-1]
        except ValueError:
            relative_parts = ()

        if any(part.lower() in ignored for part in relative_parts):
            continue

        found.append(path)

    return sorted(found)


def person_folders(folder):
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def images_directly_or_in_person_folders(folder):
    """
    Supports unknown images either directly inside unregistered/
    or inside subfolders.
    """
    folders = person_folders(folder)

    if folders:
        result = []

        for person_folder in folders:
            for path in image_files(person_folder):
                result.append((path, person_folder.name))

        return result

    return [
        (path, "UNKNOWN")
        for path in image_files(folder)
    ]


def read_image(path):
    return cv2.imread(str(path))


def make_row(
    split,
    image_path=None,
    expected="",
    predicted="NOT_EVALUATED",
    result="NOT_EVALUATED",
    status="NOT_EVALUATED",
    confidence="",
    blur_score="",
    similarity="",
    margin="",
    notes="",
):
    return {
        "timestamp_utc": timestamp_utc(),
        "split": split,
        "image": image_path.name if image_path else "",
        "relative_path": relpath(image_path),
        "expected_identity": expected,
        "predicted_identity": predicted,
        "result": result,
        "detection_status": status,
        "detection_confidence": (
            round(float(confidence), 4)
            if confidence != ""
            else ""
        ),
        "blur_score": (
            round(float(blur_score), 2)
            if blur_score != ""
            else ""
        ),
        "best_similarity": (
            round(float(similarity), 4)
            if similarity != ""
            else ""
        ),
        "match_margin": (
            round(float(margin), 4)
            if margin != ""
            else ""
        ),
        "notes": notes,
    }


def write_csv(rows):
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_report(lines):
    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    missing = [
        str(path)
        for path in (DETECTION_MODEL, RECOGNITION_MODEL)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model file(s):\n  "
            + "\n  ".join(missing)
            + "\nPlace both ONNX files beside actor_gallery_test.py."
        )

    detector = cv2.FaceDetectorYN.create(
        str(DETECTION_MODEL),
        "",
        (320, 320),
        MIN_DETECTION_CONFIDENCE,
        0.3,
        5000,
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(RECOGNITION_MODEL),
        "",
    )

    return detector, recognizer


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image, detector):
    """
    Try different detection thresholds and upscale small images.
    Return the highest-confidence detected face in original coordinates.
    """
    if image is None or image.size == 0:
        return None

    original_height, original_width = image.shape[:2]

    scales = (
        [1.0, 2.0]
        if max(original_height, original_width) < 800
        else [1.0]
    )

    best_face = None
    best_confidence = -1.0

    for scale in scales:
        if scale == 1.0:
            scaled_image = image
        else:
            scaled_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        height, width = scaled_image.shape[:2]
        detector.setInputSize((width, height))

        for threshold in (
            MIN_DETECTION_CONFIDENCE,
            0.40,
            0.25,
            0.10,
        ):
            detector.setScoreThreshold(float(threshold))

            _, faces = detector.detect(scaled_image)

            if faces is None:
                continue

            for face_row in faces:
                face = face_row.copy()
                confidence = float(face[14])

                # Map bounding box and landmarks back to original coordinates.
                face[:14] /= scale

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_face = face

            if best_confidence >= MIN_DETECTION_CONFIDENCE:
                break

    detector.setScoreThreshold(MIN_DETECTION_CONFIDENCE)

    return best_face


# ============================================================
# FACE QUALITY AND EMBEDDING
# ============================================================

def blur_score_for_face(image, face):
    x, y, width, height = [
        int(round(value)) for value in face[:4]
    ]

    image_height, image_width = image.shape[:2]

    x = max(0, x)
    y = max(0, y)

    width = min(width, image_width - x)
    height = min(height, image_height - y)

    if width <= 0 or height <= 0:
        return 0.0

    crop = image[y:y + height, x:x + width]

    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    return float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )


def normalize_feature(feature):
    feature = np.asarray(
        feature,
        dtype=np.float32,
    ).reshape(-1)

    norm = float(np.linalg.norm(feature))

    if norm == 0:
        return None

    return feature / norm


def extract_feature(
    image,
    detector,
    recognizer,
    check_quality=True,
):
    if image is None:
        return None, "IMAGE_READ_ERROR", 0.0, 0.0

    face = detect_face(image, detector)

    if face is None:
        return None, "NO_FACE_DETECTED", 0.0, 0.0

    confidence = float(face[14])
    face_width = int(face[2])
    face_height = int(face[3])

    if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
        return (
            None,
            "INSUFFICIENT_QUALITY",
            confidence,
            0.0,
        )

    blur = blur_score_for_face(image, face)

    if check_quality and blur < MIN_BLUR_SCORE:
        return (
            None,
            "INSUFFICIENT_QUALITY",
            confidence,
            blur,
        )

    try:
        aligned_face = recognizer.alignCrop(
            image,
            face.reshape(1, -1),
        )

        feature = normalize_feature(
            recognizer.feature(aligned_face)
        )

        if feature is None:
            return (
                None,
                "FEATURE_EXTRACTION_ERROR",
                confidence,
                blur,
            )

        return feature, "OK", confidence, blur

    except cv2.error:
        return (
            None,
            "FEATURE_EXTRACTION_ERROR",
            confidence,
            blur,
        )


# ============================================================
# DATA AUGMENTATION - GALLERY IMAGES ONLY
# ============================================================

def rotate_image(image, degrees):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        degrees,
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def create_augmentation_variants(image):
    """
    Create mild variations from one reference image.
    These augmentations do not create real side-angle face photographs.
    """
    return {
        "brightness_up": cv2.convertScaleAbs(
            image,
            alpha=1.08,
            beta=8,
        ),
        "brightness_down": cv2.convertScaleAbs(
            image,
            alpha=0.92,
            beta=-8,
        ),
        "contrast_up": cv2.convertScaleAbs(
            image,
            alpha=1.10,
            beta=0,
        ),
        "rotate_left_4deg": rotate_image(
            image,
            -4,
        ),
        "rotate_right_4deg": rotate_image(
            image,
            4,
        ),
        "mild_blur": cv2.GaussianBlur(
            image,
            (3, 3),
            0.35,
        ),
    }


def build_gallery(detector, recognizer):
    """
    Build one normalized reference vector per enrolled identity.
    Original and successfully processed augmented features are combined.
    """
    gallery = {}
    augmentation_count = 0
    original_count = 0
    gallery_rows = []

    for person_folder in person_folders(GALLERY_DIR):
        person_id = person_folder.name
        features = []

        originals = image_files(person_folder)

        if not originals:
            print(
                f"[WARNING] No images for enrolled ID: {person_id}"
            )
            continue

        for image_path in originals:
            image = read_image(image_path)

            feature, status, confidence, blur = extract_feature(
                image,
                detector,
                recognizer,
                check_quality=True,
            )

            if status != "OK" or feature is None:
                print(
                    f"[GALLERY SKIPPED] {person_id}/{image_path.name}: "
                    f"{status} (blur={blur:.2f})"
                )

                gallery_rows.append(
                    make_row(
                        "gallery",
                        image_path,
                        expected=person_id,
                        predicted="NOT_ENROLLED",
                        result="REFERENCE_SKIPPED",
                        status=status,
                        confidence=confidence,
                        blur_score=blur,
                        notes=(
                            "Could not create a usable reference embedding."
                        ),
                    )
                )

                continue

            original_count += 1
            features.append(feature)

            gallery_rows.append(
                make_row(
                    "gallery",
                    image_path,
                    expected=person_id,
                    predicted=person_id,
                    result="ENROLLED_REFERENCE",
                    status="OK",
                    confidence=confidence,
                    blur_score=blur,
                    notes=(
                        "Reference image enrolled; not counted as an "
                        "independent test."
                    ),
                )
            )

            if not USE_AUGMENTATION:
                continue

            variants = create_augmentation_variants(image)

            for variant_name, variant in variants.items():
                output_path = (
                    AUGMENTED_GALLERY_DIR
                    / person_id
                    / f"{image_path.stem}_{variant_name}.jpg"
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if not cv2.imwrite(
                    str(output_path),
                    variant,
                ):
                    print(
                        f"[WARNING] Could not save augmentation: "
                        f"{output_path}"
                    )
                    continue

                augmentation_count += 1

                augmented_feature, augmented_status, _, _ = (
                    extract_feature(
                        variant,
                        detector,
                        recognizer,
                        check_quality=False,
                    )
                )

                if (
                    augmented_status == "OK"
                    and augmented_feature is not None
                ):
                    features.append(augmented_feature)

        if features:
            centroid = normalize_feature(
                np.mean(
                    np.vstack(features),
                    axis=0,
                )
            )

            if centroid is not None:
                gallery[person_id] = centroid

                print(
                    f"[ENROLLED] {person_id}: "
                    f"{len(features)} usable feature(s), "
                    "including usable augmentations"
                )

    return (
        gallery,
        gallery_rows,
        original_count,
        augmentation_count,
    )


# ============================================================
# ID MATCHING
# ============================================================

def cosine_similarity(feature_a, feature_b):
    a = normalize_feature(feature_a)
    b = normalize_feature(feature_b)

    if a is None or b is None:
        return -1.0

    return float(np.dot(a, b))


def match_identity(feature, gallery):
    scores = sorted(
        [
            (
                person_id,
                cosine_similarity(feature, reference),
            )
            for person_id, reference in gallery.items()
        ],
        key=lambda pair: pair[1],
        reverse=True,
    )

    if not scores:
        return "UNKNOWN", 0.0, 0.0

    best_id, best_score = scores[0]

    second_score = (
        scores[1][1]
        if len(scores) > 1
        else -1.0
    )

    margin = best_score - second_score

    if best_score < SIMILARITY_THRESHOLD:
        return "UNKNOWN", best_score, margin

    if len(scores) > 1 and margin < MATCH_MARGIN:
        return "UNKNOWN", best_score, margin

    return best_id, best_score, margin


# ============================================================
# PROBE EVALUATION
# ============================================================

def evaluate_image(
    image_path,
    split,
    expected,
    gallery,
    detector,
    recognizer,
):
    image = read_image(image_path)

    feature, status, confidence, blur = extract_feature(
        image,
        detector,
        recognizer,
        check_quality=True,
    )

    if status != "OK" or feature is None:
        predicted = "NOT_EVALUATED"
        result = status
        similarity = ""
        margin = ""

    else:
        predicted, similarity, margin = match_identity(
            feature,
            gallery,
        )

        if expected == "UNKNOWN":
            result = (
                "CORRECT_REJECTION"
                if predicted == "UNKNOWN"
                else "WRONG_MATCH"
            )

        elif predicted == expected:
            result = "CORRECT_MATCH"

        elif predicted == "UNKNOWN":
            result = "MISSED_MATCH"

        else:
            result = "WRONG_MATCH"

    row = make_row(
        split,
        image_path,
        expected=expected,
        predicted=predicted,
        result=result,
        status=status,
        confidence=confidence,
        blur_score=blur,
        similarity=similarity,
        margin=margin,
    )

    print(
        f"[{split}] {image_path.name} -> {predicted} | "
        f"{result} | status={status} | blur={blur:.2f}"
    )

    return row


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== ACTOR GALLERY FACE RECOGNITION ==========")
    print(f"Project folder: {BASE_DIR}")
    print(f"Gallery folder: {GALLERY_DIR}")
    print(f"Probes folder: {PROBES_DIR}")
    print(f"Unregistered folder: {UNREGISTERED_DIR}")

    if not GALLERY_DIR.exists():
        raise FileNotFoundError(
            f"Create the gallery folder first: {GALLERY_DIR}"
        )

    if not person_folders(GALLERY_DIR):
        raise RuntimeError(
            "No enrolled person folders found. Create one folder per "
            "person, for example gallery/person_001/ref_01.jpg."
        )

    detector, recognizer = load_models()

    (
        gallery,
        gallery_rows,
        original_count,
        augmentation_count,
    ) = build_gallery(
        detector,
        recognizer,
    )

    if not gallery:
        write_csv(gallery_rows)

        raise RuntimeError(
            "No usable face references were enrolled. Check that gallery "
            "images contain a visible face and try a clearer, larger image. "
            "See actor_gallery_results.csv."
        )

    rows = list(gallery_rows)
    limitations = []

    # Registered people: probe images must be different from gallery images.
    probe_people = person_folders(PROBES_DIR)
    gallery_ids = set(gallery.keys())
    usable_registered_probe_count = 0

    for folder in probe_people:
        person_id = folder.name

        if person_id not in gallery_ids:
            limitations.append(
                f"Probe folder {person_id} has no matching enrolled "
                "gallery identity; skipped."
            )
            continue

        for image_path in image_files(folder):
            row = evaluate_image(
                image_path,
                "registered_probe",
                person_id,
                gallery,
                detector,
                recognizer,
            )

            rows.append(row)

            if row["result"] in {
                "CORRECT_MATCH",
                "MISSED_MATCH",
                "WRONG_MATCH",
            }:
                usable_registered_probe_count += 1

    if usable_registered_probe_count == 0:
        limitations.append(
            "No independent probe images for registered people were "
            "evaluated. Gallery photos are reference images, not "
            "independent recognition tests. Add separate photos under "
            "probes/<same_person_id>/."
        )

    # Unregistered people: expected identity is UNKNOWN.
    unknown_items = images_directly_or_in_person_folders(
        UNREGISTERED_DIR
    )

    if not unknown_items:
        limitations.append(
            "No unregistered images found. Add the two unknown photos "
            "under unregistered/."
        )

    else:
        for image_path, folder_id in unknown_items:
            rows.append(
                evaluate_image(
                    image_path,
                    "unregistered_probe",
                    "UNKNOWN",
                    gallery,
                    detector,
                    recognizer,
                )
            )

    write_csv(rows)

    summary = {}

    for row in rows:
        summary[row["result"]] = (
            summary.get(row["result"], 0) + 1
        )

    report_lines = [
        "ACTOR GALLERY FACE RECOGNITION - RUN REPORT",
        f"Run time (UTC): {timestamp_utc()}",
        f"Registered identities enrolled: {len(gallery)}",
        f"Usable original gallery images: {original_count}",
        f"Augmented gallery images saved this run: {augmentation_count}",
        f"Augmented gallery path: {AUGMENTED_GALLERY_DIR}",
        (
            "Registered probe images independently evaluated: "
            f"{usable_registered_probe_count}"
        ),
        f"Unregistered images found: {len(unknown_items)}",
        "",
        "RESULT COUNTS:",
    ]

    report_lines.extend(
        f"- {label}: {count}"
        for label, count in sorted(summary.items())
    )

    if limitations:
        report_lines.extend(["", "LIMITATIONS:"])
        report_lines.extend(
            f"- {item}"
            for item in limitations
        )

    report_lines.extend([
        "",
        "Data augmentation was applied to gallery/reference images only.",
        "Generated copies are not independent test images.",
        "The SFace neural network is not retrained by this script.",
        (
            "A reference listed as ENROLLED_REFERENCE is not counted "
            "as an independent recognition success."
        ),
        (
            "Do not upload real face images or personally identifying "
            "result files to a public GitHub repository."
        ),
    ])

    write_report(report_lines)

    print("\n========== SUMMARY ==========")
    print(f"Registered identities enrolled: {len(gallery)}")
    print(f"Original gallery images used: {original_count}")
    print(f"Augmented gallery images saved: {augmentation_count}")

    for label, count in sorted(summary.items()):
        print(f"{label}: {count}")

    if limitations:
        print("\nWHAT IS STILL MISSING:")

        for item in limitations:
            print(f"- {item}")

    print(f"\nCSV results: {OUTPUT_CSV}")
    print(f"Run report: {REPORT_FILE}")

    print(
        "\nNote: similarity decisions are experimental and must be "
        "validated on separate images."
    )


if __name__ == "__main__":
    main()