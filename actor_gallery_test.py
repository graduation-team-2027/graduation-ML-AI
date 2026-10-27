from pathlib import Path
import csv
import cv2
import numpy as np
from datetime import datetime, timezone

# ============================================================
# PROJECT CONFIGURATION
# ============================================================
# Keep both values False until you have confirmed that:
# 1) every enrolled participant is an adult and has given informed consent;
# 2) the probe images were captured in a separate session from gallery images.
# When either value is False, this script will use synthetic fixtures only.
CONSENT_CONFIRMED = False
SEPARATE_SESSION_PROBES_CONFIRMED = False

BASE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = BASE_DIR / "actor_gallery_test_package"
GALLERY_DIR = PACKAGE_DIR / "gallery"
PROBES_DIR = PACKAGE_DIR / "probes"
UNREGISTERED_DIR = PACKAGE_DIR / "unregistered"
AUGMENTED_GALLERY_DIR = PACKAGE_DIR / "augmented_gallery"
FIXTURES_DIR = PACKAGE_DIR / "fixtures"

DETECTION_MODEL = BASE_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = BASE_DIR / "face_recognition_sface_2021dec.onnx"

OUTPUT_CSV = BASE_DIR / "actor_gallery_results.csv"
BLOCKER_REPORT = BASE_DIR / "collection_blocker.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}

MIN_VOLUNTEERS = 3
MAX_VOLUNTEERS = 5
MIN_UNREGISTERED_PEOPLE = 1
MAX_UNREGISTERED_PEOPLE = 2

MIN_DETECTION_CONFIDENCE = 0.60
MIN_FACE_SIZE = 50
# Starting point only. Calibrate this using consented validation images from your camera.
MIN_BLUR_SCORE = 15.0

# OpenCV Zoo's commonly documented SFace cosine threshold. Tune with a separate
# validation set for your actual camera/data before relying on results.
SIMILARITY_THRESHOLD = 0.363
# Reject ambiguous top matches when the best score is too close to the runner-up.
MATCH_MARGIN = 0.05

CSV_FIELDS = [
    "timestamp_utc",
    "mode",
    "image",
    "relative_path",
    "split",
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
# FILE / IMAGE HELPERS
# ============================================================
def image_files(folder):
    """Find images recursively inside one person's folder, excluding generated augmentation folders."""
    if not folder.exists() or not folder.is_dir():
        return []
    ignored_folder_names = {"augmented", "augmented_gallery", ".git", "__pycache__"}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.lower() in ignored_folder_names for part in p.relative_to(folder).parts[:-1])
    )


def person_folders(folder):
    """Return immediate subfolders representing person IDs."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def unregistered_folders():
    """Return one folder per unregistered person; support legacy images directly in unregistered/."""
    folders = person_folders(UNREGISTERED_DIR)
    if folders:
        return folders
    # Backward compatibility: images directly inside unregistered/ count as one unknown person.
    if image_files(UNREGISTERED_DIR):
        return [UNREGISTERED_DIR]
    return []


def read_image(path):
    return cv2.imread(str(path))


def relative_path(path):
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def current_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_result_row(
    mode,
    image="",
    path="",
    split="",
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
        "timestamp_utc": current_timestamp(),
        "mode": mode,
        "image": image,
        "relative_path": relative_path(path) if path else "",
        "split": split,
        "expected_identity": expected,
        "predicted_identity": predicted,
        "result": result,
        "detection_status": status,
        "detection_confidence": round(float(confidence), 4) if confidence != "" else "",
        "blur_score": round(float(blur_score), 2) if blur_score != "" else "",
        "best_similarity": round(float(similarity), 4) if similarity != "" else "",
        "match_margin": round(float(margin), 4) if margin != "" else "",
        "notes": notes,
    }


def write_csv(rows):
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# FIXTURE-ONLY MODE
# ============================================================
def create_synthetic_fixture(path, index):
    """Create an abstract non-human image; this is not a face or person image."""
    height, width = 360, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # A simple color gradient plus geometric shapes. Avoid face-like arrangements.
    for y in range(height):
        image[y, :, 0] = 35 + int(35 * y / height)
        image[y, :, 1] = 45 + int(45 * y / height)
        image[y, :, 2] = 55 + int(30 * y / height)

    cv2.rectangle(image, (35, 55), (210, 110), (160, 190, 90), -1)
    cv2.rectangle(image, (260, 65), (590, 95), (95, 135, 190), -1)
    cv2.circle(image, (115, 245), 42, (180, 105, 70), -1)
    cv2.line(image, (230, 160), (585, 280), (220, 220, 220), 5)
    cv2.putText(
        image,
        "SYNTHETIC FIXTURE ONLY",
        (35, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"NO CONSENTED FACE DATA - {index}",
        (35, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def run_fixtures_only(blockers):
    """Create fixtures and record the collection blocker without using real photos."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    reasons = list(dict.fromkeys(blockers))
    reason_text = "\n".join(f"- {reason}" for reason in reasons)

    report_text = (
        "COLLECTION BLOCKER: real-person face recognition was not run.\n\n"
        f"Reason(s):\n{reason_text}\n\n"
        "Current action: synthetic non-face fixtures only. No identity result is claimed.\n"
        "Next steps: obtain consent from 3-5 adult volunteers; store gallery images by generic ID; "
        "capture probes in a separate session; add 1-2 unregistered volunteer IDs; then set both "
        "confirmation flags to True in actor_gallery_test.py.\n"
        "Do not commit real face images, face embeddings, or personally identifying results to a public repository.\n"
    )
    BLOCKER_REPORT.write_text(report_text, encoding="utf-8")

    rows = []
    for index in range(1, 4):
        fixture_path = FIXTURES_DIR / f"synthetic_fixture_{index:03d}.png"
        create_synthetic_fixture(fixture_path, index)
        rows.append(
            make_result_row(
                mode="FIXTURE_ONLY",
                image=fixture_path.name,
                path=fixture_path,
                split="fixture",
                expected="",
                predicted="NOT_EVALUATED",
                result="COLLECTION_BLOCKER",
                status="SYNTHETIC_FIXTURE_ONLY",
                notes="; ".join(reasons),
            )
        )

    write_csv(rows)
    print("\n========== FIXTURE-ONLY MODE ==========")
    print("No real-person photos were used.")
    for reason in reasons:
        print(f"COLLECTION BLOCKER: {reason}")
    print(f"Synthetic fixtures: {FIXTURES_DIR}")
    print(f"Blocker report: {BLOCKER_REPORT}")
    print(f"CSV report: {OUTPUT_CSV}")
    print("No face-recognition accuracy is claimed in fixture-only mode.")


# ============================================================
# DATASET VALIDATION
# ============================================================
def validate_dataset_structure():
    blockers = []

    if not GALLERY_DIR.exists():
        blockers.append(f"Gallery folder is missing: {GALLERY_DIR}")
        return blockers
    if not PROBES_DIR.exists():
        blockers.append(f"Probe folder is missing: {PROBES_DIR}")
    if not UNREGISTERED_DIR.exists():
        blockers.append(f"Unregistered folder is missing: {UNREGISTERED_DIR}")

    gallery_people = person_folders(GALLERY_DIR)
    probe_people = person_folders(PROBES_DIR)
    unregistered_people = unregistered_folders()

    gallery_ids = {p.name for p in gallery_people}
    probe_ids = {p.name for p in probe_people}

    if not MIN_VOLUNTEERS <= len(gallery_people) <= MAX_VOLUNTEERS:
        blockers.append(
            f"Gallery must contain {MIN_VOLUNTEERS}-{MAX_VOLUNTEERS} enrolled adult volunteer folders; "
            f"found {len(gallery_people)}."
        )

    if gallery_ids != probe_ids:
        blockers.append(
            "The person-ID folders under gallery and probes must match exactly. "
            "Each probe folder must contain images collected in a separate session."
        )

    for person_folder in gallery_people:
        if not image_files(person_folder):
            blockers.append(f"No gallery images found for {person_folder.name}.")

    for person_folder in probe_people:
        if not image_files(person_folder):
            blockers.append(f"No separate-session probe images found for {person_folder.name}.")

    if not MIN_UNREGISTERED_PEOPLE <= len(unregistered_people) <= MAX_UNREGISTERED_PEOPLE:
        blockers.append(
            f"Create {MIN_UNREGISTERED_PEOPLE}-{MAX_UNREGISTERED_PEOPLE} unregistered person folders; "
            f"found {len(unregistered_people)}."
        )

    for person_folder in unregistered_people:
        if not image_files(person_folder):
            blockers.append(f"No images found for unregistered person {person_folder.name}.")

    return blockers


# ============================================================
# MODEL INITIALIZATION AND FACE DETECTION
# ============================================================
def load_models():
    missing = []
    if not DETECTION_MODEL.exists():
        missing.append(str(DETECTION_MODEL))
    if not RECOGNITION_MODEL.exists():
        missing.append(str(RECOGNITION_MODEL))
    if missing:
        raise FileNotFoundError("Model file(s) not found: " + ", ".join(missing))

    detector = cv2.FaceDetectorYN.create(
        str(DETECTION_MODEL),
        "",
        (320, 320),
        MIN_DETECTION_CONFIDENCE,
        0.3,
        5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNITION_MODEL), "")
    return detector, recognizer


def detect_face_robust(image, detector):
    """Try useful scales/thresholds, returning the strongest detection in original coordinates."""
    if image is None or image.size == 0:
        return None

    original_height, original_width = image.shape[:2]
    max_dimension = max(original_height, original_width)
    scale_options = [1.0]
    if max_dimension < 800:
        scale_options.append(2.0)
    elif max_dimension > 1600:
        scale_options.append(0.75)

    best_face = None
    best_confidence = -1.0

    for scale in scale_options:
        if scale == 1.0:
            scaled_image = image
        else:
            scaled_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            )

        height, width = scaled_image.shape[:2]
        detector.setInputSize((width, height))

        for threshold in (MIN_DETECTION_CONFIDENCE, 0.40, 0.25, 0.10):
            detector.setScoreThreshold(float(threshold))
            _, faces = detector.detect(scaled_image)
            if faces is None or len(faces) == 0:
                continue

            for row in faces:
                face = row.copy()
                confidence = float(face[14])
                face[:14] /= scale
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_face = face

            if best_confidence >= MIN_DETECTION_CONFIDENCE:
                break

    detector.setScoreThreshold(MIN_DETECTION_CONFIDENCE)
    return best_face


def calculate_blur_score(image, face):
    x, y, width, height = [int(round(v)) for v in face[:4]]
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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm == 0:
        return None
    return feature / norm


def extract_face_feature(image, detector, recognizer, check_quality=True):
    if image is None:
        return None, "IMAGE_READ_ERROR", 0.0, 0.0

    face = detect_face_robust(image, detector)
    if face is None:
        return None, "NO_FACE_DETECTED", 0.0, 0.0

    face_width, face_height = int(face[2]), int(face[3])
    confidence = float(face[14])
    if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
        return None, "INSUFFICIENT_QUALITY", confidence, 0.0

    blur_score = calculate_blur_score(image, face)
    if check_quality and blur_score < MIN_BLUR_SCORE:
        return None, "INSUFFICIENT_QUALITY", confidence, blur_score

    try:
        aligned = recognizer.alignCrop(image, face.reshape(1, -1))
        feature = recognizer.feature(aligned)
        feature = normalize_feature(feature)
        if feature is None:
            return None, "FEATURE_EXTRACTION_ERROR", confidence, blur_score
        return feature, "OK", confidence, blur_score
    except cv2.error:
        return None, "FEATURE_EXTRACTION_ERROR", confidence, blur_score


# ============================================================
# DATA AUGMENTATION FOR GALLERY IMAGES ONLY
# ============================================================
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def create_augmentation_variants(image):
    """Mild transforms that preserve identity; the original is handled separately."""
    return {
        "brightness_up": cv2.convertScaleAbs(image, alpha=1.08, beta=8),
        "brightness_down": cv2.convertScaleAbs(image, alpha=0.92, beta=-8),
        "contrast_up": cv2.convertScaleAbs(image, alpha=1.10, beta=0),
        "rotate_left_4deg": rotate_image(image, -4),
        "rotate_right_4deg": rotate_image(image, 4),
        "mild_blur": cv2.GaussianBlur(image, (3, 3), 0.35),
    }


def save_augmentation(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save augmented image: {path}")


def build_gallery(detector, recognizer):
    """Build one normalized centroid per person using originals + mild augmentations."""
    gallery_features = {}
    usable_original_count = 0
    augmented_saved_count = 0

    for person_folder in person_folders(GALLERY_DIR):
        person_id = person_folder.name
        person_features = []
        originals = image_files(person_folder)

        for image_path in originals:
            image = read_image(image_path)
            feature, status, confidence, blur_score = extract_face_feature(
                image, detector, recognizer, check_quality=True
            )

            if status != "OK" or feature is None:
                print(
                    f"[GALLERY SKIPPED] {person_id}/{image_path.name}: "
                    f"{status} (blur={blur_score:.2f})"
                )
                continue

            usable_original_count += 1
            person_features.append(feature)

            # Keep augmented files separate from the original gallery so they are
            # never re-read as originals on the next run.
            variants = create_augmentation_variants(image)
            for variant_name, augmented_image in variants.items():
                save_path = (
                    AUGMENTED_GALLERY_DIR
                    / person_id
                    / f"{image_path.stem}_{variant_name}.jpg"
                )
                save_augmentation(save_path, augmented_image)
                augmented_saved_count += 1

                # Do not quality-reject these intentionally mild transformed
                # references; still require a detected face and a valid embedding.
                aug_feature, aug_status, _, _ = extract_face_feature(
                    augmented_image, detector, recognizer, check_quality=False
                )
                if aug_status == "OK" and aug_feature is not None:
                    person_features.append(aug_feature)

        if person_features:
            centroid = normalize_feature(np.mean(np.vstack(person_features), axis=0))
            if centroid is not None:
                gallery_features[person_id] = centroid
                print(
                    f"[GALLERY] {person_id}: {len(person_features)} feature(s) "
                    "including usable augmentations"
                )

    return gallery_features, usable_original_count, augmented_saved_count


# ============================================================
# MATCHING AND TESTING
# ============================================================
def cosine_similarity(feature_a, feature_b):
    a = normalize_feature(feature_a)
    b = normalize_feature(feature_b)
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))


def match_identity(feature, gallery_features):
    scores = [
        (person_id, cosine_similarity(feature, person_feature))
        for person_id, person_feature in gallery_features.items()
    ]
    if not scores:
        return "UNKNOWN", 0.0, 0.0

    scores.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    if best_score < SIMILARITY_THRESHOLD or margin < MATCH_MARGIN:
        return "UNKNOWN", best_score, margin
    return best_id, best_score, margin


def evaluate_folder(folder, split, expected_identity, gallery_features, detector, recognizer):
    rows = []
    paths = image_files(folder)
    usable_images = 0

    for image_path in paths:
        image = read_image(image_path)
        feature, status, confidence, blur_score = extract_face_feature(
            image, detector, recognizer, check_quality=True
        )

        if status != "OK" or feature is None:
            predicted = "NOT_EVALUATED"
            result = status
            similarity = ""
            margin = ""
        else:
            usable_images += 1
            predicted, similarity, margin = match_identity(feature, gallery_features)
            if expected_identity == "UNKNOWN":
                result = "CORRECT_REJECTION" if predicted == "UNKNOWN" else "WRONG_MATCH"
            else:
                if predicted == expected_identity:
                    result = "CORRECT_MATCH"
                elif predicted == "UNKNOWN":
                    result = "MISSED_MATCH"
                else:
                    result = "WRONG_MATCH"

        rows.append(
            make_result_row(
                mode="CONSENTED_DEVELOPMENT",
                image=image_path.name,
                path=image_path,
                split=split,
                expected=expected_identity,
                predicted=predicted,
                result=result,
                status=status,
                confidence=confidence,
                blur_score=blur_score,
                similarity=similarity,
                margin=margin,
            )
        )
        print(
            f"[{split}] {image_path.name} -> {predicted} | {result} | "
            f"status={status} | blur={blur_score:.2f}"
        )

    return rows, usable_images



def preflight_validate_usable_images(detector, recognizer):
    """Check that each required person has at least one usable face before augmentation."""
    blockers = []

    checks = []
    for folder in person_folders(GALLERY_DIR):
        checks.append((folder, "gallery", folder.name))
    for folder in person_folders(PROBES_DIR):
        checks.append((folder, "separate-session probe", folder.name))
    for folder in unregistered_folders():
        checks.append((folder, "unregistered probe", "UNKNOWN"))

    for folder, split_name, expected in checks:
        usable_count = 0
        image_paths = image_files(folder)
        if not image_paths:
            blockers.append(f"No images found in {split_name} folder {folder.name}.")
            continue

        for image_path in image_paths:
            image = read_image(image_path)
            feature, status, _, blur_score = extract_face_feature(
                image, detector, recognizer, check_quality=True
            )
            if status == "OK" and feature is not None:
                usable_count += 1
            else:
                print(
                    f"[PREFLIGHT SKIPPED] {split_name}/{image_path.name}: "
                    f"{status} (blur={blur_score:.2f})"
                )

        if usable_count == 0:
            blockers.append(
                f"No usable face image in {split_name} folder {folder.name}; "
                "at least one detected, sufficiently clear face is required per person."
            )

    return blockers

def run_consent_based_test(detector, recognizer):
    gallery_features, original_count, augmentation_count = build_gallery(detector, recognizer)

    if len(gallery_features) < MIN_VOLUNTEERS:
        return None, (
            f"Only {len(gallery_features)} enrolled identities had usable gallery faces; "
            f"need {MIN_VOLUNTEERS}-{MAX_VOLUNTEERS}."
        )

    rows = []

    # Probe images must be from a separate session and are never augmented.
    for person_folder in person_folders(PROBES_DIR):
        person_id = person_folder.name
        person_rows, usable_count = evaluate_folder(
            person_folder, "separate_session_probe", person_id,
            gallery_features, detector, recognizer
        )
        if usable_count == 0:
            return None, f"No usable separate-session probe face for {person_id}."
        rows.extend(person_rows)

    # Each folder under unregistered represents a different unregistered person.
    for person_folder in unregistered_folders():
        person_rows, usable_count = evaluate_folder(
            person_folder, "unregistered_probe", "UNKNOWN",
            gallery_features, detector, recognizer
        )
        if usable_count == 0:
            return None, f"No usable unregistered probe face in {person_folder.name}."
        rows.extend(person_rows)

    if not rows:
        return None, "No probe images were evaluated."

    write_csv(rows)
    if BLOCKER_REPORT.exists():
        BLOCKER_REPORT.unlink()

    print("\n========== FACE RECOGNITION SUMMARY ==========")
    summary = {}
    for row in rows:
        summary[row["result"]] = summary.get(row["result"], 0) + 1
    for label, count in sorted(summary.items()):
        print(f"{label}: {count}")

    print(f"Original gallery images usable: {original_count}")
    print(f"Augmented gallery images saved: {augmentation_count}")
    print(f"Augmented image folder: {AUGMENTED_GALLERY_DIR}")
    print(f"Results CSV: {OUTPUT_CSV}")
    print("Probe images were not augmented; they remain an independent test set.")
    print("These development results are not proof of identity or a production accuracy claim.")
    return rows, None


# ============================================================
# MAIN
# ============================================================
def main():
    blockers = []

    # Do not even inspect real-person folders while the manual confirmations are off.
    if not CONSENT_CONFIRMED:
        blockers.append(
            "CONSENT_CONFIRMED is False. Confirm adult status and informed consent for all enrolled volunteers first."
        )
    if not SEPARATE_SESSION_PROBES_CONFIRMED:
        blockers.append(
            "SEPARATE_SESSION_PROBES_CONFIRMED is False. Probe images must be captured in a separate session from gallery images."
        )

    if blockers:
        run_fixtures_only(blockers)
        return

    blockers = validate_dataset_structure()
    if blockers:
        run_fixtures_only(blockers)
        return

    try:
        detector, recognizer = load_models()
    except Exception as exc:
        run_fixtures_only([f"Could not load the face models: {exc}"])
        return

    preflight_blockers = preflight_validate_usable_images(detector, recognizer)
    if preflight_blockers:
        run_fixtures_only(preflight_blockers)
        return

    try:
        rows, blocker = run_consent_based_test(detector, recognizer)
    except Exception as exc:
        rows, blocker = None, f"Runtime validation failed before a complete test could be produced: {exc}"

    if blocker:
        run_fixtures_only([blocker])
        return


if __name__ == "__main__":
    main()
