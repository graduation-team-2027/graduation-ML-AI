
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent

PACKAGE_DIR = BASE_DIR / "actor_gallery_test_package"

GALLERY_DIR = PACKAGE_DIR / "gallery"
PROBES_DIR = PACKAGE_DIR / "probes"
UNREGISTERED_DIR = PACKAGE_DIR / "unregistered"
AUGMENTED_DIR = PACKAGE_DIR / "augmented_gallery"

OUTPUT_FILE = BASE_DIR / "enrollment_probe_manifest.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"
}

FIELDS = [
    "image_id",
    "split",
    "identity_id",
    "expected_identity",
    "relative_path",
    "image_type",
    "parent_reference",
    "capture_session_id",
    "separate_session_status",
    "notes",
]


def find_images(folder):
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return []

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_path(path):
    return str(path.relative_to(BASE_DIR))


def add_row(
    rows,
    path,
    split,
    identity_id="",
    expected_identity="",
    image_type="original",
    parent_reference="",
    session_id="NOT_RECORDED",
    session_status="NOT_VERIFIED",
    notes="",
):
    rows.append({
        "image_id": f"IMG_{len(rows) + 1:03d}",
        "split": split,
        "identity_id": identity_id,
        "expected_identity": expected_identity,
        "relative_path": relative_path(path),
        "image_type": image_type,
        "parent_reference": parent_reference,
        "capture_session_id": session_id,
        "separate_session_status": session_status,
        "notes": notes,
    })


def main():
    rows = []

    # 1. Original enrolled/reference images.
    for path in find_images(GALLERY_DIR):
        relative = path.relative_to(GALLERY_DIR)

        if len(relative.parts) < 2:
            print(f"Skipping gallery image outside person folder: {path}")
            continue

        person_id = relative.parts[0]

        # The gallery uses original images, not generated copies.
        if "augmented" in {part.lower() for part in relative.parts[:-1]}:
            continue

        add_row(
            rows,
            path,
            split="gallery",
            identity_id=person_id,
            expected_identity=person_id,
            image_type="original_reference",
            notes="Used as an enrolled reference image.",
        )

    # 2. Registered-person probe/test images.
    for path in find_images(PROBES_DIR):
        relative = path.relative_to(PROBES_DIR)

        if len(relative.parts) < 2:
            print(f"Skipping probe image outside person folder: {path}")
            continue

        person_id = relative.parts[0]

        add_row(
            rows,
            path,
            split="registered_probe",
            identity_id=person_id,
            expected_identity=person_id,
            image_type="original_probe",
            notes=(
                "Probe image. Verify it is distinct from gallery images "
                "and whether it came from a separate capture session."
            ),
        )

    # 3. Unregistered-person test images.
    for path in find_images(UNREGISTERED_DIR):
        relative = path.relative_to(UNREGISTERED_DIR)

        if len(relative.parts) >= 2:
            image_subject_id = relative.parts[0]
        else:
            image_subject_id = "UNREGISTERED"

        add_row(
            rows,
            path,
            split="unregistered_probe",
            identity_id=image_subject_id,
            expected_identity="UNKNOWN",
            image_type="original_unregistered",
            notes="Expected outcome: reject identity match as UNKNOWN.",
        )

    # 4. Generated augmentation images.
    # These are reference variations, NOT independent test images.
    for path in find_images(AUGMENTED_DIR):
        relative = path.relative_to(AUGMENTED_DIR)

        if len(relative.parts) < 2:
            continue

        person_id = relative.parts[0]

        add_row(
            rows,
            path,
            split="augmented_reference",
            identity_id=person_id,
            expected_identity=person_id,
            image_type="generated_augmentation",
            parent_reference="CHECK_FILENAME_FOR_SOURCE_IMAGE",
            session_id="GENERATED_FROM_REFERENCE",
            session_status="NOT_APPLICABLE",
            notes=(
                "Generated from a gallery image. Do not count as an "
                "independent probe or test image."
            ),
        )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== MANIFEST CREATED ==========")
    print(f"Manifest: {OUTPUT_FILE}")
    print(f"Total image records: {len(rows)}")

    for split in (
        "gallery",
        "registered_probe",
        "unregistered_probe",
        "augmented_reference",
    ):
        count = sum(row["split"] == split for row in rows)
        print(f"{split}: {count}")

    print("\nImportant:")
    print("- Session information is not guessed by this script.")
    print("- Review capture_session_id and separate_session_status.")
    print("- Augmented images are not independent test samples.")
    print("- Original photos and generated results should remain private.")


if __name__ == "__main__":
    main()
