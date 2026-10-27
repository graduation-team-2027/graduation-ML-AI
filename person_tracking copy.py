
import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO

# =========================================================
# 1. Settings
# =========================================================

VIDEO_PATH = "testvedio.mp4"
YOLO_MODEL_PATH = "yolo26n.pt"
FACE_MODEL_PATH = "face_detection_yunet.onnx"
TRACKER_PATH = "bytetrack.yaml"

OUTPUT_VIDEO_PATH = "tracked_output.mp4"
TRACKING_CSV_PATH = "tracking_results.csv"
FACE_QUALITY_CSV_PATH = "face_quality_results.csv"
FACE_CROPS_FOLDER = "face_crops"

# Face-quality thresholds; these are initial values to test.
MIN_FACE_WIDTH = 40
MIN_FACE_HEIGHT = 40
MIN_BLUR_VARIANCE = 40.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0

# Limit the face detector's working image size.
MAX_FACE_PROCESSING_SIZE = 640

# =========================================================
# 2. Check required files
# =========================================================

for required_file in (
    VIDEO_PATH,
    FACE_MODEL_PATH,
):
    if not os.path.isfile(required_file):
        print(f"ERROR: Required file not found: {required_file}")
        raise SystemExit(1)

# =========================================================
# 3. Load models
# =========================================================

print("Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

print("Loading YuNet face detector...")
face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# =========================================================
# 4. Open input video
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open the input video.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or width <= 0 or height <= 0:
    print("ERROR: Invalid video properties.")
    cap.release()
    raise SystemExit(1)

# =========================================================
# 5. Prepare output video
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)

if not video_writer.isOpened():
    print("ERROR: Could not create the output video.")
    cap.release()
    raise SystemExit(1)

os.makedirs(FACE_CROPS_FOLDER, exist_ok=True)

# =========================================================
# 6. Store tracking and face-quality information
# =========================================================

people = {}
face_quality = {}

frame_number = 0

print(f"Input video: {VIDEO_PATH}")
print(f"Video size: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print("Tracking and face-quality checks started.")
print("Press Q to stop early.")

# =========================================================
# 7. Face-quality helper function
# =========================================================

def assess_face_quality(face_crop):
    """Return face size, blur, brightness and quality status."""

    if face_crop is None or face_crop.size == 0:
        return {
            "face_width": 0,
            "face_height": 0,
            "blur_variance": 0.0,
            "brightness": 0.0,
            "quality_status": "Insufficient Quality"
        }

    face_height, face_width = face_crop.shape[:2]

    gray_face = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    blur_variance = float(
        cv2.Laplacian(gray_face, cv2.CV_64F).var()
    )

    brightness = float(np.mean(gray_face))

    if (
        face_width < MIN_FACE_WIDTH
        or face_height < MIN_FACE_HEIGHT
        or blur_variance < MIN_BLUR_VARIANCE
        or brightness < MIN_BRIGHTNESS
        or brightness > MAX_BRIGHTNESS
    ):
        status = "Insufficient Quality"
    else:
        status = "Quality OK"

    return {
        "face_width": face_width,
        "face_height": face_height,
        "blur_variance": round(blur_variance, 2),
        "brightness": round(brightness, 2),
        "quality_status": status
    }

# =========================================================
# 8. Main tracking loop
# =========================================================

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1
        current_time = (frame_number - 1) / fps

        # Track people only (COCO class 0).
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=TRACKER_PATH,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Keep coordinates within the image.
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))

                # -----------------------------------------
                # A. Update original tracking information
                # -----------------------------------------

                if track_id not in people:
                    people[track_id] = {
                        "first_seen_frame": frame_number,
                        "last_seen_frame": frame_number,
                        "first_seen_seconds": current_time,
                        "last_seen_seconds": current_time,
                    }
                else:
                    people[track_id]["last_seen_frame"] = frame_number
                    people[track_id]["last_seen_seconds"] = current_time

                # Draw original person tracking box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person ID: {track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # -----------------------------------------
                # B. Detect a face inside THIS person's box
                # -----------------------------------------

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                crop_height, crop_width = person_crop.shape[:2]

                scale = min(
                    1.0,
                    MAX_FACE_PROCESSING_SIZE / max(
                        crop_width, crop_height
                    )
                )

                processing_width = max(
                    1, int(crop_width * scale)
                )
                processing_height = max(
                    1, int(crop_height * scale)
                )

                small_person_crop = cv2.resize(
                    person_crop,
                    (processing_width, processing_height)
                )

                face_detector.setInputSize(
                    (processing_width, processing_height)
                )

                _, detected_faces = face_detector.detect(
                    small_person_crop
                )

                # Create one summary record per tracking ID.
                if track_id not in face_quality:
                    face_quality[track_id] = {
                        "face_detections": 0,
                        "quality_ok_count": 0,
                        "insufficient_count": 0,
                        "best_quality_score": -1.0,
                        "best_face_width": 0,
                        "best_face_height": 0,
                        "best_blur_variance": 0.0,
                        "best_brightness": 0.0,
                        "best_quality_status": "No Face Detected",
                        "best_face_frame": "",
                        "best_face_seconds": "",
                        "best_face_path": ""
                    }

                face_data = face_quality[track_id]

                if detected_faces is None or len(detected_faces) == 0:
                    # Do not guess a face or identity when no face
                    # is visible in this person's tracking box.
                    cv2.putText(
                        frame,
                        "No Face Detected",
                        (x1, min(y2 + 22, height - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )
                    continue

                for detected_face in detected_faces:
                    fx, fy, fw, fh = detected_face[:4]

                    # Convert detector coordinates back to the
                    # original person's crop coordinates.
                    fx1 = max(0, int(fx / scale))
                    fy1 = max(0, int(fy / scale))
                    fx2 = min(crop_width, int((fx + fw) / scale))
                    fy2 = min(crop_height, int((fy + fh) / scale))

                    if fx2 <= fx1 or fy2 <= fy1:
                        continue

                    # This face crop comes only from the current
                    # person's own tracking box.
                    face_crop = person_crop[fy1:fy2, fx1:fx2].copy()

                    quality = assess_face_quality(face_crop)

                    face_data["face_detections"] += 1

                    if quality["quality_status"] == "Quality OK":
                        face_data["quality_ok_count"] += 1
                        face_color = (0, 255, 0)
                    else:
                        face_data["insufficient_count"] += 1
                        face_color = (0, 165, 255)

                    # Face coordinates on the original frame.
                    face_x1 = x1 + fx1
                    face_y1 = y1 + fy1
                    face_x2 = x1 + fx2
                    face_y2 = y1 + fy2

                    cv2.rectangle(
                        frame,
                        (face_x1, face_y1),
                        (face_x2, face_y2),
                        face_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        quality["quality_status"],
                        (face_x1, max(face_y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        face_color,
                        2,
                        cv2.LINE_AA
                    )

                    # Keep the highest blur score among faces
                    # that pass the initial quality checks.
                    if quality["quality_status"] == "Quality OK":
                        score = quality["blur_variance"]

                        if score > face_data["best_quality_score"]:
                            face_filename = (
                                f"track_{track_id}_best_face.jpg"
                            )

                            face_path = os.path.join(
                                FACE_CROPS_FOLDER,
                                face_filename
                            )

                            saved = cv2.imwrite(
                                face_path,
                                face_crop
                            )

                            if saved:
                                face_data["best_quality_score"] = score
                                face_data["best_face_width"] = quality["face_width"]
                                face_data["best_face_height"] = quality["face_height"]
                                face_data["best_blur_variance"] = quality["blur_variance"]
                                face_data["best_brightness"] = quality["brightness"]
                                face_data["best_quality_status"] = "Quality OK"
                                face_data["best_face_frame"] = frame_number
                                face_data["best_face_seconds"] = round(current_time, 3)
                                face_data["best_face_path"] = face_path

                    elif face_data["best_quality_score"] < 0:
                        # Remember that a face was found, but no
                        # sufficiently clear crop has been found yet.
                        face_data["best_quality_status"] = (
                            "Insufficient Quality"
                        )

        # Save every processed frame to the output video.
        video_writer.write(frame)

        # Display a smaller preview for high-resolution videos.
        preview_width = min(1000, width)
        preview_height = max(
            1, int(height * preview_width / width)
        )

        preview = cv2.resize(
            frame,
            (preview_width, preview_height)
        )

        cv2.imshow(
            "YOLO Tracking and Face Quality - Press Q to stop",
            preview
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# =========================================================
# 9. Save original tracking CSV (same columns as before)
# =========================================================

with open(
    TRACKING_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "track_id",
        "first_seen_frame",
        "last_seen_frame",
        "first_seen_seconds",
        "last_seen_seconds",
        "observed_duration_seconds"
    ])

    for track_id, data in sorted(people.items()):
        duration = (
            data["last_seen_seconds"]
            - data["first_seen_seconds"]
        )

        writer.writerow([
            track_id,
            data["first_seen_frame"],
            data["last_seen_frame"],
            round(data["first_seen_seconds"], 3),
            round(data["last_seen_seconds"], 3),
            round(duration, 3)
        ])

# =========================================================
# 10. Save face-quality summary CSV
# =========================================================

with open(
    FACE_QUALITY_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    fieldnames = [
        "track_id",
        "face_detections",
        "quality_ok_count",
        "insufficient_count",
        "best_quality_status",
        "best_face_width",
        "best_face_height",
        "best_blur_variance",
        "best_brightness",
        "best_face_frame",
        "best_face_seconds",
        "best_face_path"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for track_id, data in sorted(face_quality.items()):
        writer.writerow({
            "track_id": track_id,
            "face_detections": data["face_detections"],
            "quality_ok_count": data["quality_ok_count"],
            "insufficient_count": data["insufficient_count"],
            "best_quality_status": data["best_quality_status"],
            "best_face_width": data["best_face_width"],
            "best_face_height": data["best_face_height"],
            "best_blur_variance": data["best_blur_variance"],
            "best_brightness": data["best_brightness"],
            "best_face_frame": data["best_face_frame"],
            "best_face_seconds": data["best_face_seconds"],
            "best_face_path": data["best_face_path"]
        })

# =========================================================
# 11. Final report
# =========================================================

print("\nProcessing finished.")
print("Number of tracked IDs:", len(people))
print(f"Annotated video: {OUTPUT_VIDEO_PATH}")
print(f"Tracking results: {TRACKING_CSV_PATH}")
print(f"Face-quality results: {FACE_QUALITY_CSV_PATH}")
print(f"Best face crops folder: {FACE_CROPS_FOLDER}")

print(
    "\nNote: observed_duration_seconds is the time between "
    "the first and last observation for each tracking ID."
)
print(
    "Face detection can fail when a face is turned away, "
    "blurred, hidden, or too small."
)
print(
    "Quality thresholds are initial settings and may need "
    "adjustment after inspecting the results."
)
print(
    "A tracking ID is not a verified real-world identity."
)
