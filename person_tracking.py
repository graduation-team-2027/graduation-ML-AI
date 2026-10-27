

import cv2
import csv
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# =========================================================
# 1. Project paths and settings
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "testvedio.mp4"
FACE_MODEL_PATH = BASE_DIR / "face_detection_yunet.onnx"

OUTPUT_VIDEO_PATH = BASE_DIR / "tracked_output.mp4"
TRACKING_CSV_PATH = BASE_DIR / "tracking_results.csv"
FACE_QUALITY_CSV_PATH = BASE_DIR / "face_quality_results.csv"
EVENTS_CSV_PATH = BASE_DIR / "tracking_events.csv"
FACE_CROPS_DIR = BASE_DIR / "face_crops"

YOLO_MODEL = "yolo26n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

PROCESSING_WIDTH = 640

MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 50
MIN_FACE_CONFIDENCE = 0.60
MIN_BLUR_SCORE = 30.0

# Number of consecutive missing frames before declaring a track lost.
LOST_AFTER_FRAMES = 15

# Person boxes with IoU at or above this value are considered overlapping.
OVERLAP_IOU_THRESHOLD = 0.10

# A track near the video border may be entering or leaving the scene.
BORDER_MARGIN_RATIO = 0.03

# Avoid writing the same overlap event repeatedly for the same pair
# in consecutive frames.
OVERLAP_EVENT_INTERVAL = 15


# =========================================================
# 2. Helper functions
# =========================================================

def write_csv(path, fieldnames, rows):
    """Write rows to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union between two boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def is_near_frame_border(box, frame_width, frame_height):
    """Return True when a person box touches a video border."""

    x1, y1, x2, y2 = box

    margin_x = int(frame_width * BORDER_MARGIN_RATIO)
    margin_y = int(frame_height * BORDER_MARGIN_RATIO)

    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - margin_x
        or y2 >= frame_height - margin_y
    )


def safe_crop(frame, x1, y1, x2, y2):
    """Crop a region safely, returning None if it is invalid."""

    height, width = frame.shape[:2]

    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def calculate_blur_score(face_crop):
    """Estimate image sharpness using Laplacian variance."""

    if face_crop is None or face_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# =========================================================
# 3. Validate inputs
# =========================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Input video was not found:\n{VIDEO_PATH}\n"
        "Make sure testvedio.mp4 is inside the project folder."
    )

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face detection model was not found:\n{FACE_MODEL_PATH}\n"
        "Make sure face_detection_yunet.onnx is inside the project folder."
    )

FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 4. Load models
# =========================================================

print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)

print("Loading YuNet face detector...")

face_detector = cv2.FaceDetectorYN.create(
    str(FACE_MODEL_PATH),
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# =========================================================
# 5. Open video and prepare output
# =========================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25.0

if original_width <= 0 or original_height <= 0:
    cap.release()
    raise RuntimeError("The video dimensions are invalid.")

scale = min(1.0, PROCESSING_WIDTH / original_width)

processing_width = max(1, int(original_width * scale))
processing_height = max(1, int(original_height * scale))

print(f"Input video: {VIDEO_PATH.name}")
print(f"Video size: {original_width} x {original_height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Processing size: {processing_width} x {processing_height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    fps,
    (original_width, original_height)
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not create output video. "
        "Check the output path and available disk space."
    )


# =========================================================
# 6. Tracking and event storage
# =========================================================

tracks = {}
face_rows = []
event_rows = []

# Track IDs present in the previous processed frame.
previous_frame_ids = set()

# Last frame in which each track was detected.
last_seen_frame = {}

# Remember the last frame where an overlap event was recorded.
last_overlap_event_frame = {}

# Keep track of pairs already reported as overlapping.
# The frame interval above allows later overlap events to be logged.
start_time = time.time()
frame_number = 0


def record_event(clip_id, frame_no, time_seconds, track_id, event, details=""):
    event_rows.append({
        "clip_id": clip_id,
        "frame": frame_no,
        "time_seconds": round(time_seconds, 3),
        "track_id": track_id,
        "event": event,
        "details": details
    })


clip_id = VIDEO_PATH.stem


# =========================================================
# 7. Process video frames
# =========================================================

while True:
    success, original_frame = cap.read()

    if not success:
        break

    frame_number += 1
    current_time = (frame_number - 1) / fps

    small_frame = cv2.resize(
        original_frame,
        (processing_width, processing_height)
    )

    # Keep YuNet's input dimensions aligned with the resized frame.
    face_detector.setInputSize(
        (processing_width, processing_height)
    )

    # -----------------------------------------------------
    # 7A. Detect and track people
    # -----------------------------------------------------

    results = model.track(
        small_frame,
        persist=True,
        classes=[0],
        tracker=TRACKER_CONFIG,
        imgsz=640,
        verbose=False
    )

    current_people = {}
    current_frame_ids = set()

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes

        if boxes.id is not None:
            box_coordinates = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for coordinates, track_id, confidence in zip(
                box_coordinates, track_ids, confidences
            ):
                x1, y1, x2, y2 = map(int, coordinates)

                # Clip coordinates to the resized frame.
                x1 = max(0, min(x1, processing_width - 1))
                y1 = max(0, min(y1, processing_height - 1))
                x2 = max(0, min(x2, processing_width - 1))
                y2 = max(0, min(y2, processing_height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                person_box = (x1, y1, x2, y2)

                current_people[track_id] = {
                    "box": person_box,
                    "confidence": float(confidence)
                }

                current_frame_ids.add(track_id)

    # -----------------------------------------------------
    # 7B. Start, update, and reappearance events
    # -----------------------------------------------------

    for track_id, person_data in current_people.items():

        person_box = person_data["box"]

        if track_id not in tracks:
            tracks[track_id] = {
                "first_seen_frame": frame_number,
                "last_seen_frame": frame_number,
                "first_seen_seconds": current_time,
                "last_seen_seconds": current_time,
                "observed_frames": 1,
                "max_gap_frames": 0,
                "was_declared_lost": False,
                "near_border_first_seen": is_near_frame_border(
                    person_box, processing_width, processing_height
                ),
                "near_border_last_seen": is_near_frame_border(
                    person_box, processing_width, processing_height
                )
            }

            record_event(
                clip_id,
                frame_number,
                current_time,
                track_id,
                "track_started",
                "New tracking ID detected; not a verified real-world identity."
            )

            if is_near_frame_border(
                person_box, processing_width, processing_height
            ):
                record_event(
                    clip_id,
                    frame_number,
                    current_time,
                    track_id,
                    "possible_entry_from_border",
                    "Track first appeared near a video border."
                )

        else:
            previous_seen = last_seen_frame.get(track_id, frame_number)
            gap = frame_number - previous_seen - 1

            tracks[track_id]["max_gap_frames"] = max(
                tracks[track_id]["max_gap_frames"],
                max(0, gap)
            )

            if gap >= LOST_AFTER_FRAMES:
                record_event(
                    clip_id,
                    frame_number,
                    current_time,
                    track_id,
                    "track_reappeared_after_gap",
                    f"Track returned after {gap} missing frames."
                )

            elif gap > 1:
                record_event(
                    clip_id,
                    frame_number,
                    current_time,
                    track_id,
                    "track_reappeared_after_short_gap",
                    f"Track returned after {gap} missing frames."
                )

            tracks[track_id]["last_seen_frame"] = frame_number
            tracks[track_id]["last_seen_seconds"] = current_time
            tracks[track_id]["observed_frames"] += 1
            tracks[track_id]["was_declared_lost"] = False
            tracks[track_id]["near_border_last_seen"] = is_near_frame_border(
                person_box, processing_width, processing_height
            )

        last_seen_frame[track_id] = frame_number

    # -----------------------------------------------------
    # 7C. Detect missing/lost tracks
    # -----------------------------------------------------

    for track_id, last_frame in list(last_seen_frame.items()):

        missing_frames = frame_number - last_frame

        if (
            missing_frames >= LOST_AFTER_FRAMES
            and not tracks[track_id]["was_declared_lost"]
        ):
            record_event(
                clip_id,
                frame_number,
                current_time,
                track_id,
                "track_lost",
                f"No detection for {missing_frames} consecutive frames."
            )

            tracks[track_id]["was_declared_lost"] = True

            if tracks[track_id]["near_border_last_seen"]:
                record_event(
                    clip_id,
                    frame_number,
                    current_time,
                    track_id,
                    "possible_exit_from_border",
                    "Track was last seen near a video border; exit is not confirmed."
                )

    # -----------------------------------------------------
    # 7D. Detect person-box overlap
    # -----------------------------------------------------

    current_ids = list(current_people.keys())

    for i in range(len(current_ids)):
        for j in range(i + 1, len(current_ids)):

            id_a = current_ids[i]
            id_b = current_ids[j]

            box_a = current_people[id_a]["box"]
            box_b = current_people[id_b]["box"]

            overlap = calculate_iou(box_a, box_b)

            if overlap >= OVERLAP_IOU_THRESHOLD:

                pair = tuple(sorted((id_a, id_b)))
                previous_overlap_frame = last_overlap_event_frame.get(
                    pair, -OVERLAP_EVENT_INTERVAL
                )

                if (
                    frame_number - previous_overlap_frame
                    >= OVERLAP_EVENT_INTERVAL
                ):
                    details = (
                        f"Overlapping person boxes; IoU={overlap:.3f}. "
                        "Face-to-track association may be ambiguous."
                    )

                    record_event(
                        clip_id, frame_number, current_time,
                        id_a, "overlap_detected",
                        f"Other track: {id_b}. {details}"
                    )

                    record_event(
                        clip_id, frame_number, current_time,
                        id_b, "overlap_detected",
                        f"Other track: {id_a}. {details}"
                    )

                    last_overlap_event_frame[pair] = frame_number

    # -----------------------------------------------------
    # 7E. Detect faces
    # -----------------------------------------------------

    face_status_by_track = {}
    face_info_by_track = {}

    face_result = face_detector.detect(small_frame)
    detected_faces = face_result[1]

    faces = []

    if detected_faces is not None:
        for face in detected_faces:
            fx, fy, fw, fh = face[:4]

            face_confidence = float(face[-1])

            fx1 = int(fx)
            fy1 = int(fy)
            fx2 = int(fx + fw)
            fy2 = int(fy + fh)

            fx1 = max(0, min(fx1, processing_width - 1))
            fy1 = max(0, min(fy1, processing_height - 1))
            fx2 = max(0, min(fx2, processing_width))
            fy2 = max(0, min(fy2, processing_height))

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            faces.append({
                "box": (fx1, fy1, fx2, fy2),
                "confidence": face_confidence,
                "width": fx2 - fx1,
                "height": fy2 - fy1
            })

    # -----------------------------------------------------
    # 7F. Assign a face only when its center belongs to
    #     exactly one detected person box.
    # -----------------------------------------------------

    faces_by_track = defaultdict(list)
    ambiguous_tracks = set()

    for face in faces:

        fx1, fy1, fx2, fy2 = face["box"]

        face_center_x = (fx1 + fx2) / 2
        face_center_y = (fy1 + fy2) / 2

        candidate_ids = []

        for track_id, person_data in current_people.items():
            px1, py1, px2, py2 = person_data["box"]

            if (
                px1 <= face_center_x <= px2
                and py1 <= face_center_y <= py2
            ):
                candidate_ids.append(track_id)

        if len(candidate_ids) == 1:
            faces_by_track[candidate_ids[0]].append(face)

        elif len(candidate_ids) > 1:
            # Do not guess which person owns a face when
            # multiple person boxes contain the face center.
            ambiguous_tracks.update(candidate_ids)

    # -----------------------------------------------------
    # 7G. Evaluate face quality and save valid face crops
    # -----------------------------------------------------

    for track_id, person_data in current_people.items():

        person_box = person_data["box"]

        face_status = "No Face Detected"
        face_confidence = ""
        face_width = ""
        face_height = ""
        blur_score = ""
        crop_path = ""

        assigned_faces = faces_by_track.get(track_id, [])

        if track_id in ambiguous_tracks:
            face_status = "Insufficient Quality"

        elif assigned_faces:
            # Choose the largest face only among faces
            # uniquely assigned to this person's track.
            face = max(
                assigned_faces,
                key=lambda item: item["width"] * item["height"]
            )

            fx1, fy1, fx2, fy2 = face["box"]

            face_confidence = round(face["confidence"], 4)
            face_width = face["width"]
            face_height = face["height"]

            # Convert resized-frame coordinates to original coordinates.
            original_fx1 = round(fx1 / scale)
            original_fy1 = round(fy1 / scale)
            original_fx2 = round(fx2 / scale)
            original_fy2 = round(fy2 / scale)

            face_crop = safe_crop(
                original_frame,
                original_fx1,
                original_fy1,
                original_fx2,
                original_fy2
            )

            blur_score_value = calculate_blur_score(face_crop)
            blur_score = round(blur_score_value, 2)

            quality_ok = (
                face_width >= MIN_FACE_WIDTH
                and face_height >= MIN_FACE_HEIGHT
                and face["confidence"] >= MIN_FACE_CONFIDENCE
                and blur_score_value >= MIN_BLUR_SCORE
                and face_crop is not None
                and face_crop.size > 0
            )

            if quality_ok:
                face_status = "Quality Acceptable"

                crop_filename = (
                    f"track_{track_id}_frame_{frame_number}.jpg"
                )

                crop_full_path = FACE_CROPS_DIR / crop_filename

                if cv2.imwrite(str(crop_full_path), face_crop):
                    crop_path = str(
                        Path(FACE_CROPS_DIR.name) / crop_filename
                    )
                else:
                    face_status = "Insufficient Quality"
                    crop_path = ""

            else:
                face_status = "Insufficient Quality"

        face_status_by_track[track_id] = face_status

        face_info_by_track[track_id] = {
            "status": face_status,
            "box": (
                assigned_faces[0]["box"]
                if len(assigned_faces) == 1
                else None
            )
        }

        face_rows.append({
            "clip_id": clip_id,
            "frame": frame_number,
            "time_seconds": round(current_time, 3),
            "track_id": track_id,
            "face_status": face_status,
            "face_confidence": face_confidence,
            "face_width": face_width,
            "face_height": face_height,
            "blur_score": blur_score,
            "crop_path": crop_path
        })

    # -----------------------------------------------------
    # 7H. Draw tracking and face-quality information
    # -----------------------------------------------------

    for track_id, person_data in current_people.items():

        x1, y1, x2, y2 = person_data["box"]

        # Map person box to original video coordinates.
        ox1 = round(x1 / scale)
        oy1 = round(y1 / scale)
        ox2 = round(x2 / scale)
        oy2 = round(y2 / scale)

        status = face_status_by_track.get(
            track_id, "No Face Detected"
        )

        color = (0, 200, 0)

        if status == "Insufficient Quality":
            color = (0, 165, 255)

        cv2.rectangle(
            original_frame,
            (ox1, oy1),
            (ox2, oy2),
            color,
            2
        )

        label = f"Person ID: {track_id}"

        cv2.putText(
            original_frame,
            label,
            (ox1, max(25, oy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.putText(
            original_frame,
            status,
            (ox1, min(original_height - 10, oy2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2
        )

        # Draw uniquely assigned face boxes only.
        for face in faces_by_track.get(track_id, []):
            fx1, fy1, fx2, fy2 = face["box"]

            cv2.rectangle(
                original_frame,
                (
                    round(fx1 / scale),
                    round(fy1 / scale)
                ),
                (
                    round(fx2 / scale),
                    round(fy2 / scale)
                ),
                (255, 180, 0),
                2
            )

    video_writer.write(original_frame)

    previous_frame_ids = current_frame_ids

    if frame_number % 30 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")


# =========================================================
# 8. Finalize video and log track endings
# =========================================================

cap.release()
video_writer.release()

final_time = max(0, (frame_number - 1) / fps)

for track_id, track_data in tracks.items():

    record_event(
        clip_id,
        frame_number,
        final_time,
        track_id,
        "track_ended",
        "Processing ended. This does not prove the person exited the scene."
    )

    if track_data["near_border_last_seen"]:
        record_event(
            clip_id,
            track_data["last_seen_frame"],
            track_data["last_seen_seconds"],
            track_id,
            "last_seen_near_border",
            "Possible exit location; leaving the scene is not confirmed."
        )


# =========================================================
# 9. Save tracking results
# =========================================================

tracking_rows = []

for track_id, data in tracks.items():

    observed_duration = (
        data["last_seen_seconds"] - data["first_seen_seconds"]
    )

    tracking_rows.append({
        "clip_id": clip_id,
        "track_id": track_id,
        "first_seen_frame": data["first_seen_frame"],
        "last_seen_frame": data["last_seen_frame"],
        "first_seen_seconds": round(
            data["first_seen_seconds"], 3
        ),
        "last_seen_seconds": round(
            data["last_seen_seconds"], 3
        ),
        "observed_duration_seconds": round(
            observed_duration, 3
        ),
        "observed_frames": data["observed_frames"],
        "max_gap_frames": data["max_gap_frames"]
    })

tracking_rows.sort(key=lambda row: row["first_seen_frame"])

write_csv(
    TRACKING_CSV_PATH,
    [
        "clip_id",
        "track_id",
        "first_seen_frame",
        "last_seen_frame",
        "first_seen_seconds",
        "last_seen_seconds",
        "observed_duration_seconds",
        "observed_frames",
        "max_gap_frames"
    ],
    tracking_rows
)


# =========================================================
# 10. Save face-quality results
# =========================================================

write_csv(
    FACE_QUALITY_CSV_PATH,
    [
        "clip_id",
        "frame",
        "time_seconds",
        "track_id",
        "face_status",
        "face_confidence",
        "face_width",
        "face_height",
        "blur_score",
        "crop_path"
    ],
    face_rows
)


# =========================================================
# 11. Save tracking events
# =========================================================

write_csv(
    EVENTS_CSV_PATH,
    [
        "clip_id",
        "frame",
        "time_seconds",
        "track_id",
        "event",
        "details"
    ],
    event_rows
)


# =========================================================
# 12. Print summary
# =========================================================

elapsed = time.time() - start_time

print("\nProcessing finished.")
print(f"Frames processed: {frame_number}/{total_frames}")
print(f"Tracked IDs: {len(tracks)}")
print(f"Tracking events: {len(event_rows)}")
print(f"Face-quality rows: {len(face_rows)}")
print(f"Total processing time: {elapsed:.2f} seconds")

if elapsed > 0:
    print(f"Average processing speed: {frame_number / elapsed:.2f} FPS")

print("\nOutput files:")
print(f"Video: {OUTPUT_VIDEO_PATH}")
print(f"Tracking results: {TRACKING_CSV_PATH}")
print(f"Face quality: {FACE_QUALITY_CSV_PATH}")
print(f"Tracking events: {EVENTS_CSV_PATH}")
print(f"Face crops folder: {FACE_CROPS_DIR}")

print("\nImportant notes:")
print("- A track_id is a tracking label, not a verified human identity.")
print("- Overlap events indicate possible association problems.")
print("- Lost tracks and border events are indicators, not proof of exit.")
print("- Poor or ambiguous face crops are not saved as valid face evidence.")
print("- No nearest-name identity guessing is performed.")
