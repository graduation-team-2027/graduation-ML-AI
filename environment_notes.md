
# Environment & Version Notes

**Project:** CCTV-Based Smoking and Vaping Detection
**Task:** Person Detection and Tracking
**Operating System:** Windows
**IDE:** Visual Studio Code
**Python Version:** 3.13.9
**Virtual Environment:** `venv`

## 1. Libraries and Tools

| Component             | Version / Configuration |
| --------------------- | ----------------------- |
| Python                | 3.13.9                  |
| Ultralytics           | 8.4.184                 |
| OpenCV                | 5.0.0                   |
| PyTorch               | 2.14.1+cpu              |
| YOLO Model            | `yolo26n.pt`            |
| Tracking Algorithm    | ByteTrack               |
| Tracker Configuration | `bytetrack.yaml`        |

## 2. Input and Output Files

- **Input video:** `testvedio.mp4`
- **Tracking script:** `person_tracking.py`
- **Annotated output video:** `tracked_output.mp4`
- **Tracking results file:** `tracking_results.csv`

## 3. Execution Procedure

1. Activate the Python virtual environment.
2. Run `person_tracking.py` from the project directory.
3. Use the YOLO model `yolo26n.pt` to detect people in the video frames.
4. Use ByteTrack with `bytetrack.yaml` to track detected people and assign track IDs.
5. Draw bounding boxes and track IDs on the video frames.
6. Save the annotated video as `tracked_output.mp4`.
7. Save tracking information and timestamps in `tracking_results.csv`.

## 4. Tracking Information

The tracking results CSV file is used to record information such as:

- `track_id`: The ID assigned to a tracked person.
- `first_seen`: The first time the person was detected.
- `last_seen`: The last time the person was detected.
- `observed_duration_seconds`: The time between the first and last observation of the track.

## 5. Notes and Limitations

- The input video was opened and processed by the tracking script.
- Person detection and tracking were executed using YOLO and ByteTrack.
- The script generated an annotated output video and a CSV file.
- The output video must be checked to confirm that all input frames were processed and that bounding boxes and track IDs are clearly visible.
- Track IDs represent tracking identities, not verified real-world identities. IDs may change if a person is temporarily lost or re-enters the scene.
- The tracking duration represents the observed time between the first and last detection of a track. It does not necessarily represent the person's total time in the scene.

## 6. Verification Status

- [x] Python libraries and their versions verified.
- [x] YOLO model and tracker configuration recorded.
- [x] Input and output filenames documented.
- [x] Tracking script executed.
- [x] Output video checked for complete frame processing.
- [x] Bounding boxes and track IDs visually verified.

