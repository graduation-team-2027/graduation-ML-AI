

import cv2
import time

video_path = "testvedio.mp4"
model_path = "face_detection_yunet.onnx"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

# Reduce processing resolution to speed up face detection
processing_width = 640

face_detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (processing_width, processing_width),
    0.8,
    0.3,
    5000
)

window_name = "Face Detection Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

print("Press Q to stop the test.")

frame_number = 0
start_time = time.time()

try:
    while True:
        success, original_frame = cap.read()
        if not success:
            break

        frame_number += 1

        original_height, original_width = original_frame.shape[:2]

        # Resize only the working copy, not the original video
        scale = processing_width / original_width
        processing_height = max(1, int(original_height * scale))

        small_frame = cv2.resize(
            original_frame,
            (processing_width, processing_height)
        )

        face_detector.setInputSize(
            (processing_width, processing_height)
        )

        _, faces = face_detector.detect(small_frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])

                # Convert coordinates to the original display frame
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w) / scale)
                y2 = int((y + h) / scale)

                cv2.rectangle(
                    original_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    original_frame,
                    "Face",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Display a smaller copy to avoid a huge window
        display_width = 800
        display_height = int(
            original_height * display_width / original_width
        )

        display_frame = cv2.resize(
            original_frame,
            (display_width, display_height)
        )

        cv2.imshow(window_name, display_frame)

        # Keep playback close to the video's original frame rate
        target_time = frame_number / fps
        elapsed = time.time() - start_time
        delay_ms = max(1, int((target_time - elapsed) * 1000))

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Face detection test finished.")
print("The original video was not modified.")
