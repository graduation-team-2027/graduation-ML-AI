
import cv2

video_path = "testvedio.mp4"

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("Video opened successfully!")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Total frames:", total_frames)
    print("FPS:", fps)

    cap.release()
else:
    print("Error: Could not open the video.")
    