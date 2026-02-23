from helper.load_UPNA import UPNALoader
from helper.mediapipe_helper import MediaPipeHeadPoseEstimator
from helper.screen_util import ScreenUtility

import tkinter as tk
import sys
import cv2

screen_utility = ScreenUtility()
upna_loader = UPNALoader()
mediapipe_face_estimator = MediaPipeHeadPoseEstimator()

# --- Initialization ---
try:
    screen_utility = ScreenUtility()
    upna_loader = UPNALoader()
    mediapipe_face_estimator = MediaPipeHeadPoseEstimator()
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# pick video
USER_ID = "01"
VIDEO_ID = "04"
video = upna_loader.load_video_cv2(user_id=USER_ID, video_id=VIDEO_ID)

is_playing = True
paused_frame = None

while True:
    if is_playing:
        ret, frame = video.read()
    else:
        if paused_frame is not None:
            frame = paused_frame.copy()

    result = mediapipe_face_estimator.detect_landmarks(frame)
    vis_frame = mediapipe_face_estimator.draw_landmarks(frame, result)
    paused_frame = vis_frame

    status = "PLAYING" if is_playing else "PAUSED"
    cv2.putText(vis_frame, f"User: {USER_ID} | {status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("",vis_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        is_playing = not is_playing
        print(f"State: {'Paused' if not is_playing else 'Playing'}")

# --- Cleanup ---
video.release()
mediapipe_face_estimator.close()
cv2.destroyAllWindows()
print("Test finished.")