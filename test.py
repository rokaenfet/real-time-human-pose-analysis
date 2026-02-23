from helper.load_UPNA import UPNALoader
from helper.mediapipe_helper import MediaPipeHeadPoseEstimator
from helper.screen_util import ScreenUtility

import tkinter as tk
import sys
import cv2
import time

# --- Initialization ---
try:
    screen_utility = ScreenUtility()
    upna_loader = UPNALoader()
    mediapipe_face_estimator = MediaPipeHeadPoseEstimator(angle_smoothener_alpha=.1)
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# pick video
USER_ID = "01"
VIDEO_ID = "05"
video = upna_loader.load_video_cv2(user_id=USER_ID, video_id=VIDEO_ID)


fps = 0.0
frame_count = 0
last_time = time.time()

is_playing = True
paused_frame = None

while True:
    if is_playing:
        ret, frame = video.read()
        if not ret: break

        # 1. Detect
        result = mediapipe_face_estimator.detect_landmarks(frame)
        vis_frame = frame.copy()

        if result and result.face_landmarks:
            # 2. Estimate
            rot_deg, axes = mediapipe_face_estimator.estimate_head_pose(
                result.face_landmarks[0], frame.shape
            )

            # 3. Draw All Visualization (Mesh + Pose)
            vis_frame = mediapipe_face_estimator.draw_landmarks(
                vis_frame, result, draw_index_of_interest=False, draw_tesselation=False, draw_contours=False, draw_iris=False
            )
            
            # This handles smoothing, text, and 3D lines internally
            vis_frame, smooth_angles = mediapipe_face_estimator.draw_pose_info(
                vis_frame, rot_deg, axes
            )
        
        paused_frame = vis_frame.copy()

        # --- FPS Calculation ---
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - last_time
        # Update FPS every 1 second
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            last_time = current_time

        # rendered info
        cv2.putText(vis_frame, f"User: {USER_ID} | FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        if paused_frame is not None:
            frame = paused_frame.copy()
        else:
            time.sleep(0.01)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): is_playing = not is_playing
            continue
    
    cv2.imshow("", vis_frame)

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