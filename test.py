import cv2
import numpy as np
import os
import glob
import math
from helper.mediapipe_helper import MediaPipeBodyPoseEstimator
from helper.screen_util import ScreenUtility

# --- Configuration ---
VIDEO_DIR = "Body_Pose_Database_CMU/hdVideos/"
video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))[:4]

SAVE_VIDEO = False
OUTPUT_PATH = "cmu_body_analysis_grid.mp4"

# --- Setup ---
caps = [cv2.VideoCapture(p) for p in video_paths]
num_vids = len(caps)
cols = math.ceil(math.sqrt(num_vids))
rows = math.ceil(num_vids / cols)

screen_utility = ScreenUtility()

screen_w, screen_h = screen_utility.get_available_screen_size()
tile_w = int((screen_w) // cols)
tile_h = int((screen_h) // rows)

estimator = MediaPipeBodyPoseEstimator()

video_writer = None
is_playing = True
paused_tiles = [None] * num_vids

try:
    while True:
        display_tiles = []
        
        for i in range(num_vids):
            if is_playing:
                ret, frame = caps[i].read()
                if not ret:
                    caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = caps[i].read()

                # 1. Detection
                result = estimator.detect_landmarks(frame)
                
                # 2. Pose Estimation & Drawing
                vis_frame = frame.copy()
                if result and result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    
                    # Calculate Euler angles and axes
                    angles, axes_data = estimator.estimate_body_pose(landmarks, frame.shape)
                    
                    # Draw official skeleton and custom pose info
                    vis_frame = estimator.draw_landmarks(vis_frame, result, draw_skeleton=False)
                    vis_frame = estimator.draw_pose_info(vis_frame, angles, axes_data)

                # 3. Resize for Grid
                tile = screen_utility.resize_with_padding(vis_frame, [tile_w, tile_h])
                paused_tiles[i] = tile
            else:
                tile = paused_tiles[i]
            
            display_tiles.append(tile)

        # Fill empty grid slots
        while len(display_tiles) < (rows * cols):
            display_tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

        # Assemble Grid
        grid_rows = [np.hstack(display_tiles[r*cols : (r+1)*cols]) for r in range(rows)]
        final_grid = np.vstack(grid_rows)

        # Video Export Logic
        if SAVE_VIDEO:
            if video_writer is None:
                gh, gw, _ = final_grid.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (gw, gh))
            if is_playing:
                video_writer.write(final_grid)

        # UI
        cv2.imshow("CMU Panoptic - Euler Body Analysis", final_grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): is_playing = not is_playing

finally:
    # Cleanup
    if video_writer: video_writer.release()
    for c in caps: c.release()
    estimator.close()
    cv2.destroyAllWindows()
    print(f"Export complete: {OUTPUT_PATH}")