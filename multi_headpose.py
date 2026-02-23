import cv2
import numpy as np
import time
import sys
from helper.load_UPNA import UPNALoader
from helper.mediapipe_helper import MediaPipeHeadPoseEstimator
from helper.screen_util import ScreenUtility

# --- Configuration ---
USER_ID = "02"
VIDEO_IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"] # Add as many as you want
ALPHA = 0.2
# Export settings
SAVE_VIDEO = True  
OUTPUT_PATH = "upna_grid_analysis.mp4"
video_writer = None

try:
    screen_utility = ScreenUtility()
    upna_loader = UPNALoader()
    # Create an estimator per video for independent smoothing
    estimators = [MediaPipeHeadPoseEstimator(angle_smoothener_alpha=ALPHA) for _ in VIDEO_IDS]
    videos = [upna_loader.load_video_cv2(user_id=USER_ID, video_id=vid) for vid in VIDEO_IDS]
    videos = [n for n in videos if n is not None]
except Exception as e:
    print(f"Init Error: {e}")
    sys.exit()

# Grid Settings
num_vids = len(videos)
cols = int(np.ceil(np.sqrt(num_vids)))
rows = int(np.ceil(num_vids / cols))

# Calculate tile size based on screen (w margin)
target_w, target_h = screen_utility.get_available_screen_size()
print(target_w, target_h, rows, cols)
tile_size = (int(target_w//cols), int(target_h//rows))

is_playing = True
paused_tiles = [None] * num_vids

while True:
    frames_to_show = []

    for i in range(num_vids):
        if is_playing:
            ret, frame = videos[i].read()
            if not ret:
                # Loop video if it ends
                videos[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = videos[i].read()

            # Process Individual Video
            result = estimators[i].detect_landmarks(frame)
            vis_frame = frame.copy()

            if result and result.face_landmarks:
                rot_deg, axes = estimators[i].estimate_head_pose(result.face_landmarks[0], frame.shape)
                vis_frame = estimators[i].draw_landmarks(vis_frame, result, 
                                                         draw_index_of_interest=False,
                                                         draw_tesselation=False,
                                                         draw_contours=False,
                                                         draw_iris=False
                                                         )
                vis_frame, _ = estimators[i].draw_pose_info(vis_frame, rot_deg, axes)
            
            # Prepare tile
            tile = screen_utility.resize_with_padding(vis_frame, tile_size)
            cv2.putText(tile, f"ID: {VIDEO_IDS[i]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            paused_tiles[i] = tile
        else:
            tile = paused_tiles[i]
        
        frames_to_show.append(tile)

    # Padding the list with empty black tiles if grid isn't full
    while len(frames_to_show) < (rows * cols):
        frames_to_show.append(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))

    # Construct Grid
    grid_rows = []
    for r in range(rows):
        row_data = np.hstack(frames_to_show[r*cols : (r+1)*cols])
        grid_rows.append(row_data)
    
    # Construct final_display (assuming you used np.vstack on grid_rows)
    final_display = np.vstack(grid_rows)

    # Initialize VideoWriter ONCE after the first frame is built
    if SAVE_VIDEO and video_writer is None:
        h, w, _ = final_display.shape
        # 'mp4v' is widely compatible for .mp4 files
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (w, h))

    # Write the frame
    if SAVE_VIDEO and video_writer is not None:
        video_writer.write(final_display)

    cv2.imshow("UPNA Multi-View Pose Estimator", final_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): is_playing = not is_playing

# Cleanup
for v in videos: v.release()
for e in estimators: e.close()
cv2.destroyAllWindows()