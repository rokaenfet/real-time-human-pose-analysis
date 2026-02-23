import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks import python
import numpy as np
import os
import time
import glob
import math
import tkinter as tk

# --- Configuration ---
FACE_MODEL_PATH = 'face_landmarker.task' 
BASE_DIR = 'Head_Pose_Database_UPNA'
GAP_SIZE = 5
PADDING_COLOR = (0, 0, 0)  
SCREEN_MARGIN = 100

FACE_MESH_INDICES = [1, 152, 234, 454, 33, 362]

def calculate_euler_angles(rvec, tvec):
    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0,0]**2 + rot_matrix[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rot_matrix[2,0], sy))
        yaw = np.degrees(np.arctan2(rot_matrix[1,0], rot_matrix[0,0]))
        roll = np.degrees(np.arctan2(rot_matrix[2,1], rot_matrix[2,2]))
    else:
        pitch = np.degrees(np.arctan2(-rot_matrix[1,2], rot_matrix[1,1]))
        yaw = 0
        roll = 0
    return yaw, pitch, roll

def process_frame(frame, detector, stream_id):
    h, w = frame.shape[:2]
    if h == 0 or w == 0: return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            lm_list = face_landmarks
            image_points = []
            world_points = []
            valid_detection = True
            
            nose_lm = lm_list[1]
            
            for idx in FACE_MESH_INDICES:
                lm = lm_list[idx]
                if (lm.z - nose_lm.z) > 0.3: 
                    valid_detection = False
                    break
                
                px, py = lm.x * w, lm.y * h
                image_points.append([px, py])
                world_points.append([lm.x * w, lm.y * h, lm.z * w])
            
            if valid_detection:
                image_points = np.array(image_points, dtype=np.float32)
                world_points = np.array(world_points, dtype=np.float32)

                focal_length = 1.0 * w 
                cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.zeros((4,1))

                try:
                    success, rvec, tvec = cv2.solvePnP(world_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if success:
                        yaw, pitch, roll = calculate_euler_angles(rvec, tvec)
                        
                        # Stability Fix
                        if abs(abs(yaw) - 180) < 10:
                            yaw = yaw - 180 if yaw > 0 else yaw + 180

                        nose_2d = (int(nose_lm.x * w), int(nose_lm.y * h))
                        scale_factor = min(w, h) / 500.0 
                        axis_len = int(80 * scale_factor)
                        line_thickness = max(1, int(2 * scale_factor))

                        # Get the full 3x3 Rotation Matrix from rvec
                        rot_matrix, _ = cv2.Rodrigues(rvec)

                        # Define unit vectors for X, Y, Z in the object's local space
                        # X = Right, Y = Up (inverted for image coords later), Z = Forward
                        x_axis_obj = np.array([1.0, 0.0, 0.0])
                        y_axis_obj = np.array([0.0, -1.0, 0.0]) # Negative because Image Y is Down
                        z_axis_obj = np.array([0.0, 0.0, 1.0])

                        # Rotate these vectors by the head's rotation matrix
                        # This transforms them from "Object Space" to "Camera Space"
                        x_axis_cam = rot_matrix @ x_axis_obj
                        y_axis_cam = rot_matrix @ y_axis_obj
                        z_axis_cam = rot_matrix @ z_axis_obj

                        # Scale them to desired length
                        # We multiply Z by 5 again for visibility as you requested
                        x_axis_cam *= axis_len
                        y_axis_cam *= axis_len
                        z_axis_cam *= axis_len * 5

                        # Calculate the 2D end points by adding the rotated vector to the nose position
                        # We only take the X and Y components because we are drawing on a 2D screen
                        red_end = (int(nose_2d[0] + x_axis_cam[0]), int(nose_2d[1] + x_axis_cam[1]))
                        green_end = (int(nose_2d[0] + y_axis_cam[0]), int(nose_2d[1] + y_axis_cam[1]))
                        blue_end = (int(nose_2d[0] + z_axis_cam[0]), int(nose_2d[1] + z_axis_cam[1]))

                        # Draw the axes
                        cv2.line(frame, nose_2d, red_end, (0, 0, 255), line_thickness)
                        cv2.putText(frame, "X", (red_end[0]-10, red_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale_factor, (0, 0, 255), 1)

                        cv2.line(frame, nose_2d, green_end, (0, 255, 0), line_thickness)
                        cv2.putText(frame, "Y", (green_end[0]-10, green_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale_factor, (0, 255, 0), 1)

                        cv2.arrowedLine(frame, nose_2d, blue_end, (255, 0, 0), line_thickness, tipLength=0.3)
                        cv2.circle(frame, blue_end, int(4 * scale_factor), (255, 0, 0), -1)
                        cv2.putText(frame, "Z", (blue_end[0]+10, blue_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale_factor, (255, 0, 0), 1)
                except Exception as e:
                    pass
    
    return frame

def pad_to_size(frame, target_w, target_h):
    h, w = frame.shape[:2]
    if w == target_w and h == target_h: return frame
    canvas = np.full((target_h, target_w, 3), PADDING_COLOR, dtype=np.uint8)
    y_start = (target_h - h) // 2
    x_start = (target_w - w) // 2
    canvas[y_start:y_start+h, x_start:x_start+w] = frame
    return canvas

def create_grid(frames, cols, cell_w, cell_h, gap):
    rows = math.ceil(len(frames) / cols)
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap
    grid_canvas = np.full((total_h, total_w, 3), PADDING_COLOR, dtype=np.uint8)
    
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        y_start = r * (cell_h + gap)
        y_end = y_start + cell_h
        x_start = c * (cell_w + gap)
        x_end = x_start + cell_w
        
        if frame.shape[0] == cell_h and frame.shape[1] == cell_w:
            grid_canvas[y_start:y_end, x_start:x_end] = frame
        else:
            resized = cv2.resize(frame, (cell_w, cell_h))
            grid_canvas[y_start:y_end, x_start:x_end] = resized
    return grid_canvas

# --- Main Execution ---

user_num = input("Enter User Number (e.g., 01): ").strip()
user_dir = os.path.join(BASE_DIR, f"User_{user_num}")

if not os.path.exists(user_dir):
    print(f"Error: Directory '{user_dir}' not found.")
    exit()

video_files = sorted(glob.glob(os.path.join(user_dir, "*.mp4")))
if not video_files: video_files = sorted(glob.glob(os.path.join(user_dir, "*.avi")))
if not video_files: video_files = sorted(glob.glob(os.path.join(user_dir, "*.mov")))

if not video_files:
    print(f"No video files found in '{user_dir}'.")
    exit()

print(f"Found {len(video_files)} videos. Analyzing resolutions...")

captures = []
max_w, max_h = 0, 0

for path in video_files:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: Could not open {path}")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_w = max(max_w, w)
    max_h = max(max_h, h)
    captures.append(cap)

if not captures:
    print("No valid videos to play.")
    exit()

print(f"Max Resolution: {max_w}x{max_h}. Loading Model...")

base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
options = face_landmarker.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)

try:
    detector = face_landmarker.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

num_videos = len(captures)
cols = math.ceil(math.sqrt(num_videos))
rows = math.ceil(num_videos / cols)
print(f"Grid Layout: {rows} rows x {cols} cols")

root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
available_w = screen_w - SCREEN_MARGIN
available_h = screen_h - SCREEN_MARGIN

print("\n=== CONTROLS ===")
print("SPACE : Play / Pause")
print("E     : Next Frame (Forward)")
print("R     : Prev Frame (Backward)")
print("Q     : Quit")
print("================\n")

is_playing = False
last_time = time.time()
frame_count = 0

current_frames = []
for cap in captures:
    ret, frame = cap.read()
    if not ret: frame = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    current_frames.append(frame)

running = True
while running:
    processed_frames = []
    
    # If playing, read next frames
    if is_playing:
        new_frames = []
        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
                ret, frame = cap.read()
                if not ret: frame = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            new_frames.append(frame)
        current_frames = new_frames

    # Process current frames
    for i, frame in enumerate(current_frames):
        proc_frame = process_frame(frame.copy(), detector, i+1)
        final_frame = pad_to_size(proc_frame, max_w, max_h)
        processed_frames.append(final_frame)

    # Fill empty slots
    total_slots = rows * cols
    while len(processed_frames) < total_slots:
        blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        processed_frames.append(blank)

    # Create Grid
    grid_image = create_grid(processed_frames, cols, max_w, max_h, GAP_SIZE)
    
    # Scale to Fit Screen
    grid_h, grid_w = grid_image.shape[:2]
    scale_x = available_w / grid_w
    scale_y = available_h / grid_h
    scale_factor = min(scale_x, scale_y)
    
    if scale_factor < 1.0:
        new_w = int(grid_w * scale_factor)
        new_h = int(grid_h * scale_factor)
        display_image = cv2.resize(grid_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_image = grid_image

    # Overlay Info
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed >= 1.0 and is_playing:
        fps = frame_count / elapsed
        font_scale = 0.7 * max(0.5, scale_factor)
        thickness = max(1, int(2 * scale_factor))
        cv2.putText(display_image, f"FPS: {fps:.1f} | PLAY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        frame_count = 0
        last_time = current_time
    elif not is_playing:
        font_scale = 0.7 * max(0.5, scale_factor)
        thickness = max(1, int(2 * scale_factor))
        cv2.putText(display_image, "PAUSED (E=Next, R=Prev)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    if is_playing: frame_count += 1

    cv2.imshow('Head Pose Grid', display_image)

    # Use waitKey(1) for responsiveness
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):
        is_playing = not is_playing
        last_time = time.time()
        frame_count = 0
    elif key == ord('e'): # Next Frame
        if not is_playing:
            for i, cap in enumerate(captures):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if ret: current_frames[i] = frame
    elif key == ord('r'): # Prev Frame
        if not is_playing:
            for i, cap in enumerate(captures):
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # Move back 2 frames (one to undo the current read, one to go back)
                new_pos = max(0, pos - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret: current_frames[i] = frame

for cap in captures:
    cap.release()
cv2.destroyAllWindows()
detector.close()