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

# --- Configuration ---
FACE_MODEL_PATH = 'face_landmarker.task' 
BASE_DIR = 'Head_Pose_Database_UPNA'
TARGET_FRAME_WIDTH = 320   # Width of each individual video cell
TARGET_FRAME_HEIGHT = 240  # Height of each individual video cell
PADDING_COLOR = (0, 0, 0)  # Black padding for empty grid slots
GAP_SIZE = 5               # Pixels between videos

# Landmark Indices for PnP
FACE_MESH_INDICES = [1, 152, 234, 454, 33, 362]

# 3D Model Coordinates (cm)
OBJECT_POINTS = np.array([
    [0.0, 0.0, 0.0],        # Nose
    [0.0, -7.0, 0.0],       # Chin
    [-5.0, -2.0, -5.0],     # Left Ear
    [5.0, -2.0, -5.0],      # Right Ear
    [-2.5, -3.0, -4.0],     # Left Eye
    [2.5, -3.0, -4.0],      # Right Eye
], dtype=np.float32)

def is_landmark_visible(landmark, nose_landmark, threshold=0.3):
    if landmark.z is None or nose_landmark.z is None:
        return False
    return (landmark.z - nose_landmark.z) < threshold

def calculate_euler_angles(rvec, tvec):
    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0,0]**2 + rot_matrix[1,0]**2)
    
    if sy > 1e-6:
        x = np.arctan2(rot_matrix[2,1], rot_matrix[2,2])
        y = np.arctan2(-rot_matrix[2,0], sy)
        z = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])
    else:
        x = np.arctan2(-rot_matrix[1,2], rot_matrix[1,1])
        y = np.arctan2(-rot_matrix[2,0], sy)
        z = 0
        
    return np.degrees(x), np.degrees(y), np.degrees(z)

def process_frame(frame, detector, stream_id):
    """Processes a single frame and returns the annotated image."""
    # Ensure frame is correct size before processing
    h, w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            image_points = []
            world_points = []
            valid_detection = True
            
            nose_lm = face_landmarks[1]
            
            for i, idx in enumerate(FACE_MESH_INDICES):
                lm = face_landmarks[idx]
                if not is_landmark_visible(lm, nose_lm):
                    valid_detection = False
                    break
                
                # Scale landmarks to current resized frame dimensions
                image_points.append([lm.x * w, lm.y * h])
                world_points.append(OBJECT_POINTS[i])
            
            if valid_detection:
                image_points = np.array(image_points, dtype=np.float32)
                world_points = np.array(world_points, dtype=np.float32)

                focal_length = 1.0 * w 
                cam_matrix = np.array([
                    [focal_length, 0, w/2],
                    [0, focal_length, h/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                dist_coeffs = np.zeros((4,1))

                try:
                    success, rvec, tvec = cv2.solvePnP(world_points, image_points, cam_matrix, dist_coeffs)
                    
                    if success:
                        pitch, yaw, roll = calculate_euler_angles(rvec, tvec)
                        
                        nose_2d, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, cam_matrix, dist_coeffs)
                        nose_2d = (int(nose_2d[0][0][0]), int(nose_2d[0][0][1]))

                        # X Axis (Red)
                        x_axis_3d = np.float32([[30, 0, 0]]) # Scaled down for smaller frames
                        x_axis_2d, _ = cv2.projectPoints(x_axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
                        x_end = (int(x_axis_2d[0][0][0]), int(x_axis_2d[0][0][1]))
                        cv2.line(frame, nose_2d, x_end, (0, 0, 255), 1) 

                        # Y Axis (Green)
                        y_axis_3d = np.float32([[0, -30, 0]])
                        y_axis_2d, _ = cv2.projectPoints(y_axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
                        y_end = (int(y_axis_2d[0][0][0]), int(y_axis_2d[0][0][1]))
                        cv2.line(frame, nose_2d, y_end, (0, 255, 0), 1) 

                        # Z Axis (Blue)
                        arrow_length = 50 
                        yaw_rad = np.radians(yaw)
                        pitch_rad = np.radians(pitch)
                        dx = arrow_length * np.sin(yaw_rad) * np.cos(pitch_rad)
                        dy = -arrow_length * np.sin(pitch_rad)

                        z_end = (int(nose_2d[0] + dx), int(nose_2d[1] + dy))
                        cv2.arrowedLine(frame, nose_2d, z_end, (255, 0, 0), 1, tipLength=0.3)
                        cv2.circle(frame, z_end, 3, (255, 0, 0), -1)

                        # Text Overlay
                        info = f"Y:{yaw:.0f} P:{pitch:.0f}"
                        cv2.putText(frame, info, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        
                except cv2.error:
                    pass
    
    # Label
    cv2.putText(frame, f"#{stream_id}", (2, frame.shape[0] - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def create_grid(frames, cols, target_w, target_h, gap):
    """Arranges frames into a grid with padding and gaps."""
    rows = math.ceil(len(frames) / cols)
    
    # Calculate total grid dimensions
    total_w = cols * target_w + (cols - 1) * gap
    total_h = rows * target_h + (rows - 1) * gap
    
    # Create blank canvas
    grid_canvas = np.full((total_h, total_w, 3), PADDING_COLOR, dtype=np.uint8)
    
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        
        y_start = r * (target_h + gap)
        y_end = y_start + target_h
        x_start = c * (target_w + gap)
        x_end = x_start + target_w
        
        grid_canvas[y_start:y_end, x_start:x_end] = frame
        
    return grid_canvas

# --- Main Execution ---

# 1. Get User Input
user_num = input("Enter User Number (e.g., 01, 13): ").strip()
user_dir = os.path.join(BASE_DIR, f"User_{user_num}")

if not os.path.exists(user_dir):
    print(f"Error: Directory '{user_dir}' not found.")
    exit()

# 2. Find all video files
video_files = sorted(glob.glob(os.path.join(user_dir, "*.mp4")))
if not video_files:
    video_files = sorted(glob.glob(os.path.join(user_dir, "*.avi")))
if not video_files:
    video_files = sorted(glob.glob(os.path.join(user_dir, "*.mov")))

if not video_files:
    print(f"No video files found in '{user_dir}'.")
    exit()

print(f"Found {len(video_files)} videos. Loading...")

# 3. Initialize Detector
base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
options = face_landmarker.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

try:
    detector = face_landmarker.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 4. Initialize Captures
captures = []
for path in video_files:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: Could not open {path}")
        continue
    captures.append(cap)

if not captures:
    print("No valid videos to play.")
    exit()

# 5. Calculate Grid Dimensions
num_videos = len(captures)
cols = math.ceil(math.sqrt(num_videos))
rows = math.ceil(num_videos / cols)

print(f"Grid Layout: {rows} rows x {cols} cols (Total slots: {rows*cols})")

last_time = time.time()
frame_count = 0

running = True
while running:
    processed_frames = []
    
    for i, cap in enumerate(captures):
        ret, frame = cap.read()
        
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                # Fallback black frame
                frame = np.zeros((TARGET_FRAME_HEIGHT, TARGET_FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Resize to fixed cell size
        resized_frame = cv2.resize(frame, (TARGET_FRAME_WIDTH, TARGET_FRAME_HEIGHT))
        
        # Process
        proc_frame = process_frame(resized_frame, detector, i+1)
        processed_frames.append(proc_frame)

    # If we have fewer frames than grid slots, add black padding frames
    total_slots = rows * cols
    while len(processed_frames) < total_slots:
        blank = np.zeros((TARGET_FRAME_HEIGHT, TARGET_FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank, "Empty", (10, TARGET_FRAME_HEIGHT//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        processed_frames.append(blank)

    # Create Grid
    grid_image = create_grid(processed_frames, cols, TARGET_FRAME_WIDTH, TARGET_FRAME_HEIGHT, GAP_SIZE)
    
    # Add Global FPS
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        cv2.putText(grid_image, f"System FPS: {fps:.1f} | Videos: {num_videos}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        frame_count = 0
        last_time = current_time
    frame_count += 1

    cv2.imshow('Head Pose Grid Analysis', grid_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
detector.close()