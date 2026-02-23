from typing import List, Tuple, Optional, Dict

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

import cv2
import numpy as np
from pathlib import Path



class AngleSmoother:
    def __init__(self, alpha=0.2):
        """
        alpha: Smoothing factor (0.0 to 1.0)
        - Lower (0.05): Very smooth, but laggy.
        - Higher (0.5): Very responsive, but jittery.
        - Sweet spot: 0.1 to 0.25 for head pose.
        """
        self.alpha = alpha
        self.last_yaw = None
        self.last_pitch = None
        self.last_roll = None

    def update(self, yaw, pitch, roll):
        # Handle angle wrapping for Yaw (-180 to 180) to prevent flipping at the boundary
        if self.last_yaw is not None:
            diff = yaw - self.last_yaw
            if diff > 180:
                yaw -= 360
            elif diff < -180:
                yaw += 360
        
        # Apply EMA
        if self.last_yaw is None:
            self.last_yaw, self.last_pitch, self.last_roll = yaw, pitch, roll
        else:
            self.last_yaw = self.alpha * yaw + (1 - self.alpha) * self.last_yaw
            self.last_pitch = self.alpha * pitch + (1 - self.alpha) * self.last_pitch
            self.last_roll = self.alpha * roll + (1 - self.alpha) * self.last_roll
            
        return self.last_yaw, self.last_pitch, self.last_roll



class MediaPipeBodyPoseEstimator:
    """
    Encapsulates MediaPipe Pose Landmarker.
    Handles model loading and 3D pose estimation for the torso.
    """

    # Indices for Torso: 11(L Sh), 12(R Sh), 23(L Hip), 24(R Hip)
    TARGET_IDX = [11, 12, 23, 24]

    # Reference 3D model of a generic torso
    BODY_3D_POINTS = np.array([
        [-20.0, -25.0, 0.0],  # 11: Left Shoulder
        [ 20.0, -25.0, 0.0],  # 12: Right Shoulder
        [-15.0,  25.0, 0.0],  # 23: Left Hip
        [ 15.0,  25.0, 0.0],  # 24: Right Hip
    ], dtype=np.float32)

    def __init__(self, model_path: str = "./mediapipe_models/pose_landmarker.task", angle_smoothener_alpha: float = 0.2):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at: {self.model_path.absolute()}")
        
        self.detector = None
        self._initialize_model()
        self.smoother = AngleSmoother(alpha=angle_smoothener_alpha)

        self.smooth_rvec = None
        self.alpha = 0.2 # Adjust for smoothness vs. lag

    def _initialize_model(self):
        """Initializes the MediaPipe PoseLandmarker."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        try:
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Body detector: {e}")

    def close(self):
        """Releases resources."""
        if self.detector:
            self.detector.close()
            self.detector = None

    def detect_landmarks(self, frame: np.ndarray):
        """Detects pose landmarks in a given BGR frame."""
        if self.detector is None or frame is None or frame.size == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            return self.detector.detect(mp_image)
        except Exception as e:
            print(f"Body detection error: {e}")
            return None
        
    def estimate_body_pose(self, landmarks, image_shape: tuple[int, int, int]):
        if not landmarks: return None, None
        h, w, _ = image_shape
        
        # 1. 2D Image Points for PnP (to find Translation/Depth)
        img_points = np.array([
            [landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.TARGET_IDX
        ], dtype=np.float32)

        # 2. Camera Matrix
        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 3. Use PnP ONLY for Translation (tvec)
        # We use a fixed rvec (identity) temporarily just to get the position
        success, _, tvec = cv2.solvePnP(
            self.BODY_3D_POINTS, img_points, cam_matrix, np.zeros((4, 1)), 
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success: return None, None

        # 4. Manual Rotation Matrix Construction (The "Un-breakable" Part)
        # Extract 3D landmarks (using MediaPipe's relative Z)
        def get_vec(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

        l_sh, r_sh = get_vec(11), get_vec(12)
        l_hip, r_hip = get_vec(23), get_vec(24)

        # X-Axis: Vector from Left to Right Shoulder
        x_axis = r_sh - l_sh
        x_axis /= np.linalg.norm(x_axis)

        # Y-Axis: Vector from Mid-Shoulder to Mid-Hip
        y_axis = ((l_hip + r_hip) / 2) - ((l_sh + r_sh) / 2)
        y_axis /= np.linalg.norm(y_axis)

        # Z-Axis: Cross product (Perpendicular to the chest)
        # Resulting Z points forward/backward depending on order
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        # Re-orthogonalize Y to ensure a perfect 90-degree tripod
        y_axis = np.cross(z_axis, x_axis)

        # Build Rotation Matrix: Columns are the new axes
        rmat = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # Convert to Rotation Vector for cv2.projectPoints
        rvec, _ = cv2.Rodrigues(rmat)

        # 5. Extract Euler Angles using direct Trigonometry
        # We look at where our unit vectors are pointing in 3D space.
        
        # Yaw: How much the Z-axis (chest normal) turns left/right
        yaw = np.degrees(np.arctan2(z_axis[0], z_axis[2]))
        
        # Pitch: How much the Z-axis tips up/down
        # We use the hypotenuse of X and Z to get a stable angle
        pitch = np.degrees(np.arctan2(-z_axis[1], np.sqrt(z_axis[0]**2 + z_axis[2]**2)))
        
        # Roll: How much the X-axis (shoulders) tilts like a see-saw
        roll = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
        
        # Apply standard offsets to make "Standing straight" = 0,0,0
        # MediaPipe's coordinate system often requires these flips:
        yaw = -yaw
        roll = -roll

        # 6. Project 3D Axes for Drawing
        axis_len = 50 
        axis_points_3d = np.float32([
            [axis_len, 0, 0], 
            [0, axis_len, 0], 
            [0, 0, axis_len]])
        
        # Origin at center of chest
        origin_2d = np.mean(img_points, axis=0).astype(int)
        axis_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, cam_matrix, np.zeros((4, 1)))
        
        axes_data = {
            'origin': tuple(origin_2d),
            'x_end': tuple(axis_2d[0][0].astype(int)),
            'y_end': tuple(axis_2d[1][0].astype(int)),
            'z_end': tuple(axis_2d[2][0].astype(int))
        }

        return (yaw, pitch, roll), axes_data

    def draw_pose_info(self, frame: np.ndarray, phi_coords, axes_data):
        if phi_coords is None or axes_data is None:
            return frame

        # 1. Smoothen angles
        yaw, pitch, roll = self.smoother.update(*phi_coords)

        # 2. Draw Text (Top Left)
        info_text = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. Draw 3D Axes Tripod
        origin = axes_data['origin']
        # X: Red, Y: Green, Z: Blue
        cv2.line(frame, origin, axes_data['x_end'], (0, 0, 255), 3) # X
        cv2.line(frame, origin, axes_data['y_end'], (0, 255, 0), 3) # Y
        cv2.line(frame, origin, axes_data['z_end'], (255, 0, 0), 3) # Z (using BGR Blue)

        # Label the axes
        for key, label, color in [('x_end', 'X', (0,0,255)), ('y_end', 'Y', (0,255,0)), ('z_end', 'Z', (255,0,0))]:
            cv2.putText(frame, label, axes_data[key], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def draw_landmarks(self, frame: np.ndarray, detection_result, draw_skeleton: bool = True):
        if not detection_result or not detection_result.pose_landmarks: return frame
        
        annotated_image = np.copy(frame)
        # We must draw on RGB for MediaPipe's utilities to look right, then convert back
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in detection_result.pose_landmarks:
            if draw_skeleton:
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=pose_landmark_style,
                    connection_drawing_spec=pose_connection_style)

        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    


class MediaPipeHeadPoseEstimator:
    """
    Encapsulates MediaPipe Face Landmark.
    Handles model loading.
    """

    TARGET_IDX = [
        1,      # Nose
        152,    # Chin
        10,     # Forehead
        33,     # L Eye Inner
        133,    # L Eye Outer
        362,    # R Eye Inner
        263,    # R Eye Outer
        468,    # L Iris
        473     # R Iris
    ]

    FACE_3D_POINTS = np.array([
        [0.0,    0.0,    0.0],    # 1. Nose tip
        [0.0,    100.0, 30.0],  # 152. Chin
        [0.0,    -95.0,   25.0],  # 10. Forehead
        [-25.0,  -45.0,   35.0],  # 33. L Eye Inner
        [-55.0,  -45.0,   35.0],  # 133. L Eye Outer
        [25.0,   -45.0,   35.0],  # 362. R Eye Inner
        [55.0,   -45.0,   35.0],  # 263. R Eye Outer
        [-40.0,  -45.0,   45.0],  # 468. L Iris 
        [40.0,   -45.0,   45.0]   # 473. R Iris 
    ], dtype=np.float32)
    
    def __init__(self, model_path: str = "./mediapipe_models/face_landmarker.task", angle_smoothener_alpha:float = .2):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at: {self.model_path.absolute()}")
        
        self.detector = None
        self._initialize_model()
        self.smoother = AngleSmoother(alpha=angle_smoothener_alpha)

    def _initialize_model(self):
        """Initializes the MediaPipe FaceLandmarker."""
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        try:
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe detector: {e}")

    def close(self):
        """Properly closes the detector to release resources."""
        if self.detector:
            self.detector.close()
            self.detector = None

    def detect_landmarks(self, frame: np.ndarray) -> FaceLandmarkerResult | None:
        """
        Detects face landmarks in a given BGR frame.
        
        Args:
            frame: Input image as a numpy array (BGR format from OpenCV).
            
        Returns:
            The detection_result object containing face_landmarks list.
            Returns None if no face is detected or on error.
        """
        if self.detector is None or frame is None or frame.size == 0:
            return None

        # Convert BGR (OpenCV) to RGB (MediaPipe expects RGB for SRGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = self.detector.detect(mp_image)
            # Return the full result object so we can access face_landmarks later
            return detection_result
        except Exception as e:
            print(f"Detection error: {e}")
            return None
        
    def estimate_head_pose(self, landmarks:List[List[NormalizedLandmark]], image_shape:tuple[int, int, int]):
        """
        Calculates rotation vectors and projects 3D axes onto the 2D image.
        Returns: (yaw, pitch, roll) in degrees AND the projected 2D axis points for drawing.
        """
        if not landmarks: return None, None
        h, w, _ = image_shape
        
        img_points = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] 
                          for idx in self.TARGET_IDX], dtype=np.float32)

        focal_length = w 
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.FACE_3D_POINTS, 
            img_points, 
            cam_matrix, 
            np.zeros((4, 1)), 
            flags=cv2.SOLVEPNP_EPNP
            )
        # cv2.solvePnP flags > SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_UPNP

        if not success:
            return None, None

        # Rotation Mat
        rot_matrix, _ = cv2.Rodrigues(rvec)
        # Euler Angle
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)

        # Map to Pitch, Yaw, Roll based on OpenCV camera coords
        # These multipliers adjust for the Y-down flip
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Optional: Adjusting signs so 'Straight' is ~0,0,0
        # If looking UP gives positive pitch, use: pitch = -pitch
        
        # 5. Project Axes
        axis_len = 100
        axis_points_3d = np.float32([
            [axis_len, 0, 0],   # X-axis (Red)
            [0, axis_len, 0],   # Y-axis (Green)
            [0, 0, -axis_len]   # Z-axis (Blue)
        ])
        
        # Use landmark 1 (Nose) as the base for the drawing
        axis_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, cam_matrix, np.zeros((4, 1)))
    
        axes_data = {
            'nose': tuple(img_points[0].astype(int)),
            'x_end': tuple(axis_2d[0][0].astype(int)),
            'y_end': tuple(axis_2d[1][0].astype(int)),
            'z_end': tuple(axis_2d[2][0].astype(int))
        }

        return (yaw, pitch, roll), axes_data

    def draw_pose_info(self, frame:np.ndarray, phi_coords, axes_data):
        """
        Draws the Euler angles text and the 3D projected axes onto the frame.
        """
        if phi_coords is None or axes_data is None:
            return frame

        # 1. Smoothen and Unpack
        yaw, pitch, roll = self.smoother.update(*phi_coords)

        # 2. Draw Text Overlay
        info_text = f"Yaw: {yaw:.1f} | Pitch: {pitch:.1f} | Roll: {roll:.1f}"
        cv2.putText(frame, info_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. Draw 3D Axes
        # Define colors and labels for iteration
        # X: Red, Y: Green, Z: Blue
        draw_configs = [
            ('x_end', (0, 0, 255), "X"),
            ('y_end', (0, 255, 0), "Y"),
            ('z_end', (255, 0, 0), "Z")
        ]

        for key, color, label in draw_configs:
            cv2.line(frame, axes_data['nose'], axes_data[key], color, 3)
            cv2.putText(frame, label, axes_data[key], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, (yaw, pitch, roll)

    def draw_landmarks(
            self, 
            frame: np.ndarray, 
            detection_result: FaceLandmarkerResult,
            draw_index_of_interest: bool = True,
            draw_tesselation: bool = True,
            draw_contours: bool = True,
            draw_iris: bool = True
            ) -> np.ndarray:
        """
        Draws the detected landmarks on the frame using MediaPipe's official drawing utils.
        
        Args:
            frame: Input image (BGR).
            detection_result: The result object returned by detect_landmarks.
            
        Returns:
            The frame with landmarks drawn on it (RGB format converted back to BGR).
            If no landmarks, returns original frame.
        """
        if detection_result is None or not detection_result.face_landmarks:
            return frame
        
        frame_h, frame_w, _ = frame.shape
        # The drawing utils expect an RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = np.copy(rgb_frame)
        
        # Loop through detected faces (usually just 1 based on our config)
        for face_landmarks in detection_result.face_landmarks:
            # draw index of interest
            if draw_index_of_interest:
                for idx,lm in enumerate(face_landmarks):
                    if idx in MediaPipeHeadPoseEstimator.TARGET_IDX:
                        # print(idx, [lm.x, lm.y, lm.z])
                        cv2.circle(annotated_image, (int(lm.x*frame_w), int(lm.y*frame_h)), 5, (255, 0, 0), -1)
                        cv2.putText(annotated_image, str(idx), (int(lm.x*frame_w + 10), int(lm.y*frame_h - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if draw_tesselation:
                # 1. Draw Tesselation (The full mesh)
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
                )
            
            if draw_contours:
                # 2. Draw Contours (The outline of lips, eyes, eyebrows)
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
                )
            
            if draw_iris:
                # 3. Draw Left Iris
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
                )
                
                # 4. Draw Right Iris
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # Convert back to BGR for OpenCV display consistency
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)