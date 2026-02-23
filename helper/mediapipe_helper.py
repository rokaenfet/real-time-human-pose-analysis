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