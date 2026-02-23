from typing import List, Tuple, Optional, Dict

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import cv2
import numpy as np
from pathlib import Path

class MediaPipeHeadPoseEstimator:
    """
    Encapsulates MediaPipe Face Landmark.
    Handles model loading.
    """

    
    def __init__(self, model_path: str = "./mediapipe_models/face_landmarker.task"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at: {self.model_path.absolute()}")
        
        self.detector = None
        self._initialize_model()

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

    def detect_landmarks(self, frame: np.ndarray):
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

    def draw_landmarks(self, frame: np.ndarray, detection_result) -> np.ndarray:
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

        # The drawing utils expect an RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = np.copy(rgb_frame)

        # Loop through detected faces (usually just 1 based on our config)
        for face_landmarks in detection_result.face_landmarks:
            
            # 1. Draw Tesselation (The full mesh)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # 2. Draw Contours (The outline of lips, eyes, eyebrows)
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            )
            
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