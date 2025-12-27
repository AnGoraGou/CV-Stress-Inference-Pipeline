"""
Face landmark extraction using MediaPipe Tasks API.

We explicitly use the Tasks API instead of the deprecated `solutions`
subpackage to ensure forward compatibility with newer MediaPipe releases.

This module exposes only normalized landmark coordinates and hides all
MediaPipe-specific complexity from downstream logic.
"""
"""
Face landmark extraction using MediaPipe Tasks API.
"""

import mediapipe as mp
import numpy as np
import os


class FaceLandmarkDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "face_landmarker.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "face_landmarker.task not found. "
                "Download it into the models/ directory."
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if not result.face_landmarks:
            return None

        return np.array([
            (lm.x, lm.y, lm.z)
            for lm in result.face_landmarks[0]
        ])



class FaceLandmarkDetector__:
    """
    Thin wrapper around MediaPipe FaceLandmarker.

    Outputs:
        - 3D normalized landmarks in image-relative coordinates
    """

    def __init__(self):
        """
        Initializes MediaPipe FaceLandmarker in VIDEO mode.

        NOTE:
        - VIDEO mode requires monotonically increasing timestamps
        - The face_landmarker.task file must be downloaded separately
        """
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "face_landmarker.task"
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        #options = FaceLandmarkerOptions(
        #    base_options=BaseOptions(
        #        model_asset_path="face_landmarker.task"
        #    ),
        #    running_mode=VisionRunningMode.VIDEO,
        #    num_faces=1
        #)

        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        """
        Detect face landmarks for a single video frame.

        Args:
            frame (np.ndarray): BGR image
            timestamp_ms (int): Timestamp in milliseconds

        Returns:
            np.ndarray | None:
                Array of shape (N, 3) containing (x, y, z) landmarks,
                or None if no face is detected.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if not result.face_landmarks:
            return None

        # Convert landmark proto objects into a simple NumPy array
        return np.array([
            (lm.x, lm.y, lm.z)
            for lm in result.face_landmarks[0]
        ])

