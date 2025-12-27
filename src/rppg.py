"""
Remote photoplethysmography (rPPG) signal extraction.

This module intentionally avoids end-to-end black-box models.
Instead, it exposes:
- Raw signal accumulation
- Explicit band-pass filtering
- Simple variability metrics

This makes the physiological signal interpretable and debuggable.
"""

import numpy as np
from scipy.signal import butter, filtfilt
"""
Remote photoplethysmography (rPPG) extraction.

Uses green-channel mean for better physiological sensitivity.
"""

import numpy as np
from scipy.signal import butter, filtfilt


class RPPGExtractor:
    def __init__(self, fs=30):
        self.fs = fs
        self.signal = []

    def add_frame(self, frame):
        # Green channel is most informative for rPPG
        green_mean = frame[:, :, 1].mean()
        self.signal.append(green_mean)

    def _bandpass(self, sig, low=0.7, high=4.0):
        b, a = butter(
            3,
            [low / (self.fs / 2), high / (self.fs / 2)],
            btype="band"
        )
        return filtfilt(b, a, sig)

    def get_hr_variability(self):
        if len(self.signal) < self.fs * 5:
            return 0.0

        window = np.array(self.signal[-self.fs * 10:])
        filtered = self._bandpass(window)
        return float(np.std(filtered))

class RPPGExtractor__:
    """
    Maintains a rolling rPPG signal and computes variability metrics.

    We do NOT aim for medical-grade heart rate accuracy.
    The goal is to detect relative physiological arousal trends.
    """

    def __init__(self, fs=30):
        """
        Args:
            fs (int): Approximate frame rate of the video stream
        """
        self.fs = fs
        self.signal = []

    def add_frame(self, roi_rgb_mean):
        """
        Append the mean RGB value of a facial ROI for one frame.

        Args:
            roi_rgb_mean (float): Mean pixel intensity (or channel-projected)
        """
        self.signal.append(roi_rgb_mean)

    def _bandpass(self, sig, low=0.7, high=4.0):
        """
        Apply a physiological band-pass filter.

        Typical human heart rates fall within ~0.7–4 Hz (40–240 BPM).
        """
        b, a = butter(
            3,
            [low / (self.fs / 2), high / (self.fs / 2)],
            btype="band"
        )
        return filtfilt(b, a, sig)

    def get_hr_variability(self):
        """
        Compute short-term variability of the filtered rPPG signal.

        Returns:
            float: Standard deviation of the filtered signal
        """
        # Require sufficient temporal context
        if len(self.signal) < self.fs * 5:
            return 0.0

        window = np.array(self.signal[-self.fs * 10:])
        filtered = self._bandpass(window)

        # Variability correlates better with stress than absolute HR
        return np.std(filtered)

