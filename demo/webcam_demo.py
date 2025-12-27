from src.video import VideoStream
from src.landmarks import FaceLandmarkDetector
from src.rppg import RPPGExtractor
from src.behavior import HeadMotionAnalyzer
from src.fusion import StressFusion
from src.visualizer import Visualizer
import cv2
import time

vs = VideoStream()
lm = FaceLandmarkDetector()
rppg = RPPGExtractor()
motion = HeadMotionAnalyzer()
fusion = StressFusion()
viz = Visualizer()

while True:
    frame, ts = vs.read()
    if frame is None:
        break

    landmarks = lm.detect(frame, int(ts * 1000))
    if landmarks is not None:
        motion.update(landmarks)

    stress = fusion.update(
        hr_var=rppg.get_hr_variability(),
        motion=motion.motion_energy()
    )

    frame = viz.draw(frame, stress)
    cv2.imshow("Stress Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
