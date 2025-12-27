import cv2


class Visualizer:
    def draw_face(self, frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(l[0]*w), int(l[1]*h)) for l in landmarks]

        xs, ys = zip(*pts)
        cv2.rectangle(
            frame,
            (min(xs), min(ys)),
            (max(xs), max(ys)),
            (0, 255, 0), 2
        )

        for x, y in pts:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        return frame

    def draw_metrics(self, frame, stress, motion, hr_var, blink_rate):
        y = 30
        lines = [
            f"Stress: {stress}",
            f"Motion: {motion:.4f}",
            f"rPPG var: {hr_var:.4f}",
            f"Blink rate: {blink_rate:.1f}/min"
        ]

        for text in lines:
            cv2.putText(
                frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )
            y += 28

        return frame

    def draw_no_face(self, frame):
        cv2.putText(
            frame, "No face detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )
        return frame

