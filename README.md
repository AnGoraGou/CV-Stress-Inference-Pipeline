# CV-Stress-Inference-Pipeline
CV-SIP is a real-time computer vision pipeline for stress inference that fuses rPPG-derived physiological signals with facial behavioral markers. It applies classical signal processing and custom fusion logic to produce an interpretable stress index without black-box emotion models.
## Multi-Signal Stress Detection (Computer Vision)

This project implements a real-time stress detection system by fusing
physiological and behavioral signals extracted from facial video.

## Signals Used
- Physiological: Remote Photoplethysmography (rPPG) from forehead ROI
- Behavioral: Eye Aspect Ratio (EAR) based blink rate

## Stress Index Logic
Both signals are normalized and fused into a single Stress Index (0–100):

Stress = 0.6 × rPPG_variability + 0.4 × Blink_rate

The weights reflect higher confidence in physiological arousal while
still incorporating behavioral stress markers.

## Live Demo
Run locally:
```bash
python src/webcam_local.py
