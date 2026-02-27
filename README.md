# Real-Time Head Pose Monitoring (Webcam) — Looking-Away Flagging

A real-time head pose estimation pipeline using a laptop webcam to detect whether a user is looking away from the screen.  
The system detects a face, estimates yaw/pitch/roll using **Hopenet**, and displays warnings when head pose exceeds configurable thresholds.

> **Use case examples:** attention monitoring, UX studies, online proctoring research prototypes, HCI experiments.  

---

## Demo


---

## Installation

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
# .venv\Scripts\activate    # (Windows)
```
### 2) Install Dependencies
```
pip install -r requirements.txt
```
---


## Project Structure
```
├── main.py
├── model.py
├── preprocessing.py
├── utils.py
├── hopenet.py
├── hopenet_robust_alpha1.pkl
├── requirements.txt
└── README.md
```
---
## Model Weights
The hopenet model architecture as well as weights have been borrowed from this repository: https://github.com/human-analysis/RankGAN/blob/master/models/
Files: hopenet.py, hopenet_robust_alpha1.pkl

---

## How It Works
```
1) Face Detection

Each webcam frame is converted to grayscale and passed to a frontal face detector.
If a face is detected, the first bounding box is selected.

2) Preprocessing

The face crop is resized to 224×224, converted to tensor, and normalized.

3) Head Pose Estimation (Yaw/Pitch/Roll)

The preprocessed face is passed through Hopenet to predict yaw/pitch/roll angles in degrees.

4) Flagging Rule

A warning is displayed when angles exceed thresholds:

|yaw| > 20

|pitch| > 15

|roll| > 10

These are configurable in utils.py:

```

---

