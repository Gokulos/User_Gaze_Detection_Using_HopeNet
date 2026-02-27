from dataclasses import dataclass
import os

@dataclass
class Config:
    camera_id: int = 0
    model_path: str = "hopenet_robust_alpha1.pkl"
    num_bins: int = 66

    yaw_thr: float = 20.0
    pitch_thr: float = 15.0
    roll_thr: float = 10.0

    ema_alpha: float = 0.2
    no_face_grace_s: float = 1.0
    away_required_s: float = 1.2

    log_dir: str = "logs"
    session_name: str = "session"

cfg = Config()
os.makedirs(cfg.log_dir, exist_ok=True)
