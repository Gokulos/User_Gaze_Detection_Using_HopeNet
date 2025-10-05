import cv2
import torch
import torch.nn.functional as F

def estimate_head_pose(model, face_img, num_bins=66):
    face_img = face_img.unsqueeze(0)
    with torch.no_grad():
        yaw, pitch, roll = model(face_img)

    idx_tensor = torch.arange(num_bins, dtype=torch.float32).unsqueeze(0)
    yaw = torch.sum(F.softmax(yaw, dim=1) * idx_tensor, dim=1) * 3 - 99
    pitch = torch.sum(F.softmax(pitch, dim=1) * idx_tensor, dim=1) * 3 - 99
    roll = torch.sum(F.softmax(roll, dim=1) * idx_tensor, dim=1) * 3 - 99

    return yaw.item(), pitch.item(), roll.item()

def is_looking_away(yaw, pitch, roll, yaw_thr=20, pitch_thr=15, roll_thr=10):
    return abs(yaw) > yaw_thr or abs(pitch) > pitch_thr or abs(roll) > roll_thr

def draw_pose(frame, yaw, pitch, roll, face_bbox):
    x, y, w, h = face_bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Roll: {roll:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if is_looking_away(yaw, pitch, roll):
        cv2.putText(frame, "Warning: Looking away!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
