import cv2
import torch
from model import load_hopenet
from preprocessing import preprocess, detect_face
from utils import estimate_head_pose, draw_pose

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_hopenet(device=device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_bbox = detect_face(frame)
    if face_bbox is not None:
        x, y, w, h = face_bbox
        face_img = preprocess(frame[y:y+h, x:x+w])
        yaw, pitch, roll = estimate_head_pose(model, face_img)
        draw_pose(frame, yaw, pitch, roll, face_bbox)
    else:
        cv2.putText(frame, "Warning: No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
