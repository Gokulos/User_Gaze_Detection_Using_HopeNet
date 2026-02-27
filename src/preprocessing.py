import cv2
import mediapipe as mp
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    boxes = []
    if results.detections:
        for d in results.detections:
            bb = d.location_data.relative_bounding_box
            x = int(bb.xmin*w)
            y = int(bb.ymin*h)
            bw = int(bb.width*w)
            bh = int(bb.height*h)
            boxes.append((x,y,bw,bh))
    return boxes

def crop_preprocess(frame, bbox):
    x,y,w,h = bbox
    face = frame[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return preprocess(Image.fromarray(face))
