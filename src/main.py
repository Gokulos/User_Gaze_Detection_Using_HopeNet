import cv2, time, torch
from config import cfg
from model import load_hopenet
from preprocessing import detect_faces, crop_preprocess
from utils import estimate_pose, EMA, looking_away, draw
from logger import create_logger, log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_hopenet(cfg.model_path, cfg.num_bins, device)

cap = cv2.VideoCapture(cfg.camera_id)
smoother = EMA(cfg.ema_alpha)

f, writer, path = create_logger(cfg)
start = time.time()

away_start=None
away=False
last_face=time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    t = time.time()-start
    boxes = detect_faces(frame)

    if len(boxes)>1:
        log(writer,t,"MULTI_FACE",extra=str(len(boxes)))
        cv2.putText(frame,"MULTI FACE!",(10,60),0,1,(0,0,255),2)

    elif len(boxes)==0:
        if time.time()-last_face>cfg.no_face_grace_s:
            log(writer,t,"NO_FACE")
            cv2.putText(frame,"NO FACE!",(10,60),0,1,(0,0,255),2)

    else:
        last_face=time.time()
        bbox=boxes[0]
        face=crop_preprocess(frame,bbox)
        y,p,r = estimate_pose(model,face,device,cfg.num_bins)
        y,p,r = smoother.update((y,p,r))

        if looking_away(y,p,r,cfg.yaw_thr,cfg.pitch_thr,cfg.roll_thr):
            if not away:
                away=True
                away_start=time.time()
                log(writer,t,"AWAY_START",y,p,r)
            elif time.time()-away_start>cfg.away_required_s:
                cv2.putText(frame,"LOOKING AWAY!",(10,60),0,1,(0,0,255),2)
                log(writer,t,"AWAY_FLAG",y,p,r)
        else:
            if away:
                log(writer,t,"AWAY_END",y,p,r)
            away=False

        draw(frame,bbox,y,p,r)

    cv2.imshow("Monitor",frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release()
f.close()
cv2.destroyAllWindows()
print("log:",path)
