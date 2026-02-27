import torch
import torch.nn.functional as F
import cv2

def estimate_pose(model, tensor, device, num_bins=66):
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        yaw, pitch, roll = model(tensor)

    idx = torch.arange(num_bins, dtype=torch.float32, device=device).unsqueeze(0)
    yaw = torch.sum(F.softmax(yaw,1)*idx,1)*3-99
    pitch = torch.sum(F.softmax(pitch,1)*idx,1)*3-99
    roll = torch.sum(F.softmax(roll,1)*idx,1)*3-99
    return yaw.item(), pitch.item(), roll.item()

class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.state = None

    def update(self, vals):
        if self.state is None:
            self.state = vals
        else:
            self.state = tuple((1-self.alpha)*s + self.alpha*v for s,v in zip(self.state, vals))
        return self.state

def looking_away(y,p,r, yt,pt,rt):
    return abs(y)>yt or abs(p)>pt or abs(r)>rt

def draw(frame, bbox, y,p,r, warn=None):
    x,yb,w,h = bbox
    cv2.rectangle(frame,(x,yb),(x+w,yb+h),(255,0,0),2)
    cv2.putText(frame,f"Yaw:{y:.1f}",(x,yb-50),0,0.6,(0,255,0),2)
    cv2.putText(frame,f"Pitch:{p:.1f}",(x,yb-30),0,0.6,(0,255,0),2)
    cv2.putText(frame,f"Roll:{r:.1f}",(x,yb-10),0,0.6,(0,255,0),2)
    if warn:
        cv2.putText(frame,warn,(10,60),0,1,(0,0,255),2)
