import csv, os, time
from datetime import datetime

def create_logger(cfg):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg.log_dir, f"{cfg.session_name}_{ts}.csv")
    f = open(path,"w",newline="",encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["t","event","yaw","pitch","roll","extra"])
    writer.writeheader()
    return f, writer, path

def log(writer, t, event, y=None,p=None,r=None, extra=""):
    writer.writerow(dict(t=f"{t:.2f}",event=event,yaw=y,pitch=p,roll=r,extra=extra))
