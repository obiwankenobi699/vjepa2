"""HawkWatch — Professional HUD Renderer"""
import cv2, time
import numpy as np

C_BG    = (12,  14,  18)
C_PANEL = (22,  26,  34)
C_GREEN = (0,  210,  80)
C_RED   = (0,   40, 220)
C_ORANGE= (0,  160, 255)
C_GREY  = (130, 130, 140)
C_WHITE = (235, 235, 235)
C_GOLD  = (0,  200, 255)

def _bar(f,x,y,w,h,ratio,col,bg=(40,44,52)):
    cv2.rectangle(f,(x,y),(x+w,y+h),bg,-1)
    fw=int(w*max(0.,min(1.,ratio)))
    if fw>0: cv2.rectangle(f,(x,y),(x+fw,y+h),col,-1)

def _t(f,txt,x,y,sc=0.50,col=C_WHITE,th=1):
    cv2.putText(f,txt,(x,y),cv2.FONT_HERSHEY_DUPLEX,sc,col,th,cv2.LINE_AA)

def draw_hud(frame,fps,vjepa_state,alerter,detections,count):
    H,W=frame.shape[:2]
    hh=115
    canvas=np.zeros((H+hh,W,3),dtype=np.uint8)
    canvas[:H]=frame
    cv2.rectangle(canvas,(0,H),(W,H+hh),C_BG,-1)
    cv2.line(canvas,(0,H),(W,H),(30,34,44),2)

    # col 1 — branding + stats
    _t(canvas,"HAWKWATCH",14,H+22,0.58,C_GREEN,1)
    _t(canvas,f"FPS    {fps:5.1f}",14,H+44,0.44,C_GREY)
    _t(canvas,f"FRAME  {count:07d}",14,H+62,0.44,C_GREY)
    _t(canvas,f"INCIDENTS  {alerter.incident_count:04d}",14,H+84,0.48,C_GOLD)
    cv2.line(canvas,(195,H+6),(195,H+hh-6),C_PANEL,1)

    # col 2 — YOLO
    persons = [d for d in detections if d["class_id"]==0]
    threats = [d for d in detections if d["is_threat"]]
    o_col   = C_RED if threats else (C_ORANGE if persons else C_GREEN)
    _t(canvas,"OBJECTS",208,H+22,0.44,C_GREY)
    _t(canvas,f"Persons  {len(persons)}",208,H+42,0.52,o_col)
    if threats:
        _t(canvas,f"WEAPON : {threats[0]['label'].upper()}",208,H+64,0.52,C_RED,2)
    else:
        _t(canvas,"Weapon   none",208,H+64,0.50,C_GREY)
    if detections:
        top=max(detections,key=lambda d:d["confidence"])
        _t(canvas,f"{top['label']} conf",208,H+84,0.40,C_GREY)
        _bar(canvas,208,H+93,170,7,top["confidence"],
             C_RED if top["is_threat"] else C_GREEN)
    cv2.line(canvas,(420,H+6),(420,H+hh-6),C_PANEL,1)

    # col 3 — V-JEPA
    lbl  = vjepa_state["label"]
    conf = vjepa_state["confidence"]
    susp = vjepa_state["suspicious"]
    a_c  = C_RED if susp else C_GREY
    _t(canvas,"BEHAVIOR  (V-JEPA 2)",433,H+22,0.44,C_GREY)
    _t(canvas,lbl[:44],433,H+42,0.46,a_c,1+int(susp))
    st = "SUSPICIOUS" if susp else ("analyzing..." if vjepa_state["busy"] else "normal")
    _t(canvas,st,433,H+64,0.46,C_RED if susp else C_GREEN)
    _bar(canvas,433,H+78,210,7,conf,C_RED if susp else C_GREEN)
    cv2.line(canvas,(W-175,H+6),(W-175,H+hh-6),C_PANEL,1)

    # col 4 — alert
    _t(canvas,"STATUS",W-160,H+22,0.44,C_GREY)
    if alerter.active:
        pulse=int(abs(np.sin(time.time()*4))*50)+30
        cv2.rectangle(canvas,(W-162,H+30),(W-8,H+58),(0,20,pulse+140),-1)
        _t(canvas,"  ALERT  ",W-155,H+50,0.56,C_WHITE,2)
        _t(canvas,alerter.active_reason[:26],W-162,H+72,0.36,C_ORANGE)
        _t(canvas,"frame saved",W-162,H+90,0.36,C_GREY)
        # frame border flash
        p=abs(np.sin(time.time()*5))*0.10
        ov=canvas.copy()
        cv2.rectangle(ov,(0,0),(W,H),(0,0,200),-1)
        cv2.addWeighted(ov,p,canvas,1-p,0,canvas)
        bw=4
        cv2.rectangle(canvas,(bw,bw),(W-bw,H-bw),(0,0,220),bw)
    else:
        _t(canvas,"  CLEAR  ",W-155,H+50,0.56,C_GREEN)
        _t(canvas,"monitoring",W-158,H+72,0.40,C_GREY)

    return canvas
