"""NazarAI — HUD (4-stage pipeline display)"""
import cv2, time
import numpy as np

C_BG  = (12, 14, 18);   C_PAN = (22, 26, 34)
C_GRN = (0, 210, 80);   C_RED = (0, 40, 220)
C_ORG = (0, 160, 255);  C_GRY = (130, 130, 140)
C_WHT = (235, 235, 235); C_GLD = (0, 200, 255)
C_CYN = (200, 220, 0);  C_PRP = (200, 80, 200)
TIER_COL = {
    "CRITICAL": C_RED, "HIGH": C_RED,
    "MEDIUM": C_ORG, "LOW": C_CYN, "CLEAR": C_GRN
}

def _bar(f, x, y, w, h, r, col, bg=(40,44,52)):
    cv2.rectangle(f, (x,y), (x+w, y+h), bg, -1)
    fw = int(w * max(0., min(1., r)))
    if fw > 0: cv2.rectangle(f, (x,y), (x+fw, y+h), col, -1)

def _t(f, txt, x, y, sc=0.40, col=C_WHT, th=1):
    cv2.putText(f, str(txt), (x,y), cv2.FONT_HERSHEY_DUPLEX, sc, col, th, cv2.LINE_AA)

def draw_hud(frame, fps, s2_state, s3_state, alerter, detections,
             count, mog2_pass_rate=0.0):
    H, W = frame.shape[:2]
    hh   = 160
    c    = np.zeros((H+hh, W, 3), dtype=np.uint8)
    c[:H] = frame
    cv2.rectangle(c, (0,H), (W,H+hh), C_BG, -1)
    cv2.line(c, (0,H), (W,H), (30,34,44), 2)

    r1=H+22; r2=H+42; r3=H+62; r4=H+82; r5=H+104; rb=H+120

    # ── Col 1: System ────────────────────────────────────────────────────────
    _t(c, "NazarAI", 10, r1, 0.50, C_GRN)
    _t(c, f"FPS   {fps:5.1f}", 10, r2, 0.36, C_GRY)
    _t(c, f"Frame {count:07d}", 10, r3, 0.36, C_GRY)
    _t(c, f"Evts  {alerter.incident_count:04d}", 10, r4, 0.40, C_GLD)
    tier = alerter.active_tier
    _t(c, tier, 10, r5, 0.48, TIER_COL.get(tier, C_GRN), 2)
    cv2.line(c, (170,H+6), (170,H+hh-6), C_PAN, 1)

    # ── Col 2: Stage 1 MOG2 + YOLO ───────────────────────────────────────────
    persons = [d for d in detections if d["class_id"] == 0]
    threats = [d for d in detections if d["is_threat"]]
    yc = C_RED if threats else (C_ORG if persons else C_GRN)
    _t(c, "S1: MOG2 + YOLO", 182, r1, 0.36, C_GRY)
    _t(c, f"pass {mog2_pass_rate*100:.0f}% frames", 182, r2, 0.36, C_CYN)
    _t(c, f"persons {len(persons)}", 182, r3, 0.40, yc)
    if threats:
        _t(c, f"! {threats[0]['label'].upper()} !", 182, r4, 0.46, C_RED, 2)
    else:
        _t(c, "no weapon", 182, r4, 0.36, C_GRY)
    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        _bar(c, 182, rb, 140, 6, top["confidence"],
             C_RED if top["is_threat"] else C_GRN)
    cv2.line(c, (370,H+6), (370,H+hh-6), C_PAN, 1)

    # ── Col 3: Stage 2 MobileNetV3 ───────────────────────────────────────────
    s2_score = s2_state.get("score", 0.0)
    s2_lbl   = s2_state.get("label", "NORMAL")
    s2_conf  = s2_state.get("confirmed", False)
    s2c      = C_RED if s2_conf else (C_ORG if s2_lbl=="SUSPICIOUS" else C_GRN)
    _t(c, "S2: MobileNetV3", 382, r1, 0.36, C_GRY)
    _t(c, f"score {s2_score:.3f}", 382, r2, 0.40, s2c)
    _t(c, s2_lbl, 382, r3, 0.40, s2c, 2 if s2_conf else 1)
    _t(c, "CONFIRMED → S3" if s2_conf else "buffering...", 382, r4, 0.34,
       C_RED if s2_conf else C_GRY)
    _bar(c, 382, rb, 145, 6, s2_score, s2c)
    cv2.line(c, (580,H+6), (580,H+hh-6), C_PAN, 1)

    # ── Col 4: Stage 3 Qwen ───────────────────────────────────────────────────
    s3_cat  = s3_state.get("category", "SAFE")
    s3_sev  = s3_state.get("severity", 0)
    s3_desc = s3_state.get("description", "standby")
    s3_n    = s3_state.get("calls", 0)
    s3_busy = s3_state.get("busy", False)
    s3_susp = s3_state.get("suspicious", False)
    s3c     = C_RED if s3_susp else (C_ORG if s3_busy else C_GRY)
    _t(c, f"S3: Qwen2.5-VL  calls={s3_n}", 592, r1, 0.34, C_GRY)
    _t(c, f"[{s3_cat}] sev={s3_sev}", 592, r2, 0.40, s3c, 2 if s3_susp else 1)
    words = s3_desc.split(); l1=""; l2=""
    for w in words:
        if len(l1) < 28: l1 += w + " "
        else:             l2 += w + " "
    _t(c, l1.strip()[:32], 592, r3, 0.34, s3c)
    _t(c, l2.strip()[:32], 592, r4, 0.34, s3c)
    _t(c, "INFERRING..." if s3_busy else "idle", 592, r5, 0.32,
       C_ORG if s3_busy else C_GRY)
    cv2.line(c, (W-148,H+6), (W-148,H+hh-6), C_PAN, 1)

    # ── Col 5: Alert ─────────────────────────────────────────────────────────
    _t(c, "STATUS", W-136, r1, 0.36, C_GRY)
    if alerter.active:
        p = int(abs(np.sin(time.time()*4))*50)+30
        cv2.rectangle(c, (W-140,H+28), (W-6,H+60), (0,20,p+140), -1)
        _t(c, f" {alerter.active_tier} ", W-135, H+52, 0.48, C_WHT, 2)
        rsn = alerter.active_reason[:20]
        _t(c, rsn, W-140, r3, 0.28, C_ORG)
        _t(c, "frame saved", W-140, r4, 0.28, C_GRY)
        p2 = abs(np.sin(time.time()*5))*0.08
        ov = c.copy(); cv2.rectangle(ov,(0,0),(W,H),(0,0,180),-1)
        cv2.addWeighted(ov,p2,c,1-p2,0,c)
        cv2.rectangle(c,(4,4),(W-4,H-4),(0,0,220),4)
    else:
        _t(c, " CLEAR", W-132, H+50, 0.46, C_GRN)
        _t(c, "monitoring", W-136, r3, 0.32, C_GRY)
    return c
