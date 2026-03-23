"""HawkWatch — HUD Renderer (Dual Model)"""
import cv2, time
import numpy as np

C_BG     = (12,  14,  18)
C_PANEL  = (22,  26,  34)
C_GREEN  = (0,  210,  80)
C_RED    = (0,   40, 220)
C_ORANGE = (0,  160, 255)
C_GREY   = (130, 130, 140)
C_WHITE  = (235, 235, 235)
C_GOLD   = (0,  200, 255)
C_CYAN   = (200, 220,  0)

def _bar(f, x, y, w, h, ratio, col, bg=(40,44,52)):
    cv2.rectangle(f, (x,y), (x+w, y+h), bg, -1)
    fw = int(w * max(0., min(1., ratio)))
    if fw > 0:
        cv2.rectangle(f, (x,y), (x+fw, y+h), col, -1)

def _t(f, txt, x, y, sc=0.48, col=C_WHITE, th=1):
    cv2.putText(f, txt, (x,y), cv2.FONT_HERSHEY_DUPLEX, sc, col, th, cv2.LINE_AA)

def draw_hud(frame, fps, state, alerter, detections, count):
    H, W = frame.shape[:2]
    hh   = 145   # taller HUD — two rows
    canvas = np.zeros((H + hh, W, 3), dtype=np.uint8)
    canvas[:H] = frame
    cv2.rectangle(canvas, (0, H), (W, H+hh), C_BG, -1)
    cv2.line(canvas, (0, H), (W, H), (30,34,44), 2)

    row1 = H + 28   # first text row
    row2 = H + 52
    row3 = H + 76
    row4 = H + 100
    bar1 = H + 108

    # ── COL 1: System ────────────────────────
    _t(canvas, "HAWKWATCH", 12, row1, 0.56, C_GREEN, 1)
    _t(canvas, f"FPS    {fps:5.1f}", 12, row2, 0.42, C_GREY)
    _t(canvas, f"FRAME  {count:07d}", 12, row3, 0.42, C_GREY)
    _t(canvas, f"EVENTS {alerter.incident_count:04d}", 12, row4, 0.46, C_GOLD)
    cv2.line(canvas, (192, H+6), (192, H+hh-6), C_PANEL, 1)

    # ── COL 2: YOLO objects ──────────────────
    persons = [d for d in detections if d["class_id"] == 0]
    threats = [d for d in detections if d["is_threat"]]
    o_col   = C_RED if threats else (C_ORANGE if persons else C_GREEN)
    _t(canvas, "OBJECTS  (YOLO)", 205, row1, 0.42, C_GREY)
    _t(canvas, f"Persons  {len(persons)}", 205, row2, 0.50, o_col)
    if threats:
        _t(canvas, f"WEAPON : {threats[0]['label'].upper()}", 205, row3, 0.50, C_RED, 2)
    else:
        _t(canvas, "Weapon   none", 205, row3, 0.48, C_GREY)
    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        _t(canvas, f"{top['label']} conf", 205, row4, 0.38, C_GREY)
        _bar(canvas, 205, bar1, 155, 7, top["confidence"],
             C_RED if top["is_threat"] else C_GREEN)
    cv2.line(canvas, (400, H+6), (400, H+hh-6), C_PANEL, 1)

    # ── COL 3: V-JEPA 2 motion ──────────────
    v_lbl  = state["label"]
    v_conf = state["confidence"]
    v_susp = state["vjepa_susp"]
    v_col  = C_RED if v_susp else C_GREY
    _t(canvas, "MOTION  (V-JEPA 2)", 413, row1, 0.42, C_GREY)
    _t(canvas, v_lbl[:36], 413, row2, 0.44, v_col, 1 + int(v_susp))
    st = "SUSPICIOUS" if v_susp else ("inferring..." if state["busy"] else "normal")
    _t(canvas, st, 413, row3, 0.44, C_RED if v_susp else C_GREEN)
    _bar(canvas, 413, bar1, 170, 7, v_conf, C_RED if v_susp else C_GREEN)
    cv2.line(canvas, (630, H+6), (630, H+hh-6), C_PANEL, 1)

    # ── COL 4: Moondream scene ───────────────
    desc   = state.get("description", "loading...")
    m_susp = state.get("vlm_suspicious", False)
    m_col  = C_RED if m_susp else C_GREY
    _t(canvas, "SCENE  (Moondream2)", 643, row1, 0.42, C_GREY)
    # Wrap long description across two lines
    words = desc.split()
    line1 = ""
    line2 = ""
    for w in words:
        if len(line1) < 38:
            line1 += w + " "
        else:
            line2 += w + " "
    _t(canvas, line1.strip()[:42], 643, row2, 0.42, m_col)
    _t(canvas, line2.strip()[:42], 643, row3, 0.42, m_col)
    tier = state.get("tier", "CLEAR")
    tier_col = C_RED if tier == "HIGH" else (C_ORANGE if tier == "MEDIUM" else C_GREEN)
    _t(canvas, f"SIGNAL: {tier}", 643, row4, 0.44, tier_col, 1 + int(tier == "HIGH"))
    cv2.line(canvas, (W-172, H+6), (W-172, H+hh-6), C_PANEL, 1)

    # ── COL 5: Alert status ──────────────────
    _t(canvas, "STATUS", W-158, row1, 0.42, C_GREY)
    if alerter.active:
        pulse = int(abs(np.sin(time.time() * 4)) * 50) + 30
        cv2.rectangle(canvas, (W-160, H+34), (W-8, H+62), (0, 20, pulse+140), -1)
        _t(canvas, "  ALERT  ", W-153, H+54, 0.54, C_WHITE, 2)
        _t(canvas, alerter.active_reason[:24], W-160, row3, 0.34, C_ORANGE)
        _t(canvas, "frame saved", W-160, row4, 0.34, C_GREY)
        # Flash border
        p  = abs(np.sin(time.time() * 5)) * 0.10
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 200), -1)
        cv2.addWeighted(ov, p, canvas, 1-p, 0, canvas)
        bw = 4
        cv2.rectangle(canvas, (bw, bw), (W-bw, H-bw), (0, 0, 220), bw)
    else:
        _t(canvas, "  CLEAR  ", W-153, H+54, 0.54, C_GREEN)
        _t(canvas, "monitoring", W-155, row3, 0.38, C_GREY)

    return canvas
