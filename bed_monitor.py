import csv
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
 
import cv2
from ultralytics import YOLO
 
try:
    import winsound  # Windows only (built in, no install needed)
except ImportError:
    winsound = None
 
BED_FILE = "bed.json"
ALERT_LOG = "alerts_log.csv"
RESULTS = "results.csv"
DEFAULT_BED = (150, 100, 500, 400)
MARGIN = 40        # pixels from bed edge that count as WARNING
WINDOW = 15        # frames used for temporal smoothing
VOTES = 8          # frames that must agree to change the level
COOLDOWN = 5       # seconds between repeated alarms
LYING_RATIO = 1.1  # box width/height above this = lying down
SLEEP_SECONDS = 10 # lying + almost no movement this long = SLEEPING
MOVE_PX = 15       # center shift (pixels) that counts as movement
FALL_DROP = 0.4    # box height shrinking by this fraction ...
FALL_WINDOW = 1.5  # ... within this many seconds = possible fall
FALL_HOLD = 10     # seconds before a new fall alarm can repeat
EAR_THRESH = 0.21  # eye aspect ratio below this = eyes closed
EYE_SECONDS = 2    # eyes closed this long (not a blink) = SLEEPING
CONF = 0.5         # person detection confidence (raise to ignore false people)
MIN_AREA = 0.04    # ignore detections smaller than this fraction of the frame
COLORS = {"NONE": (150, 150, 150), "LOW": (0, 200, 0),
          "MEDIUM": (0, 200, 255), "HIGH": (0, 0, 255)}
RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
MESSAGES = {"NONE": "Patient not detected", "LOW": "Normal",
            "MEDIUM": "WARNING: moving toward bed edge",
            "HIGH": "ALERT: patient may be leaving the bed!"}
 
 
RATIO_FILE = "lying_ratio.json"
 
 
def load_ratio():
    if os.path.exists(RATIO_FILE):
        try:
            with open(RATIO_FILE) as f:
                return float(json.load(f))
        except (OSError, ValueError):
            pass
    return LYING_RATIO
 
 
def save_ratio(r):
    with open(RATIO_FILE, "w") as f:
        json.dump(r, f)
 
 
def load_bed():
    if os.path.exists(BED_FILE):
        with open(BED_FILE) as f:
            return tuple(json.load(f))
    return DEFAULT_BED
 
 
def calibrate(frame):
    """Click top-left then bottom-right corner of the bed."""
    pts = []
 
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
 
    win = "Calibrate bed"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 2:
        img = frame.copy()
        cv2.putText(img, "Click TOP-LEFT, then BOTTOM-RIGHT of the bed (Esc cancels)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(img, p, 6, (255, 0, 0), -1)
        cv2.imshow(win, img)
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    (x1, y1), (x2, y2) = pts
    bed = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    with open(BED_FILE, "w") as f:
        json.dump(bed, f)
    return bed
 
 
def risk(cx, cy, bed):
    x1, y1, x2, y2 = bed
    if not (x1 < cx < x2 and y1 < cy < y2):
        return "HIGH"
    d = min(cx - x1, x2 - cx, cy - y1, y2 - cy)
    return "MEDIUM" if d < MARGIN else "LOW"
 
 
def smooth(hist, previous):
    if hist.count("HIGH") >= VOTES:
        return "HIGH"
    if hist.count("HIGH") + hist.count("MEDIUM") >= VOTES:
        return "MEDIUM"
    if hist.count("LOW") >= VOTES:
        return "LOW"
    if hist.count("NONE") >= VOTES:
        return "NONE"
    return previous
 
 
STATUS_FILE = "status.json"
TELEGRAM_FILE = "telegram.json"   # {"token": "...", "chat_id": "..."}
 
 
def write_status(level, detected, source, activity="-", eyes="-"):
    """Dashboard reads this file."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump({"level": level, "detected": detected,
                       "message": MESSAGES[level], "activity": activity, "eyes": eyes, "source": str(source),
                       "time": datetime.now().strftime("%H:%M:%S")}, f)
    except OSError:
        pass
 
 
def send_telegram(text):
    """Optional phone alert. Does nothing if telegram.json is missing."""
    if not os.path.exists(TELEGRAM_FILE):
        return
 
    def run():
        try:
            import requests
            with open(TELEGRAM_FILE) as f:
                cfg = json.load(f)
            requests.post(
                f"https://api.telegram.org/bot{cfg['token']}/sendMessage",
                data={"chat_id": cfg["chat_id"], "text": text}, timeout=10)
        except Exception as e:
            print("Telegram error:", e)
    threading.Thread(target=run, daemon=True).start()
 
 
class EyeDetector:
    """Eyes open/closed using MediaPipe Face Mesh (eye aspect ratio)."""
    L = [362, 385, 387, 263, 373, 380]
    R = [33, 160, 158, 133, 153, 144]
 
    def __init__(self):
        self.mesh = None
        try:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as e:
            print("Eye detection OFF (pip install mediapipe==0.10.14):", e)
 
    @staticmethod
    def _ear(pts, idx):
        p = [pts[i] for i in idx]
        d = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        return (d(p[1], p[5]) + d(p[2], p[4])) / (2 * d(p[0], p[3]) + 1e-6)
 
    def state(self, frame, box):
        """Returns 'CLOSED', 'OPEN' or None (no face visible)."""
        if self.mesh is None:
            return None
        x1, y1, x2, y2 = [max(int(v), 0) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        res = self.mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = crop.shape[:2]
        pts = {i: (lm[i].x * w, lm[i].y * h) for i in self.L + self.R}
        ear = (self._ear(pts, self.L) + self._ear(pts, self.R)) / 2
        return "CLOSED" if ear < EAR_THRESH else "OPEN"
 
 
def beep():
    if winsound:
        def run():
            for _ in range(3):
                winsound.Beep(1200, 300)
                time.sleep(0.1)
        threading.Thread(target=run, daemon=True).start()
    else:
        print("\a", end="", flush=True)
 
 
def beep_fall():
    """Louder, longer alarm for falls (different from the bed-exit beep)."""
    if winsound:
        def run():
            for _ in range(6):
                winsound.Beep(2000, 250)
                winsound.Beep(1000, 250)
        threading.Thread(target=run, daemon=True).start()
    else:
        print("\a\a\a", end="", flush=True)
 
 
def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
 
 
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    is_cam = isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if is_cam else cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open source:", src)
        return
 
    model = YOLO("yolov8n.pt")
    bed = load_bed()
    hist = deque(maxlen=WINDOW)
    final, max_level = "NONE", "NONE"
    last_alarm, first_high, frames = 0.0, None, 0
    start = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    post_hist = deque(maxlen=WINDOW)
    anchor, last_move, activity = None, 0.0, "NO PATIENT"
    h_hist, fall_until = deque(), -1.0
    eyes = EyeDetector()
    lying_ratio, cur_ratio = load_ratio(), 0.0
    eye_state, eye_closed_since = None, None
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
 
        r = model(frame, classes=[0], conf=CONF, verbose=False)[0]
        level, box = "NONE", None
        fh, fw = frame.shape[:2]
        cands = [b for b in r.boxes.xyxy.tolist()
                 if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_AREA * fw * fh]
        if cands:
            # only ONE patient: the largest confident person
            box = max(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            level = risk(cx, cy, bed)
        hist.append(level)
        t = frames / fps if not is_cam else time.time() - start
        if box:
            w, h = box[2] - box[0], box[3] - box[1]
            cur_ratio = w / max(h, 1)
            post_hist.append("LYING" if cur_ratio > lying_ratio else "UPRIGHT")
            if anchor is None or abs(cx - anchor[0]) + abs(cy - anchor[1]) > MOVE_PX:
                anchor, last_move = (cx, cy), t
            posture = max(set(post_hist), key=post_hist.count)
            if posture == "LYING":
                activity = "SLEEPING" if t - last_move >= SLEEP_SECONDS else "LYING (awake)"
            else:
                activity = "UPRIGHT"
        else:
            activity, anchor = "NO PATIENT", None
            post_hist.clear()
 
        # ---- eyes closed = sleeping ----
        eye_label = "-"
        if box:
            if frames % 2 == 0:          # every 2nd frame to save CPU
                eye_state = eyes.state(frame, box)
            if eye_state == "CLOSED":
                if eye_closed_since is None:
                    eye_closed_since = t
            else:
                eye_closed_since = None
            if eye_closed_since is not None and t - eye_closed_since >= EYE_SECONDS:
                activity = "SLEEPING (eyes closed)"
                eye_label = "SLEEPING"
            elif eye_state is not None:
                eye_label = "AWAKE"
        else:
            eye_state, eye_closed_since = None, None
        prev = final
        final = smooth(hist, final)
        if RANK[final] > RANK[max_level]:
            max_level = final
 
        # ---- fall detection ----
        fell = False
        if box:
            h_hist.append((t, h))
            while h_hist and t - h_hist[0][0] > FALL_WINDOW:
                h_hist.popleft()
            tallest = max(x[1] for x in h_hist)
            dropped = len(h_hist) > 3 and h < (1 - FALL_DROP) * tallest
            outside = not (bed[0] < cx < bed[2] and bed[1] < cy < bed[3])
            lying_now = cur_ratio > lying_ratio
            # sudden drop to lying outside the bed, or lying outside the bed
            fell = (dropped and lying_now and outside) or (posture == "LYING" and final == "HIGH")
        else:
            h_hist.clear()
        if fell and t >= fall_until:
            beep_fall()
            send_telegram("FALL DETECTED! Patient may have fallen. "
                          + datetime.now().strftime("%H:%M:%S"))
            append_csv(ALERT_LOG, ["time", "source", "level"],
                       [datetime.now().isoformat(timespec="seconds"), src, "FALL"])
            fall_until = t + FALL_HOLD
        fall_active = t < fall_until
        if fall_active:
            activity = "FALL DETECTED"
            max_level = "HIGH"
 
        write_status(final, box is not None, src, activity, eye_label)
 
        # alert on entering HIGH (and repeat after cooldown while it stays HIGH)
        now = time.time()
        if final == "HIGH":
            if first_high is None:
                first_high = frames / fps if not is_cam else now - start
            if prev != "HIGH" or now - last_alarm > COOLDOWN:
                beep()
                send_telegram("ALERT: Patient may be leaving the bed! "
                              + datetime.now().strftime("%H:%M:%S"))
                last_alarm = now
                append_csv(ALERT_LOG, ["time", "source", "level"],
                           [datetime.now().isoformat(timespec="seconds"), src, final])
 
        # draw
        color = COLORS[final]
        cv2.rectangle(frame, bed[:2], bed[2:], (255, 0, 0), 2)
        if box:
            cv2.rectangle(frame, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (30, 30, 30), -1)
        cv2.putText(frame, f"Activity: {activity}  Eyes: {eye_state or '-'}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Patient: {'Detected' if box else 'Not detected'}   "
                           f"Risk: {final}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, MESSAGES[final], (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if final == "HIGH":
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 6)
        if fall_active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 10)
            cv2.putText(frame, "FALL DETECTED!", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"ratio {cur_ratio:.2f} (lying if > {lying_ratio:.2f})   c=bed  l=set lying  Esc=quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Bed Monitor  (c = calibrate, Esc = quit)", frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("l") and cur_ratio > 0:
            # lie in your normal sleeping position, then press l
            lying_ratio = round(cur_ratio * 0.9, 2)
            save_ratio(lying_ratio)
            post_hist.clear()
        if key == ord("c"):
            new_bed = calibrate(frame)
            if new_bed:
                bed = new_bed
 
    cap.release()
    cv2.destroyAllWindows()
 
    # save one summary row per run (used by evaluate.py)
    name = os.path.basename(str(src)) if not is_cam else "webcam"
    append_csv(RESULTS, ["source", "max_level", "first_high_sec", "frames"],
               [name, max_level, "" if first_high is None else round(first_high, 2), frames])
    print(f"Done: {name}  max_level={max_level}  first_high={first_high}")
 
 
if __name__ == "__main__":
    main()
 
