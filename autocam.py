import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# ───── CONFIG ────────────────────────────────────────────────────────────────
ENGINE_PATH = '/home/ibrahim/models/best.engine'
TARGET_FPS = 60
LIVE_W, LIVE_H = 640, 512
CONF_THR = 0.25
IOU_THR = 0.45
MAX_DET = 20
CLASSES = None
MIN_FRAMES_FOR_DETECTION = 3
MAX_FRAMES_MISSING = 2
CONFIDENCE_HISTORY_SIZE = 5
IOU_MATCH_THRESHOLD = 0.3

# ───── Camera Class ──────────────────────────────────────────────────────────
class FLIRCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print('Camera open failed')
            exit(1)
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()

# ───── FPS Counter ───────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
    
    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0.5:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps
    
    def draw(self, frame, inference_ms=0, raw_detections=0, filtered_detections=0):
        text1 = f"FPS: {self.fps:.1f}"
        text2 = f"Inference: {inference_ms:.1f}ms"
        text3 = f"Raw/Filtered: {raw_detections}/{filtered_detections}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (10, 10), (250, 85), (0, 0, 0), -1)
        cv2.putText(frame, text1, (15, 35), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text2, (15, 55), font, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, text3, (15, 75), font, 0.5, (255, 255, 0), 1)

# ───── Temporal Detection Tracker ────────────────────────────────────────────
class TemporalDetectionTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0

    def _init_kalman(self, bbox):
        """Initialize a Kalman filter for a bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        kf = cv2.KalmanFilter(8, 4)
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i+4] = 1  # velocity terms

        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.statePost = np.array([
            (x1 + x2) / 2, (y1 + y2) / 2,
            x2 - x1, y2 - y1,
            0, 0, 0, 0
        ], dtype=np.float32).reshape(-1, 1)
        return kf

    def _get_bbox_from_state(self, state):
        cx, cy, w, h = state[:4].flatten()
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        self.frame_count += 1
        for track in self.tracks.values():
            track['frames_missing'] += 1
            track['kalman'].predict()

        matched_ids = set()
        for det in detections:
            bbox, class_id, conf = det['bbox'], det['class_id'], det['conf']
            best_id, best_iou = None, IOU_MATCH_THRESHOLD
            for tid, t in self.tracks.items():
                if t['class_id'] != class_id:
                    continue
                iou = self.compute_iou(bbox, self._get_bbox_from_state(t['kalman'].statePost))
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                t = self.tracks[best_id]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)
                t['kalman'].correct(measurement)
                t['conf_history'].append(conf)
                t['frames_detected'] += 1
                t['frames_missing'] = 0
                t['last_seen'] = self.frame_count
                if not t['confirmed'] and t['frames_detected'] >= MIN_FRAMES_FOR_DETECTION:
                    t['confirmed'] = True
                matched_ids.add(best_id)
            else:
                kf = self._init_kalman(bbox)
                self.tracks[self.next_id] = {
                    'kalman': kf,
                    'class_id': class_id,
                    'conf_history': deque([conf], maxlen=CONFIDENCE_HISTORY_SIZE),
                    'frames_detected': 1,
                    'frames_missing': 0,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'confirmed': False,
                    'id': self.next_id
                }
                self.next_id += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t['frames_missing'] <= MAX_FRAMES_MISSING
        }

    def get_confirmed_detections(self):
        return [
            {
                'bbox': self._get_bbox_from_state(t['kalman'].statePost),
                'class_id': t['class_id'],
                'conf': np.mean(t['conf_history']),
                'track_id': t['id'],
                'frames_detected': t['frames_detected']
            }
            for t in self.tracks.values()
            if t['confirmed'] and t['frames_missing'] == 0
        ]


# ───── Drawing Function ──────────────────────────────────────────────────────
def draw_detections(frame, detections, model):
    for d in detections:
        x1, y1, x2, y2 = map(int, d['bbox'])
        label = f"{model.names[d['class_id']]} {d['conf']:.2f} [{d['frames_detected']}f]"
        color = (0, 255, 0) if d['frames_detected'] > 10 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - sz[1] - 4), (x1 + sz[0] + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ───── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Initializing...")
    model = YOLO(ENGINE_PATH)
    cam = FLIRCamera(0)
    fps_counter = FPSCounter()
    tracker = TemporalDetectionTracker()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            fps_counter.update()
            t0 = time.time()
            results = model(frame)
            inference_ms = (time.time() - t0) * 1000

            boxes = results[0].boxes
            raw_detections = [{
                'bbox': boxes.xyxy[i].cpu().numpy(),
                'class_id': int(boxes.cls[i]),
                'conf': float(boxes.conf[i])
            } for i in range(len(boxes))] if boxes is not None else []

            tracker.update(raw_detections)
            confirmed = tracker.get_confirmed_detections()
            draw_detections(frame, confirmed, model)
            fps_counter.draw(frame, inference_ms, len(raw_detections), len(confirmed))

            cv2.imshow('Thermal Detection', frame)
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
