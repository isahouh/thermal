import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from collections import deque
from flirpy.camera.boson import Boson
from scipy.optimize import linear_sum_assignment

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

# FFC Config
FFC_INTERVAL = 180  # 3 minutes in seconds
FFC_TEMP_DRIFT = 1.5  # degrees Celsius
TEMP_CHECK_INTERVAL = 5  # seconds

# Tracking Config
IOU_WEIGHT = 0.6
DISTANCE_WEIGHT = 0.4
MAX_DISTANCE = 100  # pixels
MATCH_THRESHOLD = 0.5

# ───── Camera Class ──────────────────────────────────────────────────────────
class FLIRCamera:
    def __init__(self):
        self.cam = Boson()
        self.last_ffc_time = time.time()
        self.last_ffc_temp = None
        self.last_temp_check = time.time()
        # Perform initial FFC
        self.do_ffc()
    
    def do_ffc(self):
        """Perform flat field correction"""
        print("Performing FFC...")
        self.cam.do_ffc()
        self.last_ffc_time = time.time()
        self.last_ffc_temp = self.cam.get_fpa_temperature()
        print(f"FFC complete. FPA temp: {self.last_ffc_temp:.1f}°C")
    
    def check_ffc_needed(self):
        """Check if FFC is needed based on time or temperature drift"""
        current_time = time.time()
        
        # Check every 5 seconds
        if current_time - self.last_temp_check < TEMP_CHECK_INTERVAL:
            return
        
        self.last_temp_check = current_time
        
        # Check time since last FFC
        if current_time - self.last_ffc_time >= FFC_INTERVAL:
            self.do_ffc()
            return
        
        # Check temperature drift
        current_temp = self.cam.get_fpa_temperature()
        if self.last_ffc_temp is not None:
            temp_drift = abs(current_temp - self.last_ffc_temp)
            if temp_drift >= FFC_TEMP_DRIFT:
                print(f"Temperature drift: {temp_drift:.1f}°C")
                self.do_ffc()
    
    def read(self):
        """Read frame from Boson camera"""
        frame = self.cam.grab()
        if frame is not None:
            # Convert to 8-bit if needed
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Convert to BGR for OpenCV
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return True, frame
        return False, None
    
    def release(self):
        self.cam.close()

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

    def _get_centroid(self, bbox):
        """Get centroid of bbox"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

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

    def compute_distance(self, bbox1, bbox2):
        """Compute Euclidean distance between centroids"""
        c1 = self._get_centroid(bbox1)
        c2 = self._get_centroid(bbox2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def compute_cost_matrix(self, detections, track_bboxes, track_classes):
        """Compute hybrid cost matrix using IOU and centroid distance"""
        n_det = len(detections)
        n_tracks = len(track_bboxes)
        cost_matrix = np.ones((n_det, n_tracks)) * 1e6  # Large cost for impossible matches
        
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            det_class = det['class_id']
            
            for j, (track_bbox, track_class) in enumerate(zip(track_bboxes, track_classes)):
                # Skip if different class
                if det_class != track_class:
                    continue
                
                # Compute IOU score (1 - IOU so lower is better)
                iou = self.compute_iou(det_bbox, track_bbox)
                iou_cost = 1.0 - iou
                
                # Compute distance score (normalized)
                dist = self.compute_distance(det_bbox, track_bbox)
                dist_cost = min(dist / MAX_DISTANCE, 1.0)
                
                # Hybrid cost
                cost = IOU_WEIGHT * iou_cost + DISTANCE_WEIGHT * dist_cost
                cost_matrix[i, j] = cost
        
        return cost_matrix

    def update(self, detections):
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks.values():
            track['frames_missing'] += 1
            track['kalman'].predict()
        
        if not detections or not self.tracks:
            # No matching needed
            if not detections:
                # Remove dead tracks
                self.tracks = {
                    tid: t for tid, t in self.tracks.items()
                    if t['frames_missing'] <= MAX_FRAMES_MISSING
                }
            else:
                # All detections are new
                for det in detections:
                    self._create_new_track(det)
            return
        
        # Prepare track data
        track_ids = list(self.tracks.keys())
        track_bboxes = [self._get_bbox_from_state(self.tracks[tid]['kalman'].statePost) 
                        for tid in track_ids]
        track_classes = [self.tracks[tid]['class_id'] for tid in track_ids]
        
        # Compute cost matrix
        cost_matrix = self.compute_cost_matrix(detections, track_bboxes, track_classes)
        
        # Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Process matches
        matched_dets = set()
        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < MATCH_THRESHOLD:
                # Valid match
                det = detections[det_idx]
                tid = track_ids[track_idx]
                self._update_track(tid, det)
                matched_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_new_track(det)
        
        # Remove dead tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t['frames_missing'] <= MAX_FRAMES_MISSING
        }

    def _update_track(self, track_id, detection):
        """Update existing track with new detection"""
        t = self.tracks[track_id]
        bbox = detection['bbox']
        
        # Kalman correction
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)
        t['kalman'].correct(measurement)
        
        # Update track info
        t['conf_history'].append(detection['conf'])
        t['frames_detected'] += 1
        t['frames_missing'] = 0
        t['last_seen'] = self.frame_count
        
        if not t['confirmed'] and t['frames_detected'] >= MIN_FRAMES_FOR_DETECTION:
            t['confirmed'] = True

    def _create_new_track(self, detection):
        """Create new track from detection"""
        kf = self._init_kalman(detection['bbox'])
        self.tracks[self.next_id] = {
            'kalman': kf,
            'class_id': detection['class_id'],
            'conf_history': deque([detection['conf']], maxlen=CONFIDENCE_HISTORY_SIZE),
            'frames_detected': 1,
            'frames_missing': 0,
            'first_seen': self.frame_count,
            'last_seen': self.frame_count,
            'confirmed': False,
            'id': self.next_id
        }
        self.next_id += 1

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
    
    # Warm up model with dummy frame
    print("Warming up model...")
    dummy_frame = np.zeros((LIVE_H, LIVE_W, 3), dtype=np.uint8)
    with torch.no_grad():
        _ = model(dummy_frame)
    
    cam = FLIRCamera()
    fps_counter = FPSCounter()
    tracker = TemporalDetectionTracker()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # Check if FFC is needed
            cam.check_ffc_needed()

            fps_counter.update()
            t0 = time.time()
            
            # Run inference without gradient computation
            with torch.no_grad():
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
