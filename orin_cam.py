import cv2
import socket
import struct
import threading
import time
import json
import queue
import numpy as np
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from flirpy.camera.boson import Boson
import subprocess
import os
import glob
from datetime import datetime, timedelta
import shutil

# WiFi Config - Primary (Car)
PRIMARY_WIFI_SSID = ""
PRIMARY_WIFI_PASSWORD = ""

# WiFi Config - Secondary (Home)
SECONDARY_WIFI_SSID = ""
SECONDARY_WIFI_PASSWORD = ""

# WiFi Send Config
CHUNK_PAYLOAD = 1000 # bytes of jpeg per UDP packet
PKT_HDR_FMT = '>IHHI'
FEC_K = 4            # data chunks per group
FEC_ENABLE = True
PACKET_GAP_SEC_DEFAULT = 0.00050

# Config
ENGINE_PATH = '/home/ibrahim/datasets/best.engine' #old model @ /model/backup/best.engine
CONF_THR = 0.35
IOU_THR = 0.35
MAX_DET = 15
JPEG_QUALITY = 50
INFERENCE_THREADS = 1

# Tracking config with confidence accumulation
CONFIRM_THRESHOLD = 2.5      # Cumulative confidence needed to confirm track
MAINTAIN_THRESHOLD = 0.35     # Minimum confidence to maintain confirmed track
CONFIDENCE_DECAY = 0.9       # Decay factor per frame (0.85 = 15% decay)
MAX_CONFIDENCE = 5.5         # Cap accumulated confidence
MIN_MATCH_SCORE = 0.2        # Minimum score to consider a match
IOU_WEIGHT = 0.6             # Weight for IOU component
CENTROID_WEIGHT = 0.4        # Weight for centroid component
MIN_TRACK_DISTANCE = 50     # Minimum pixels between track centers

# Alert trigger classes 
ALERT_CLASSES = ["deer", "hog", "small_animal"]
CORNER_SIZE = 20  # Size of red corner indicators
CORNER_COLOR = (0, 0, 255)  # BGR format - pure red
ALERT_DURATION_FRAMES = 5  # How many frames to keep corners red

# Animal size constraints (in pixels)
DEFAULT_SIZE_CONSTRAINTS = {"min_height": 20, "max_height": 400}
ANIMAL_SIZE_CONSTRAINTS = {
    "deer": {"min_height": 12, "max_height": 140},
    "hog": {"min_height": 10, "max_height": 120},
    "small_animal": {"min_height": 8, "max_height": 90}
#    "coyote": {"min_height": 8, "max_height": 80},
#    "hog": {"min_height": 10, "max_height": 100},
#    "raccoon": {"min_height": 8, "max_height": 60},
#    "rabbit": {"min_height": 8, "max_height": 50}
}

# FFC Config
FFC_INTERVAL = 180  # 3 minutes in seconds
FFC_TEMP_DRIFT = 1.5  # degrees Celsius
TEMP_CHECK_INTERVAL = 5  # seconds
FFC_ANIMAL_DELAY = 15  # seconds to delay FFC after animal detection

# Track class with confidence accumulation (no velocity in this version)
class SimpleTrack:
    def __init__(self, bbox, cls, conf, track_id, model_names = None):
        self.id = track_id
        self.bbox = bbox
        self.cls = cls
        self.conf = conf
        self.age = 1
        self.confirmed = False
        self.model_names = model_names
        
        # Confidence accumulation
        self.confidence_sum = conf
        
        # For centroid tracking
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.centroid = np.array([cx, cy])
    
    def predict(self):
        """Decay confidence when not detected"""
        # Decay confidence when not detected
        self.confidence_sum *= CONFIDENCE_DECAY
        
        # Increment age
        self.age += 1
        
    def update(self, bbox, conf, cls):
        """Update track with new detection"""
        # Update centroid
        new_cx = (bbox[0] + bbox[2]) / 2
        new_cy = (bbox[1] + bbox[3]) / 2
        self.centroid = np.array([new_cx, new_cy])
        
        # Update state
        self.bbox = bbox
        self.conf = conf
        self.cls = cls
        
        # Accumulate confidence (with cap)
        self.confidence_sum = min(self.confidence_sum + conf, MAX_CONFIDENCE)
        
        # Check if track should be confirmed
        if not self.confirmed and self.confidence_sum >= CONFIRM_THRESHOLD:
            self.confirmed = True
    
    def is_active(self):
        """Check if track should be kept active"""
        if self.confirmed:
            # Keep confirmed tracks until confidence drops too low
            return self.confidence_sum > MAINTAIN_THRESHOLD
        else:
            # Unconfirmed tracks need to reach confirmation threshold
            return self.confidence_sum > 0.1  # Very low threshold for potential tracks

# Enhanced IOU + Centroid Tracker
class SimpleTracker:
    def __init__(self, image_width=640, image_height=512, model_names = None):
        self.tracks = []
        self.next_id = 0
        self.image_diagonal = np.sqrt(image_width**2 + image_height**2)
        self.model_names = model_names
        
    def calculate_iou(self, bbox1, bbox2):
        """Calculate intersection over union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_match_score(self, detection_bbox, track):
        """Calculate combined IOU + Centroid score"""
        # IOU component
        iou = self.calculate_iou(detection_bbox, track.bbox)
        
        # Centroid component
        det_cx = (detection_bbox[0] + detection_bbox[2]) / 2
        det_cy = (detection_bbox[1] + detection_bbox[3]) / 2
        det_centroid = np.array([det_cx, det_cy])
        
        distance = np.linalg.norm(det_centroid - track.centroid)
        centroid_score = 1.0 - min(distance / self.image_diagonal, 1.0)
        
        # Combined score
        return (IOU_WEIGHT * iou) + (CENTROID_WEIGHT * centroid_score)
    
    def is_duplicate_track(self, new_bbox, existing_tracks, min_distance=MIN_TRACK_DISTANCE):
        """Check if a detection would create a duplicate track"""
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        
        for track in existing_tracks:
            if track.confirmed:  # Only check against confirmed tracks
                track_cx = (track.bbox[0] + track.bbox[2]) / 2
                track_cy = (track.bbox[1] + track.bbox[3]) / 2
                
                distance = np.sqrt((new_cx - track_cx)**2 + (new_cy - track_cy)**2)
                
                # Also check IOU
                iou = self.calculate_iou(new_bbox, track.bbox)
                
                if distance < min_distance or iou > 0.3:
                    return True
        
        return False
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Predict all tracks (includes confidence decay)
        for track in self.tracks:
            track.predict()
        
        # Remove inactive tracks
        self.tracks = [t for t in self.tracks if t.is_active()]
        
        if len(detections) == 0:
            # Return confirmed tracks even with no detections
            return [{
                'id': t.id,
                'bbox': t.bbox,
                'cls': t.cls,
                'conf': t.conf,
                'confidence_sum': t.confidence_sum,
                'age': t.age
            } for t in self.tracks if t.confirmed]
        
        # Sort detections by confidence (process high confidence first)
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        # Match detections to tracks
        matched_tracks = set()
        unmatched_detections = []
        
        for det in detections:
            best_track_idx = -1
            best_score = MIN_MATCH_SCORE
            
            # Find best matching track
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                    
                score = self.calculate_match_score(det['bbox'], track)
                if score > best_score:
                    best_score = score
                    best_track_idx = track_idx
            
            # Update matched track
            if best_track_idx >= 0:
                matched_tracks.add(best_track_idx)
                self.tracks[best_track_idx].update(det['bbox'], det['conf'], det['cls'])
            else:
                unmatched_detections.append(det)
        
        # Create new tracks only for non-duplicate detections
        for det in unmatched_detections:
            if not self.is_duplicate_track(det['bbox'], self.tracks):
                new_track = SimpleTrack(det['bbox'], det['cls'], det['conf'], self.next_id, self.model_names)
                self.next_id += 1
                self.tracks.append(new_track)
        
        # Return only confirmed tracks
        return [{
            'id': t.id,
            'bbox': t.bbox,
            'cls': t.cls,
            'conf': t.conf,
            'confidence_sum': t.confidence_sum,
            'age': t.age
        } for t in self.tracks if t.confirmed]

# Frame Buffer Pool (unchanged)
class FrameBufferPool:
    def __init__(self, width, height, channels, pool_size=20):
        self.shape = (height, width, channels)
        self.buffers = [np.empty(self.shape, dtype=np.uint8) for _ in range(pool_size)]
        self.available = queue.Queue()
        
        for buf in self.buffers:
            self.available.put(buf)
    
    def get(self, timeout=0.001):
        try:
            return self.available.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self, buffer):
        if buffer is not None:
            self.available.put(buffer)

# Camera class with FFC support
class FLIRCamera:
    def __init__(self):
        self.cam = Boson()
        self.last_ffc_time = time.time()
        self.last_ffc_temp = None
        self.last_temp_check = time.time()
        self.last_animal_detection_time = 0  # Track when animal last seen
        self.ffc_just_completed = False  # Flag for FFC completion
        self._lock = threading.Lock()  # Thread safety for FFC operations
        # Perform initial FFC
        self.do_ffc()
    
    def do_ffc(self):
        """Perform flat field correction"""
        with self._lock:
            print("Performing FFC...")
            self.cam.do_ffc()
            self.last_ffc_time = time.time()
            self.last_ffc_temp = self.cam.get_fpa_temperature()
            self.ffc_just_completed = True
            print("FFC completed")
    
    def update_animal_detection(self, detected_classes):
        """Update last animal detection time if any animals detected"""
        # Convert class names to check against ALERT_CLASSES
        if any(cls in ALERT_CLASSES for cls in detected_classes):
            with self._lock:
                self.last_animal_detection_time = time.time()
    
    def check_ffc_needed(self):
        """Check if FFC is needed based on time or temperature drift"""
        current_time = time.time()
        
        # Check every 5 seconds
        if current_time - self.last_temp_check < TEMP_CHECK_INTERVAL:
            return
        
        self.last_temp_check = current_time
        
        # Check if animal was recently detected
        with self._lock:
            time_since_animal = current_time - self.last_animal_detection_time
        
        if time_since_animal < FFC_ANIMAL_DELAY:
            return  # Skip FFC if animal detected recently
        
        # Check time since last FFC
        if current_time - self.last_ffc_time >= FFC_INTERVAL:
            self.do_ffc()
            return
        
        # Only check temperature if all other conditions pass
        if self.last_ffc_temp is not None:
            with self._lock:
                current_temp = self.cam.get_fpa_temperature()
            temp_drift = abs(current_temp - self.last_ffc_temp)
            if temp_drift >= FFC_TEMP_DRIFT:
                print(f"Temperature drift: {temp_drift:.1f}°C")
                self.do_ffc()
    
    def get_and_clear_ffc_flag(self):
        """Get FFC completion flag and clear it"""
        with self._lock:
            flag = self.ffc_just_completed
            self.ffc_just_completed = False
            return flag
    
    def read(self):
        """Read frame from Boson camera"""
        with self._lock:
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
        with self._lock:
            self.cam.close()

def is_valid_detection_size(bbox, class_name):
    """Check if detection meets size constraints for the class"""
    height = bbox[3] - bbox[1]
    
    # Get constraints for this class
    if class_name in ANIMAL_SIZE_CONSTRAINTS:
        constraints = ANIMAL_SIZE_CONSTRAINTS[class_name]
    else:
        constraints = DEFAULT_SIZE_CONSTRAINTS
    
    return constraints["min_height"] <= height <= constraints["max_height"]

# Inference Pipeline with confidence visualization
class TrackedInferencePipeline:
    def __init__(self, model, buffer_pool, camera, recorder = None, num_threads=2):
        self.model = model
        self.buffer_pool = buffer_pool
        self.camera = camera
        self.recorder = recorder
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.running = True
        
        # Create a tracker for each thread
        self.trackers = [SimpleTracker(model_names=self.model.names) for _ in range(num_threads)]

        # Alert state tracking
        self.alert_frames_remaining = [0] * num_threads  # Track per worker
        
        # Start workers
        for i in range(num_threads):
            self.executor.submit(self._inference_worker, i)
    
    def _inference_worker(self, worker_id):
        tracker = self.trackers[worker_id]
        # Pin inference workers to cores 0-1
        try:
            core = worker_id % 2  # Alternate between core 0 and 1
            os.sched_setaffinity(0, {core})
            print(f"Inference worker {worker_id} pinned to core {core}")
        except:
            pass
        
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    break
                
                frame_buffer, frame_id = data
                
                # Run inference
                with torch.no_grad():
                    results = self.model(
                        frame_buffer,
                        conf=CONF_THR,
                        iou=IOU_THR,
                        max_det=MAX_DET,
                        verbose=False
                    )
                
                # Extract detections
                detections = []
                filtered_count = 0  # Track filtered detections for debugging
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name for size validation
                        class_name = self.model.names[cls]
                        bbox = np.array([x1, y1, x2, y2])
                        
                        # Validate detection size
                        if is_valid_detection_size(bbox, class_name):
                            detections.append({
                                'bbox': bbox,
                                'conf': conf,
                                'cls': cls
                            })
                        else:
                            filtered_count += 1
                
                # Update tracker
                tracks = tracker.update(detections)
                # Check for alert-triggering classes
                alert_triggered = False
                detected_class_names = []
                for track in tracks:
                    class_name = self.model.names[track['cls']]
                    detected_class_names.append(class_name)
                    if class_name in ALERT_CLASSES:
                        alert_triggered = True
                        break

                # Update animal detection in camera for FFC delay
                if detected_class_names:
                    self.camera.update_animal_detection(detected_class_names)

                # Update alert state
                if alert_triggered:
                    self.alert_frames_remaining[worker_id] = ALERT_DURATION_FRAMES

                # Draw tracked objects
                for track in tracks:
                    bbox = track['bbox']
                    x1, y1, x2, y2 = bbox.astype(int)
                    cls = track['cls']
                    conf = track['conf']
                    track_id = track['id']
                    confidence_sum = track['confidence_sum']
                    
                    # Apply heat-map recoloring to the bounding box region
                    recolor_bbox_vectorized(frame_buffer, bbox)
                    
                    # Color for box outline based on accumulated confidence
                    if confidence_sum > 3.0:
                        color = (0, 255, 0)  # Strong green
                    elif confidence_sum > 2.0:
                        color = (0, 200, 100)  # Medium green
                    else:
                        color = (0, 150, 200)  # Bluish (weak but confirmed)
                    
                    # Draw box outline
                    cv2.rectangle(frame_buffer, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with accumulated confidence
                    #label = f"ID{track_id} {self.model.names[cls]} {conf:.2f} ({confidence_sum:.1f})"      # for debugging
                    label = f"{self.model.names[cls]}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame_buffer, (x1, y1 - label_size[1] - 4), 
                                (x1 + label_size[0] + 2, y1), color, -1)
                    cv2.putText(frame_buffer, label, (x1 + 1, y1 - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw red corners if alert is active
                if self.alert_frames_remaining[worker_id] > 0:
                    h, w = frame_buffer.shape[:2]
                    # Top-left corner
                    frame_buffer[0:CORNER_SIZE, 0:CORNER_SIZE] = CORNER_COLOR
                    # Top-right corner
                    frame_buffer[0:CORNER_SIZE, w-CORNER_SIZE:w] = CORNER_COLOR
                    # Bottom-left corner
                    frame_buffer[h-CORNER_SIZE:h, 0:CORNER_SIZE] = CORNER_COLOR
                    # Bottom-right corner
                    frame_buffer[h-CORNER_SIZE:h, w-CORNER_SIZE:w] = CORNER_COLOR
                    
                    self.alert_frames_remaining[worker_id] -= 1

                # Output processed frame
                self.output_queue.put((frame_buffer, frame_id))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nInference error: {e}")
                if 'frame_buffer' in locals():
                    self.buffer_pool.release(frame_buffer)
    
    def process(self, frame_buffer, frame_id):
        try:
            self.input_queue.put_nowait((frame_buffer, frame_id))
            return True
        except queue.Full:
            self.buffer_pool.release(frame_buffer)
            return False
    
    def get_result(self, timeout=0.001):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        for _ in range(self.executor._max_workers):
            self.input_queue.put(None)
        self.executor.shutdown(wait=True)

# Streamer (unchanged)
class VideoStreamerWithBuffer:
    def __init__(self, port=8888, buffer_pool=None):
        self.port = port
        self.buffer_pool = buffer_pool
        self._client_addr = None
        self._lock = threading.Lock()
        self.sequence_num = 0
        self.PACKET_GAP_SEC = float(os.getenv("PACKET_GAP_SEC", PACKET_GAP_SEC_DEFAULT))
        self._gap_ns = int(self.PACKET_GAP_SEC * 1e9)
        self._next_pkt_time_ns = time.perf_counter_ns()

        def _pacer():
            if self._gap_ns <= 0:
                return
            target = self._next_pkt_time_ns
            now = time.perf_counter_ns()
            if now < target:
                rem = target - now
                # Sleep if >200µs remaining, then spin for the last ~100µs
                if rem > 200_000:
                    time.sleep((rem - 100_000) / 1e9)
                while time.perf_counter_ns() < target:
                    pass
            # schedule next packet time
            now2 = time.perf_counter_ns()
            self._next_pkt_time_ns = max(now2, target) + self._gap_ns

        self._pacer = _pacer

        # Encoding queue
        self.encode_queue = queue.Queue(maxsize=3)
        
        # UDP socket
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Queue + buffers
        # Video queue
        try:
            self.udp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 34 << 2)
        except Exception:
            pass
        # Big send buffer for bursty frames
        try:
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except Exception:
            pass
        try:
            if hasattr(socket, 'SO_MAX_PACING_RATE'):
                self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_MAX_PACING_RATE, 10000000)  # bytes/s
        except Exception:
            pass


        # Start threads
        threading.Thread(target=self._udp_server_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._encoder_loop, daemon=True).start()
    
    def _send_fragment(self, header: bytes, payload_view, client):
        # payload_view is a memoryview or bytes
        try:
            self.udp_sock.sendmsg([header, payload_view], [], 0, client)
        except Exception:
            # Fallback (copies payload but is robust everywhere)
            self.udp_sock.sendto(header + (bytes(payload_view)), client)

    def _udp_server_loop(self):
        handshake_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        handshake_sock.bind(('0.0.0.0', self.port))
        print(f"UDP server on port {self.port}")
        
        while True:
            data, addr = handshake_sock.recvfrom(1024)
            if data == b'HELLO':
                with self._lock:
                    if self._client_addr != addr:
                        self._client_addr = addr
                        self.sequence_num = 0
                        print(f"Client connected: {addr[0]}:{addr[1]}")
                handshake_sock.sendto(b'ACK', addr)
    
    def _broadcast_loop(self):
        msg = json.dumps({"service": "webcam_udp", "port": self.port}).encode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        while True:
            sock.sendto(msg, ('<broadcast>', 9999))
            time.sleep(1)
    
    def _encoder_loop(self):
        # (Optional) pin the sender to a spare core so inference/recording don't contend
        try:
            os.sched_setaffinity(0, {5})
            print("Encoder thread pinned to core 5")
        except:
            pass

        while True:
            frame = None
            try:
                frame = self.encode_queue.get(timeout=0.1)
                if frame is None:
                    break

                # Snapshot client under lock, then drop the lock for heavy work
                with self._lock:
                    client = self._client_addr
                if client is None:
                    continue

                # --- JPEG encode (no lock) ---
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ok, jpeg_arr = cv2.imencode('.jpg', frame, encode_param)
                if not ok:
                    continue

                # Make a 1-D byte view for safe slicing and XOR (zero-copy)
                mv = memoryview(jpeg_arr).cast('B')   # flat view over the JPEG bytes

                # Fragmentation
                frame_len = len(mv)
                M = (frame_len + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD
                num_groups = (M + FEC_K - 1) // FEC_K
                P = num_groups if FEC_ENABLE else 0
                T = M + P
                seq = self.sequence_num

                if FEC_ENABLE and P > 0:
                    for g in range(num_groups):
                        parity = np.zeros(CHUNK_PAYLOAD, dtype=np.uint8)
                        # send this group's data
                        for r in range(FEC_K):
                            di = g * FEC_K + r
                            if di >= M:
                                break
                            s = di * CHUNK_PAYLOAD
                            e = min(s + CHUNK_PAYLOAD, frame_len)

                            header = struct.pack(PKT_HDR_FMT, seq, T, di, frame_len)
                            self._send_fragment(header, mv[s:e], client)
                            if self.PACKET_GAP_SEC:
                                self._pacer()

                            chunk_view = np.frombuffer(mv, dtype=np.uint8, count=e - s, offset=s)
                            parity[:e - s] ^= chunk_view

                        # send this group's parity right away
                        parity_idx = M + g
                        header = struct.pack(PKT_HDR_FMT, seq, T, parity_idx, frame_len)
                        self._send_fragment(header, memoryview(parity), client)
                        if self.PACKET_GAP_SEC:
                            self._pacer()
                else:
                    for di in range(M):
                        s = di * CHUNK_PAYLOAD
                        e = min(s + CHUNK_PAYLOAD, frame_len)
                        header = struct.pack(PKT_HDR_FMT, seq, T, di, frame_len)
                        self._send_fragment(header, mv[s:e], client)
                        if self.PACKET_GAP_SEC:
                            self._pacer()

                # bump sequence
                self.sequence_num = (self.sequence_num + 1) % 1000000

            except queue.Empty:
                continue
            except Exception as e:
                # If anything goes wrong, drop client; handshake thread will restore it
                with self._lock:
                    self._client_addr = None
            finally:
                if frame is not None and self.buffer_pool is not None:
                    self.buffer_pool.release(frame)

    def send(self, frame):
        try:
            self.encode_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False
    
    def stop(self):
        self.encode_queue.put(None)


class ContinuousRecorder:
    def __init__(self, output_dir="/mnt/nvme/thermal_recordings", segment_duration=10, max_segments=360, buffer_pool_size=10):
        """
        Continuous recording with rolling segments
        
        Args:
            output_dir: Directory for video segments
            segment_duration: Duration of each segment in seconds
            max_segments: Maximum number of segments (60 segments = 1 hour)
            buffer_pool_size: Size of the recording buffer pool
        """
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.max_segments = max_segments
        self.current_process = None
        self.current_segment_path = None
        self.recording_queue = queue.Queue(maxsize=10)
        self.running = True
        self.segment_start_time = None
        
        # Create recording buffer pool
        self.recording_buffer_pool = FrameBufferPool(640, 512, 3, pool_size=buffer_pool_size)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean up incomplete segments from previous run
        self._cleanup_incomplete_segments()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
    def _cleanup_incomplete_segments(self):
        """Remove any .tmp files from incomplete recordings"""
        tmp_files = glob.glob(os.path.join(self.output_dir, "*.tmp"))
        for tmp_file in tmp_files:
            try:
                os.remove(tmp_file)
                print(f"Removed incomplete segment: {tmp_file}")
            except:
                pass
    
    def _get_existing_segments(self):
        """Get list of existing segments sorted by timestamp"""
        segments = []
        for filepath in glob.glob(os.path.join(self.output_dir, "thermal_*.mp4")):
            try:
                filename = os.path.basename(filepath)
                # Extract timestamp from filename: thermal_YYYYMMDD_HHMMSS.mp4
                timestamp_str = filename.replace("thermal_", "").replace(".mp4", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                segments.append((timestamp, filepath))
            except:
                continue
        return sorted(segments, key=lambda x: x[0])
    
    def _manage_segments(self):
        """Maintain maximum number of segments (1 hour rolling)"""
        segments = self._get_existing_segments()
        
        # Remove oldest segments if we exceed max
        while len(segments) > self.max_segments - 1:  # -1 for the segment being recorded
            oldest_time, oldest_path = segments.pop(0)
            try:
                os.remove(oldest_path)
                print(f"Removed old segment: {os.path.basename(oldest_path)}")
            except:
                pass
    
    def _start_new_segment(self):
        """Start recording a new segment"""
        # Stop current recording if any
        if self.current_process:
            self.current_process.stdin.close()
            self.current_process.wait()
            
            # Rename temp file to final name
            if self.current_segment_path and os.path.exists(self.current_segment_path + ".tmp"):
                os.rename(self.current_segment_path + ".tmp", self.current_segment_path)
        
        # Manage segments
        self._manage_segments()
        
        # Generate new filename
        self.segment_start_time = datetime.now()
        timestamp = self.segment_start_time.strftime("%Y%m%d_%H%M%S")
        self.current_segment_path = os.path.join(self.output_dir, f"thermal_{timestamp}.mp4")
        
        # Use software x264enc (hw didnt work :()
        pipeline = [
            'gst-launch-1.0',
            '-e',  # Enable EOS handling
            'fdsrc',
            '!', 'rawvideoparse',
            'width=640', 'height=512', 'format=bgr', 'framerate=30/1',
            '!', 'videorate',  # Add videorate to handle frame drops/duplicates
            '!', 'video/x-raw,framerate=30/1',  # Ensure consistent output framerate
            '!', 'videoconvert',
            '!', 'video/x-raw,format=I420',
            '!', 'x264enc',
            'bitrate=2000',  # 2000 kbps = 2 Mbps
            'speed-preset=ultrafast',  # Fastest preset for lowest CPU
            'tune=zerolatency',  # Low latency tuning
            'key-int-max=30',  # Keyframe every 30 frames
            '!', 'h264parse',
            '!', 'mp4mux',
            'fragment-duration=1000',  # 1 second fragments
            '!', 'filesink',
            f'location={self.current_segment_path}.tmp'
        ]
        
        # Start new recording process
        self.current_process = subprocess.Popen(
            pipeline,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        print(f"Started new segment: thermal_{timestamp}.mp4")
    
    def _add_timestamp_overlay(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Add black background for text
        cv2.rectangle(frame, (10, 10), (350, 40), (0, 0, 0), -1)
        
        # Add white text
        cv2.putText(frame, timestamp, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def _recording_loop(self):
        """Main recording loop"""
        # Pin recording thread to cores 2-3
        try:
            os.sched_setaffinity(0, {2, 3})
            print("Recording thread pinned to cores 2-3")
        except:
            pass
        self._start_new_segment()
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.recording_queue.get(timeout=0.1)
                if frame_data is None:
                    break

                frame = frame_data['frame']
                is_pooled = frame_data.get('is_pooled', False)

                # Check if we need a new segment
                if (datetime.now() - self.segment_start_time).total_seconds() >= self.segment_duration:
                    self._start_new_segment()

                # Add timestamp
                self._add_timestamp_overlay(frame)

                # Write to GStreamer
                try:
                    self.current_process.stdin.write(frame.tobytes())
                except:
                    # Pipe broken, restart recording
                    print("Recording pipe broken, restarting...")
                    self._start_new_segment()
                    self.current_process.stdin.write(frame.tobytes())
                
                # Release buffer back to pool if it came from pool
                if is_pooled:
                    self.recording_buffer_pool.release(frame)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Recording error: {e}")
                # Make sure to release buffer on error
                if 'frame' in locals() and frame_data.get('is_pooled', False):
                    self.recording_buffer_pool.release(frame)

    def add_frame(self, frame):  
        """Add frame to recording queue"""
        # Get a buffer from the recording pool
        recording_buffer = self.recording_buffer_pool.get(timeout=0)
        if recording_buffer is None:
            # No buffer available, drop frame
            return
        
        # Copy frame data to recording buffer
        np.copyto(recording_buffer, frame)
        
        try:
            self.recording_queue.put_nowait({
                'frame': recording_buffer,
                'is_pooled': True  # Only need this flag now
            })
        except queue.Full:
            # Queue full, release buffer back to pool
            self.recording_buffer_pool.release(recording_buffer)
    
    def stop(self):
        """Stop recording"""
        self.running = False
        self.recording_queue.put(None)
        self.recording_thread.join()
        
        # Clear any remaining frames in queue and release buffers
        while not self.recording_queue.empty():
            try:
                frame_data = self.recording_queue.get_nowait()
                if frame_data and frame_data.get('is_pooled', False):
                    self.recording_buffer_pool.release(frame_data['frame'])
            except queue.Empty:
                break
        
        if self.current_process:
            self.current_process.stdin.close()
            self.current_process.wait()
            
            # Rename final segment
            if self.current_segment_path and os.path.exists(self.current_segment_path + ".tmp"):
                os.rename(self.current_segment_path + ".tmp", self.current_segment_path)
        
        print("Recording stopped")

def recolor_bbox_vectorized(frame_buffer, bbox):
    """Vectorized version of bbox recoloring for better performance"""
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Ensure bbox is within image bounds
    h, w = frame_buffer.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return
    
    # Extract bbox region
    bbox_region = frame_buffer[y1:y2, x1:x2]
    
    # Convert to grayscale for processing
    gray_region = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Compute average value
    BB_value = np.mean(gray_region)
    
    # Create edge mask
    edge_mask = np.zeros(gray_region.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    
    # Create masks for different conditions
    above_avg_mask = (gray_region > BB_value) & ~edge_mask
    below_or_edge_mask = ~above_avg_mask
    
    # Calculate interpolation factor for above-average pixels
    t = np.zeros_like(gray_region)
    if BB_value < 255:
        t[above_avg_mask] = (gray_region[above_avg_mask] - BB_value) / (255 - BB_value)
    
    # Create output
    output = np.zeros_like(bbox_region, dtype=np.uint8)
    
    # Set grayscale for edge and below-average pixels
    gray_vals = gray_region.astype(np.uint8)
    output[below_or_edge_mask] = np.stack([gray_vals[below_or_edge_mask]] * 3, axis=-1)
    
    # Interpolate to red for above-average pixels
    if np.any(above_avg_mask):
        gray_component = gray_vals[above_avg_mask]
        t_vals = t[above_avg_mask]
        
        output[above_avg_mask, 0] = (gray_component * (1 - t_vals)).astype(np.uint8)  # B
        output[above_avg_mask, 1] = (gray_component * (1 - t_vals)).astype(np.uint8)  # G
        output[above_avg_mask, 2] = (gray_component * (1 - t_vals) + 255 * t_vals).astype(np.uint8)  # R
    
    # Update the frame buffer
    frame_buffer[y1:y2, x1:x2] = output

def check_wifi_connected():
    """Check if WiFi is connected"""
    try:
        result = subprocess.run(['nmcli', '-t', '-f', 'TYPE,STATE', 'con', 'show', '--active'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if '802-11-wireless:activated' in line:
                return True
        return False
    except:
        return False

def connect_wifi():
    """Connect to WiFi using priority: Primary first, then Secondary"""
    # First check if we're already connected to any network
    if check_wifi_connected():
        return True
    
    # Try primary network first
    try:
        print(f"Attempting to connect to primary network: {PRIMARY_WIFI_SSID}")
        
        subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'rescan'], capture_output=True, timeout=10)
        time.sleep(1)
        # First try to activate existing connection
        subprocess.run(['sudo', 'nmcli', 'con', 'up', 'id', PRIMARY_WIFI_SSID], 
                      capture_output=True, timeout=10)
        time.sleep(2)
        if check_wifi_connected():
            print(f"Connected to primary network: {PRIMARY_WIFI_SSID}")
            return True
        
        # If that fails, create new connection
        subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'connect', PRIMARY_WIFI_SSID, 
                       'password', PRIMARY_WIFI_PASSWORD], 
                      capture_output=True, timeout=10)
        time.sleep(2)
        if check_wifi_connected():
            print(f"Connected to primary network: {PRIMARY_WIFI_SSID}")
            return True
    except Exception as e:
        print(f"Failed to connect to primary network: {e}")
    
    # Try secondary network if primary failed
    try:
        print(f"Primary network unavailable, trying secondary: {SECONDARY_WIFI_SSID}")
        
        # First try to activate existing connection
        subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'rescan'], capture_output=True, timeout=10)
        time.sleep(1)
        subprocess.run(['sudo', 'nmcli', 'con', 'up', 'id', SECONDARY_WIFI_SSID], 
                      capture_output=True, timeout=10)
        time.sleep(2)
        if check_wifi_connected():
            print(f"Connected to secondary network: {SECONDARY_WIFI_SSID}")
            return True
        
        # If that fails, create new connection
        subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'connect', SECONDARY_WIFI_SSID, 
                       'password', SECONDARY_WIFI_PASSWORD], 
                      capture_output=True, timeout=10)
        time.sleep(2)
        if check_wifi_connected():
            print(f"Connected to secondary network: {SECONDARY_WIFI_SSID}")
            return True
    except Exception as e:
        print(f"Failed to connect to secondary network: {e}")
    
    print("Failed to connect to any configured network")
    return False

def check_camera_exists():
    """Check if FLIR camera device exists"""
    try:
        # Check for Boson camera device
        result = subprocess.run(['ls', '/dev/'], capture_output=True, text=True)
        if 'video' in result.stdout:
            # Try to check if it's specifically the FLIR
            try:
                # Quick test to see if we can access the camera
                test_cam = Boson()
                test_cam.close()
                return True
            except:
                pass
        return False
    except:
        return False

def create_disconnected_image(width=640, height=512):
    """Create a simple black image with white text"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    text = "Camera Disconnected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
    
    # Center the text
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Draw white text
    cv2.putText(img, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
    
    return img

def setup_optimal_network():
    """Configure system for optimal network streaming"""   
    # 1. Disable WiFi power management
    try:
        subprocess.run(['sudo', 'iw', 'dev', 'wlP1p1s0', 'set', 'power_save', 'off'], 
                      capture_output=True, text=True)
    except:
        pass
    
    # 2. Set Jetson to max performance
    try:
        subprocess.run(['sudo', 'nvpmodel', '-m', '0'], capture_output=True)
        subprocess.run(['sudo', 'jetson_clocks'], capture_output=True)
    except:
        pass
    
    # 3. Increase network buffers
    try:
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.tcp_wmem=4096 87380 1048576'], capture_output=True)
    except:
        pass
    # Bigger UDP buffers so the kernel doesn't drop bursts
    try:
        subprocess.run(['sudo', 'sysctl', '-w', 'net.core.rmem_max=4194304'], capture_output=True)
        subprocess.run(['sudo', 'sysctl', '-w', 'net.core.wmem_max=4194304'], capture_output=True)
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.udp_rmem_min=262144'], capture_output=True)
        subprocess.run(['sudo', 'sysctl', '-w', 'net.ipv4.udp_wmem_min=262144'], capture_output=True)
    except:
        pass


def main():
    print("Starting video streaming with confidence-based tracking...")
    print(f"Confirm threshold: {CONFIRM_THRESHOLD}, Maintain: {MAINTAIN_THRESHOLD}, Decay: {CONFIDENCE_DECAY}")
    
    # Pin main thread to core 4
    try:
        os.sched_setaffinity(0, {4})
        print("Main thread pinned to core 4")
    except:
        pass

    setup_optimal_network()

    # WiFi and camera state
    wifi_connected = False
    camera_connected = False
    last_wifi_check = 0
    last_camera_check = 0
    check_interval = 5.0
    disconnected_image = create_disconnected_image()
    
    # Wait for WiFi connection first
    while not wifi_connected:
        current_time = time.time()
        if current_time - last_wifi_check >= check_interval:
            last_wifi_check = current_time
            print("\nChecking WiFi connection...")
            wifi_connected = check_wifi_connected()
            if not wifi_connected:
                print("WiFi not connected, attempting to connect...")
                wifi_connected = connect_wifi()
            
            if wifi_connected:
                print("WiFi connected!")
            else:
                print("WiFi connection failed, will retry in 5 seconds")
        time.sleep(0.1)
    
    # Wait for camera
    while not camera_connected:
        current_time = time.time()
        if current_time - last_camera_check >= check_interval:
            last_camera_check = current_time
            print("\nChecking camera connection...")
            camera_connected = check_camera_exists()
            
            if camera_connected:
                print("Camera detected!")
            else:
                print("Camera not detected, will retry in 5 seconds")
        
        # Also keep checking WiFi
        if current_time - last_wifi_check >= check_interval:
            last_wifi_check = current_time
            if not check_wifi_connected():
                print("\nWiFi connection lost! Waiting for reconnection...")
                wifi_connected = False
                while not wifi_connected:
                    wifi_connected = connect_wifi()
                    if not wifi_connected:
                        print("WiFi reconnection failed, will retry in 5 seconds")
                        time.sleep(5)
                print("WiFi reconnected!")
        
        time.sleep(0.1)
    
    # Both WiFi and camera are connected, proceed with original initialization
    # Initialize YOLO
    model = YOLO(ENGINE_PATH, task='detect')
    print("Warming up model...")
    dummy_frame = np.random.randint(0, 255, (512, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_frame)
    
    # Initialize components
    camera = FLIRCamera()
    recorder = ContinuousRecorder(
        output_dir="/home/ibrahim/blackbox",  
        segment_duration=10,  # 10 second segments
        max_segments=60  # 60 segments = 10 mins total
    )

    buffer_pool = FrameBufferPool(640, 512, 3, pool_size=20)
    inference = TrackedInferencePipeline(model, buffer_pool, camera, recorder=recorder, num_threads=INFERENCE_THREADS)
    streamer = VideoStreamerWithBuffer(port=8888, buffer_pool=buffer_pool)
    
    # Frame counting
    frame_count = 0
    sent_count = 0
    inference_count = 0
    
    # FPS calculation
    fps_sent_last_second = 0
    fps_read_last_second = 0
    fps_inference_last_second = 0
    fps_timer = time.time()
    
    # Target 30 FPS streaming
    target_fps = 30
    frame_interval = 1.0 / target_fps
    next_send_time = time.time()
    
    # Ready frames from inference
    ready_frames = {}
    
    try:
        while True:
            # Check WiFi periodically
            current_time = time.time()
            if current_time - last_wifi_check >= check_interval:
                last_wifi_check = current_time
                if not check_wifi_connected():
                    print("\nWiFi connection lost! Streaming disconnected image...")
                    # Stream disconnected image until WiFi returns
                    while not check_wifi_connected():
                        if current_time >= next_send_time:
                            streamer.send(disconnected_image.copy())
                            recorder.add_frame(disconnected_image)
                            next_send_time += frame_interval
                            if next_send_time < current_time:
                                next_send_time = current_time + frame_interval
                        
                        if current_time - last_wifi_check >= check_interval:
                            last_wifi_check = current_time
                            print("Attempting WiFi reconnection...")
                            connect_wifi()
                        
                        current_time = time.time()
                        time.sleep(0.01)
                    
                    print("WiFi reconnected!")
            
            # Read frame
            ret, frame = camera.read()
            if not ret or frame is None:
                # Camera disconnected - stream disconnected image
                print("\nCamera disconnected! Streaming disconnected image...")
                camera_connected = False
                
                while not camera_connected:
                    current_time = time.time()
                    
                    # Stream disconnected image
                    if current_time >= next_send_time:
                        streamer.send(disconnected_image.copy())
                        recorder.add_frame(disconnected_image)
                        sent_count += 1
                        fps_sent_last_second += 1
                        
                        next_send_time += frame_interval
                        if next_send_time < current_time:
                            next_send_time = current_time + frame_interval
                    
                    # Check camera periodically
                    if current_time - last_camera_check >= check_interval:
                        last_camera_check = current_time
                        print("Checking camera connection...")
                        camera_connected = check_camera_exists()
                        
                        if camera_connected:
                            print("Camera reconnected! Reinitializing...")
                            camera.release()
                            camera = FLIRCamera()
                            # Test read
                            ret, test_frame = camera.read()
                            if not ret:
                                camera_connected = False
                                print("Camera initialization failed")
                    
                    # Also check WiFi
                    if current_time - last_wifi_check >= check_interval:
                        last_wifi_check = current_time
                        if not check_wifi_connected():
                            print("\nWiFi also lost! Waiting for both connections...")
                            # Need both back
                            while not check_wifi_connected():
                                connect_wifi()
                                time.sleep(5)
                            print("WiFi reconnected!")
                    
                    # Print stats
                    if current_time - fps_timer >= 1.0:
                        print(f"\rStream: {fps_sent_last_second} FPS (Disconnected)", end='', flush=True)
                        fps_sent_last_second = 0
                        fps_timer = current_time
                    
                    time.sleep(0.01)
                
                continue
            
            camera.check_ffc_needed()
            frame_count += 1
            fps_read_last_second += 1
            
            # Get buffer and copy frame
            buffer = buffer_pool.get()
            if buffer is not None:
                np.copyto(buffer, frame)
                # Send to inference
                if inference.process(buffer, frame_count):
                    inference_count += 1
            
            # Collect inference results
            while True:
                result = inference.get_result(timeout=0.0001)
                if result:
                    processed_buffer, frame_id = result
                    ready_frames[frame_id] = processed_buffer  # Just store the buffer
                    fps_inference_last_second += 1
                    
                    # Keep only recent frames
                    if len(ready_frames) > 15:
                        oldest_id = min(ready_frames.keys())
                        old_buffer = ready_frames.pop(oldest_id)
                        buffer_pool.release(old_buffer)
                else:
                    break
            
            current_time = time.time()
            
            # Send frames at target rate
            if current_time >= next_send_time:
                if ready_frames:
                    # Send newest processed frame
                    newest_id = max(ready_frames.keys())
                    buffer_to_send = ready_frames.pop(newest_id)
                    
                    if streamer.send(buffer_to_send):
                        recorder.add_frame(buffer_to_send)
                        sent_count += 1
                        fps_sent_last_second += 1
                    
                    # Clean up older frames
                    for old_id in list(ready_frames.keys()):
                        if old_id < newest_id:
                            old_buffer = ready_frames.pop(old_id)
                            buffer_pool.release(old_buffer)
                
                # Schedule next frame
                next_send_time += frame_interval
                if next_send_time < current_time:
                    next_send_time = current_time + frame_interval
            
            # Print stats every second
            if current_time - fps_timer >= 1.0:
                # Count active tracks across all trackers
                total_tracks = sum(len([t for t in tracker.tracks if t.confirmed]) 
                                 for tracker in inference.trackers)
                
                print(f"\rCamera: {fps_read_last_second} FPS | "
                      f"Inference: {fps_inference_last_second} FPS | "
                      f"Stream: {fps_sent_last_second} FPS | "
                      f"Tracks: {total_tracks} | "
                      f"Ready: {len(ready_frames)}", end='', flush=True)
                
                fps_read_last_second = 0
                fps_sent_last_second = 0
                fps_inference_last_second = 0
                fps_timer = current_time
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.release()
        inference.stop()
        streamer.stop()
        recorder.stop()
        
        # Clean up buffers
        for _, buffer in ready_frames.items():
            buffer_pool.release(buffer)
        
        print(f"\nTotal: Read={frame_count}, Sent={sent_count}, Inference={inference_count}")

if __name__ == '__main__':
    main()
