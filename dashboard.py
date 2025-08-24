# dashboard.py

# STEP 1: FIX FOR OMP ERROR & PATHS (MUST BE FIRST)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Suppress all warnings for a cleaner deployment
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import tempfile
import streamlit as st
import cv2
import torch
import torch.nn as nn
import time
import math
import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms

# STEP 2: DYNAMIC PATH SETUP
FILE = Path(__file__).resolve()
YOLOv5_ROOT = FILE.parents[0] / 'yolov5'
if str(YOLOv5_ROOT) not in sys.path:
    sys.path.append(str(YOLOv5_ROOT))

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from sort import *

# STEP 3: SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide")

# --- Constants & Configuration ---
YOLO_WEIGHTS = 'yolov5/yolov5s.pt'
ANOMALY_MODEL_PATH = 'anomaly_detector.pth' # Using the PyTorch model
IMG_SIZE_YOLO = (640, 640)
IMG_SIZE_ANOMALY = (256, 256)
CLASSES_TO_DETECT = [0, 24, 26, 28] # person, backpack, handbag, suitcase

# --- PyTorch Autoencoder Model Definition ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Helper Function for IoU Calculation ---
def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - wh
    return wh / union

# --- Cached Model Loading ---
@st.cache_resource
def load_yolo_model():
    device = select_device('')
    model = DetectMultiBackend(YOLO_WEIGHTS, device=device, dnn=False, data=YOLOv5_ROOT / 'data/coco128.yaml')
    return model

@st.cache_resource
def load_anomaly_model():
    device = select_device('')
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(ANOMALY_MODEL_PATH, map_location=device))
    model.eval()
    return model

yolo_model = load_yolo_model()
anomaly_model = load_anomaly_model()
yolo_names = yolo_model.names
device = select_device('')

# --- Main Video Processing Function ---
def process_video(video_path, conf_slider, iou_slider, loitering_enabled, loitering_time, loitering_dist, abandon_enabled, abandon_time, abandon_dist, anomaly_enabled, anomaly_thresh):
    mot_tracker = Sort()
    tracked_items = {}
    alerts = []
    anomaly_scores = deque(maxlen=100)
    anomaly_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(IMG_SIZE_ANOMALY),
        transforms.ToTensor(),
    ])
    loss_function = nn.MSELoss()

    cap = cv2.VideoCapture(video_path)
    
    video_placeholder = st.empty()
    alerts_log_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("The video has ended.")
            break

        # --- ANOMALY DETECTION (PYTORCH AUTOENCODER) ---
        if anomaly_enabled:
            input_tensor = anomaly_transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstructed_tensor = anomaly_model(input_tensor)
                reconstruction_error = loss_function(reconstructed_tensor, input_tensor).item()
            anomaly_scores.append(reconstruction_error)
            if reconstruction_error > anomaly_thresh and len(alerts) < 100:
                alerts.append(f"[Anomaly Alert] Unusual activity! Score: {reconstruction_error:.4f}")

        # --- OBJECT DETECTION (YOLOv5) ---
        img_yolo = cv2.resize(frame, IMG_SIZE_YOLO)
        img_yolo = img_yolo.transpose((2, 0, 1))[::-1]
        img_yolo = np.ascontiguousarray(img_yolo)
        img_yolo = torch.from_numpy(img_yolo).to(device)
        img_yolo = img_yolo.half() if yolo_model.fp16 else img_yolo.float()
        img_yolo /= 255.0
        if len(img_yolo.shape) == 3: img_yolo = img_yolo[None]

        pred = yolo_model(img_yolo, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_slider, iou_slider, classes=CLASSES_TO_DETECT, agnostic=False)

        annotator = Annotator(frame, line_width=2, example=str(yolo_names))
        
        original_detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img_yolo.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    original_detections.append([*xyxy, conf, cls])

        # --- TRACKING & RULE-BASED LOGIC ---
        detections_for_sort = np.array([[*d[:4], d[4]] for d in original_detections]) if original_detections else np.empty((0, 5))
        track_bbs_ids = mot_tracker.update(detections_for_sort)

        if len(track_bbs_ids) > 0 and len(original_detections) > 0:
            iou = iou_batch(track_bbs_ids[:,:4], np.array(original_detections)[:,:4])
            best_match_indices = np.argmax(iou, axis=1)
            final_tracks = [np.append(track, original_detections[best_match_indices[i]][5]) for i, track in enumerate(track_bbs_ids) if iou[i, best_match_indices[i]] > 0.3]
        else:
            final_tracks = []

        current_time = time.time()
        current_track_ids = {int(t[4]) for t in final_tracks}
        for tid in list(tracked_items.keys()):
            if tid not in current_track_ids: del tracked_items[tid]

        for track in final_tracks:
            x1, y1, x2, y2, track_id, cls = track
            track_id, cls = int(track_id), int(cls)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            if track_id not in tracked_items:
                tracked_items[track_id] = {'class': cls, 'positions': [], 'start_time': current_time, 'alert_triggered': False}
            
            tracked_items[track_id]['positions'].append((center_x, center_y))

            if loitering_enabled and cls == 0 and not tracked_items[track_id]['alert_triggered']:
                start_point = tracked_items[track_id]['positions'][0]
                elapsed_time = current_time - tracked_items[track_id]['start_time']
                distance = math.sqrt((center_x - start_point[0])**2 + (center_y - start_point[1])**2)
                if elapsed_time > loitering_time and distance < loitering_dist:
                    alerts.append(f"[Rule Alert] Person {track_id} is loitering!")
                    tracked_items[track_id]['alert_triggered'] = True

            if abandon_enabled and cls != 0 and not tracked_items[track_id]['alert_triggered']:
                if len(tracked_items[track_id]['positions']) > 10:
                    start_point = tracked_items[track_id]['positions'][0]
                    distance = math.sqrt((center_x - start_point[0])**2 + (center_y - start_point[1])**2)
                    if distance < 10:
                        is_person_near = any(math.sqrt((center_x - p_data['positions'][-1][0])**2 + (center_y - p_data['positions'][-1][1])**2) < abandon_dist for p_id, p_data in tracked_items.items() if p_data['class'] == 0)
                        elapsed_time = current_time - tracked_items[track_id]['start_time']
                        if not is_person_near and elapsed_time > abandon_time:
                            alerts.append(f"[Rule Alert] Object {yolo_names[cls]} (ID: {track_id}) may be abandoned!")
                            tracked_items[track_id]['alert_triggered'] = True
            
            label = f'{yolo_names[cls]} ID:{track_id}'
            color = colors(cls, True) if not tracked_items[track_id]['alert_triggered'] else (0, 0, 255)
            annotator.box_label((int(x1), int(y1), int(x2), int(y2)), label, color=color)
        
        # --- UPDATE UI ---
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # **FIX 1: Display alerts in a compact list format**
        with alerts_log_placeholder.container():
            st.write("**Latest Alerts:**")
            # Display the last 10 alerts
            for alert in reversed(alerts[-10:]):
                if "[Anomaly Alert]" in alert:
                    st.markdown(f"🚨 {alert}")
                else:
                    st.markdown(f"⚠️ {alert}")
        
        chart_placeholder.line_chart(np.array(anomaly_scores))

    cap.release()

# --- STREAMLIT APP UI ---
st.title("Hybrid AI Surveillance System 🏆 (Full PyTorch)")
st.sidebar.title("Settings")
conf_slider = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.3, 0.05)
iou_slider = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
st.sidebar.title("Rule-Based Alerts")
loitering_enabled = st.sidebar.checkbox("Enable Loitering Detection", value=True)
loitering_time = st.sidebar.number_input("Loitering Time (s)", 1, value=10)
loitering_dist = st.sidebar.number_input("Loitering Distance (px)", 1, value=50)
abandon_enabled = st.sidebar.checkbox("Enable Object Abandonment", value=True)
abandon_time = st.sidebar.number_input("Abandonment Time (s)", 1, value=15)
abandon_dist = st.sidebar.number_input("Person Proximity (px)", 1, value=100)
st.sidebar.title("Anomaly Detection")
anomaly_enabled = st.sidebar.checkbox("Enable General Anomaly Detection", value=True)
anomaly_thresh = st.sidebar.slider("Anomaly Sensitivity", 0.0005, 0.01, 0.002, 0.0001, format="%.4f", help="Lower value means more sensitive to anomalies.")
st.sidebar.title("Video Source")
uploaded_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

# --- Main Layout & Execution ---
col1, col2 = st.columns([3, 1])
with col1:
    st.header("Live Video Feed")
    video_placeholder = st.empty()
    st.subheader("Live Anomaly Score")
    chart_placeholder = st.empty()
with col2:
    st.header("Alerts Log")
    alerts_log_placeholder = st.empty()

if st.sidebar.button("Start Analysis"):
    video_path = 'yolov5/data/people-walking.mp4' # Default video
    tfile = None
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    
    st.info(f"Starting analysis on video: {os.path.basename(video_path)}")
    try:
        process_video(video_path, conf_slider, iou_slider, loitering_enabled, loitering_time, loitering_dist, abandon_enabled, abandon_time, abandon_dist, anomaly_enabled, anomaly_thresh)
    finally:
        if tfile is not None:
            tfile.close()
            os.remove(tfile.name)
else:
    st.info("Adjust settings in the sidebar and click 'Start Analysis'")
