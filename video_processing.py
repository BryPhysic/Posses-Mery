# video_processing.py
import cv2
import time
import streamlit as st
from model import load_model

model = load_model()

def process_frame(frame, model):
    results = model(frame)
    return results

def detect_violence_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fight_times = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = process_frame(frame, model)
        detections = results[0].boxes
        for detection in detections:
            if 'fight' in detection.cls:  # Reemplaza 'fight' con el nombre de la clase en tu modelo
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # En segundos
                fight_times.append(timestamp)

        st.image(frame, channels="BGR")
    
    cap.release()
    return fight_times

def detect_violence_from_webcam():
    cap = cv2.VideoCapture(0)
    fight_times = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = process_frame(frame, model)
        detections = results[0].boxes
        for detection in detections:
            if 'fight' in detection.cls:  # Reemplaza 'fight' con el nombre de la clase en tu modelo
                current_time = time.time() - start_time
                fight_times.append(current_time)
        
        st.image(frame, channels="BGR")
    
    cap.release()
    return fight_times