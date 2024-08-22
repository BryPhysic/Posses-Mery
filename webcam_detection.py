import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import numpy as np
from datetime import datetime, timedelta
import os
import time

# Funciones auxiliares

def get_available_models():
    models_dir = 'Models'
    return [f for f in os.listdir(models_dir) if f.endswith('.pt')]

def plot_custom_bboxes(image, results, color=(255, 0, 0)):
    violence_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls)
            class_name = r.names[class_id]
            
            if class_name == 'Violence':
                violence_detected = True
                b = box.xyxy[0]
                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                label = f"Pelea"
                cv2.putText(image, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image, violence_detected

# Clase para el procesamiento del video
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.enable_recording = False
        self.recording = False
        self.last_recording_time = datetime.min
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.out = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = 30

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Realizar la detección con YOLO
        results = self.model.predict(img, conf=self.confidence_threshold)
        annotated_frame, violence_detected = plot_custom_bboxes(img, results, color=(255, 0, 0))
        
        # Lógica de grabación si se detecta violencia
        if violence_detected and self.enable_recording:
            current_time = datetime.now()
            if not self.recording and (current_time - self.last_recording_time) > timedelta(minutes=0.5):
                self.recording = True
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                video_filename = os.path.join(self.results_dir, f"violence_detected_{timestamp}.mp4")
                self.out = cv2.VideoWriter(video_filename, self.fourcc, self.fps, (img.shape[1], img.shape[0]))
            if self.recording:
                self.out.write(annotated_frame)
        elif self.recording:
            self.recording = False
            self.out.release()
            self.last_recording_time = datetime.now()
            st.success(f"Video guardado.")
        
        return annotated_frame

# Función principal de la aplicación
def show_webcam_detection():
    st.title("Sistema de Detección de Violencia")
    st.write("---")

    available_models = get_available_models()
    if not available_models:
        st.error("No se encontraron modelos en la carpeta /Models.")
        return

    selected_model = st.selectbox("Selecciona un modelo de detección:", available_models)

    @st.cache_resource
    def load_model(model_path):
        return YOLO(f'Models/{model_path}')

    model = load_model(selected_model)

    st.write("---")

    # Parámetros configurables en la interfaz
    confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    enable_recording = st.checkbox("Habilitar Grabación", value=False)

    # Iniciar el streaming de video con WebRTC
    webrtc_ctx = webrtc_streamer(
        key="detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoTransformer(model, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        webrtc_ctx.video_processor.enable_recording = enable_recording

if __name__ == "__main__":
    show_webcam_detection()