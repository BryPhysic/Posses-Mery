import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import os
from datetime import datetime, timedelta

def get_available_cameras():
    available_cameras = []
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            print(f"Camera {i} is available")
            cap.release()
        else:
            print(f"Camera {i} is not available")
    
    return available_cameras
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
                
                
                #label = f"{class_name} {box.conf:.2f}" 
                label = f"Pelea" 
                cv2.putText(image, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image, violence_detected

def get_unique_filename(base_path, base_name, ext):
    counter = 1
    while True:
        file_path = os.path.join(base_path, f"{base_name}_{counter}{ext}")
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def show_webcam_detection():
    st.title("Detección con Cámara Web")

    available_models = get_available_models()
    if not available_models:
        st.error("No se encontraron modelos en la carpeta /Models.")
        return

    selected_model = st.selectbox("Selecciona un modelo:", available_models)

    @st.cache_resource
    def load_model(model_path):
        return YOLO(f'Models/{model_path}')

    model = load_model(selected_model)

    available_cameras = get_available_cameras()
    if not available_cameras:
        st.error("No se encontraron cámaras disponibles.")
        return

    # Usar session_state para almacenar la cámara seleccionada
    if 'selected_camera' not in st.session_state:
        st.session_state.selected_camera = available_cameras[0]

    # Permitir la selección de cámara solo si no se está ejecutando la detección
    if not st.session_state.get('run_camera', False):
        st.session_state.selected_camera = st.selectbox("Selecciona una cámara:", available_cameras, index=available_cameras.index(st.session_state.selected_camera))

    enable_recording = st.checkbox("Habilitar grabación de video", value=False)

    confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Iniciar/Detener Cámara", key="start_stop_camera"):
            st.session_state.run_camera = not st.session_state.run_camera
    with col2:
        if st.button("Iniciar/Detener Detección", key="start_stop_detection"):
            st.session_state.detection_active = not st.session_state.detection_active

    stframe = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(st.session_state.selected_camera)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        recording = False
        fps = 30
        last_recording_time = datetime.min

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        while st.session_state.run_camera:
            if not cap.isOpened():
                cap = cv2.VideoCapture(st.session_state.selected_camera)
                if not cap.isOpened():
                    st.error(f"No se pudo reiniciar la cámara {st.session_state.selected_camera}.")
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                st.warning(f"No se pudo leer un frame de la cámara {st.session_state.selected_camera}. Intentando reiniciar...")
                cap.release()
                time.sleep(1)
                continue

            if st.session_state.detection_active:
                results = model.predict(frame, conf=confidence_threshold)
                annotated_frame, violence_detected = plot_custom_bboxes(frame, results, color=(255, 0, 0))
                
                if violence_detected:
                    alert_placeholder.error("⚠️ ¡ALERTA! Se ha detectado una posible pelea  ⚠️")
                else:
                    alert_placeholder.empty()
                
                current_time = datetime.now()
                if violence_detected and enable_recording and not recording and (current_time - last_recording_time) > timedelta(minutes=1):
                    recording = True
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    video_filename = get_unique_filename(results_dir, f"violence_detected_{timestamp}", ".mp4")
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame.shape[1], frame.shape[0]))

                if recording and violence_detected:
                    out.write(annotated_frame)
                elif recording and not violence_detected:
                    recording = False
                    out.release()
                    last_recording_time = current_time
                    st.success(f"Video guardado: {video_filename}")
                    # Reiniciar el modelo YOLO
                    model = load_model(selected_model)
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            time.sleep(0.001)

        cap.release()
        if out is not None:
            out.release()
    else:
        st.info("Presiona 'Iniciar/Detener Cámara' para comenzar.")

if __name__ == "__main__":
    show_webcam_detection()