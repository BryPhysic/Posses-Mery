import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from datetime import timedelta

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
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                
                # Añadir etiqueta
                label = f"{class_name}"
                cv2.putText(image, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image, violence_detected

def show_video_detection():
    st.title("Detección de Violencia en Video")

    # Obtener modelos disponibles
    available_models = get_available_models()
    if not available_models:
        st.error("No se encontraron modelos en la carpeta /Models.")
        return

    # Selector de modelo
    selected_model = st.selectbox("Selecciona un modelo:", available_models)

    # Cargar el modelo YOLO
    @st.cache_resource
    def load_model(model_path):
        return YOLO(f'Models/{model_path}')

    model = load_model(selected_model)

    # Subida de video
    video_file = st.file_uploader("Cargar un archivo de video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Guardar el archivo de video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())

        # Abrir el video
        cap = cv2.VideoCapture(tfile.name)
        
        # Obtener información del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        st.write(f"FPS: {fps}")
        st.write(f"Duración: {timedelta(seconds=duration)}")

        # Slider para ajustar la confianza
        confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        # Botón para iniciar el procesamiento
        if st.button("Procesar Video"):
            progress_bar = st.progress(0)
            violence_times = []
            
            # Procesar el video
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Realizar la detección
                results = model.predict(frame, conf=confidence_threshold)

                # Dibujar los resultados en el frame
                annotated_frame, violence_detected = plot_custom_bboxes(frame, results)

                if violence_detected:
                    time = frame_num / fps
                    violence_times.append(time)

                # Actualizar la barra de progreso
                progress_bar.progress((frame_num + 1) / total_frames)

            # Mostrar los resultados
            if violence_times:
                st.write("Momentos de violencia detectados:")
                for time in violence_times:
                    st.write(f"- {timedelta(seconds=time)}")
            else:
                st.write("No se detectó violencia en el video.")

        # Cerrar el video y eliminar el archivo temporal
        cap.release()
        os.unlink(tfile.name)

if __name__ == "__main__":
    show_video_detection()