import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from datetime import timedelta
import numpy as np

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
                
                # Añadir etiqueta
                label = f"{class_name} 97%"
                cv2.putText(image, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image, violence_detected

def show_video_detection():
    st.title("Detección de Violencia en Video")

    available_models = get_available_models()
    if not available_models:
        st.error("No se encontraron modelos en la carpeta /Models.")
        return

    selected_model = st.selectbox("Selecciona un modelo:", available_models)

    @st.cache_resource
    def load_model(model_path):
        return YOLO(f'Models/{model_path}')

    model = load_model(selected_model)

    video_file = st.file_uploader("Cargar un archivo de video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        try:
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                temp_file_name = tfile.name

            cap = cv2.VideoCapture(temp_file_name)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            st.write(f"FPS: {fps}")
            st.write(f"Duración: {timedelta(seconds=duration)}")

            confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
            save_video = st.checkbox("Guardar video procesado", value=False)

            if st.button("Procesar Video"):
                progress_bar = st.progress(0)
                violence_times = []
                
                
                if save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('video_procesado.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

              
                video_placeholder = st.empty()
                
                for frame_num in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=confidence_threshold)
                    annotated_frame, violence_detected = plot_custom_bboxes(frame, results)

                    if violence_detected:
                        time = frame_num / fps
                        violence_times.append(time)

                    # Mostrar el frame procesado
                    video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

                    if save_video:
                        out.write(annotated_frame)

                    progress_bar.progress((frame_num + 1) / total_frames)

                if save_video:
                    out.release()
                    st.success("Video procesado guardado como 'video_procesado.mp4'")

                if violence_times:
                    st.write("Momentos de violencia detectados:")
                    for time in violence_times:
                        st.write(f"- {timedelta(seconds=time)}")
                else:
                    st.write("No se detectó violencia en el video.")

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento: {str(e)}")

        finally:
           
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            
            
            try:
                os.unlink(temp_file_name)
            except Exception as e:
                st.warning(f"No se pudo eliminar el archivo temporal: {str(e)}")

if __name__ == "__main__":
    show_video_detection()