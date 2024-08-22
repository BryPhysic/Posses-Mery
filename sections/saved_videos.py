import streamlit as st
import os

RESULTS_DIR = "results"

def list_recorded_videos(results_dir=RESULTS_DIR):
    return [f for f in os.listdir(results_dir) if f.endswith('.mp4')]

def show_saved_videos():
    st.title("Videos Guardados")
    st.write("---")

    recorded_videos = list_recorded_videos()
    
    if recorded_videos:
        selected_video = st.selectbox("Selecciona un video para ver, descargar o eliminar:", recorded_videos)
        video_path = os.path.join(RESULTS_DIR, selected_video)

        # Opciones: Ver video, Descargar video o Eliminar video
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Ver Video"):
                # Mostrar el video si es compatible
                try:
                    st.video(video_path)
                except Exception as e:
                    st.error(f"No se puede reproducir el video: {e}")

        with col2:
            with open(video_path, "rb") as file:
                st.download_button(
                    label="Descargar Video",
                    data=file,
                    file_name=selected_video,
                    mime="video/mp4"
                )

        with col3:
            if st.button("Eliminar Video"):
                os.remove(video_path)
                st.warning(f"Video '{selected_video}' eliminado.")
                st.experimental_rerun()  # Recargar la página para actualizar la lista de videos
    else:
        st.write("No se han encontrado videos guardados.")