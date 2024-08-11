# pages/video_detection.py
import streamlit as st
from video_processing import detect_violence_from_video

def show_video_detection():
    st.title("Detección de Peleas en Video")

    video_file = st.file_uploader("Cargar un archivo de video", type=["mp4", "avi", "mov"])
    if video_file:
        fight_times = detect_violence_from_video(video_file.name)
        st.write("Tiempos de detección de conflictos (en segundos):", fight_times)