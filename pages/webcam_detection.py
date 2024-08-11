# pages/webcam_detection.py
import streamlit as st
from video_processing import detect_violence_from_webcam

def show_webcam_detection():
    st.title("Detección de Peleas con Cámara Web")

    st.write("Accediendo a la cámara web...")
    fight_times = detect_violence_from_webcam()
    st.write("Tiempos de detección de conflictos (en segundos):", fight_times)

    