# app.py
import streamlit as st
from pages.home import show_home
from pages.video_detection import show_video_detection
from pages.webcam_detection import show_webcam_detection

def main():
    st.sidebar.title("Menú de Navegación")
    page = st.sidebar.radio("Ir a", ("Inicio", "Detección en Video", "Detección con Cámara Web"))

    if page == "Inicio":
        show_home()
    elif page == "Detección en Video":
        show_video_detection()
    elif page == "Detección con Cámara Web":
        show_webcam_detection()

if __name__ == "__main__":
    main()