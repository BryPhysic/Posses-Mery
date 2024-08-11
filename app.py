import streamlit as st
from sections.home import show_home
from sections.video_detection import show_video_detection
from sections.webcam_detection import show_webcam_detection

def main():
    st.sidebar.title("Menú de Navegación")

    # Inicializar `st.session_state.sections` si no existe
    if 'sections' not in st.session_state:
        st.session_state.sections = "Inicio"

    # Navegación con botones
    if st.sidebar.button("Inicio"):
        st.session_state.sections = "Inicio"
    if st.sidebar.button("Detección en Video"):
        st.session_state.sections = "Detección en Video"
    if st.sidebar.button("Detección con Cámara Web"):
        st.session_state.sections = "Detección con Cámara Web"

    # Mostrar la página seleccionada
    if st.session_state.sections == "Inicio":
        show_home()
    elif st.session_state.sections == "Detección en Video":
        show_video_detection()
    elif st.session_state.sections == "Detección con Cámara Web":
        show_webcam_detection()

if __name__ == "__main__":
    main()