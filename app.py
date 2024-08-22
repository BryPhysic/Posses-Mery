import streamlit as st
from sections.home import show_home
from sections.video_detection import show_video_detection
from sections.webcam_detection import show_webcam_detection
from sections.saved_videos import show_saved_videos

def main():
    st.sidebar.title("Menú de Navegación")

    # Crear un menú desplegable en la barra lateral
    menu_options = ["Inicio", "Detección en Video", "Detección con Cámara Web", "Videos Guardados"]
    selected_option = st.sidebar.selectbox("Selecciona una opción:", menu_options)

    # Mostrar la página seleccionada
    if selected_option == "Inicio":
        show_home()
    elif selected_option == "Detección en Video":
        show_video_detection()
    elif selected_option == "Detección con Cámara Web":
        show_webcam_detection()
    elif selected_option == "Videos Guardados":
        show_saved_videos()

    # Agregar información adicional en la barra lateral
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Esta aplicación utiliza un modelo YOLO para detectar violencia en videos y transmisiones de cámara web."
    )
    st.sidebar.warning(
        "Nota: Esta aplicación es solo para fines demostrativos. "
        "En caso de emergencia, contacte a las autoridades locales."
    )

if __name__ == "__main__":
    main()