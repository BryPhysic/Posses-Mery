import streamlit as st

# Título de la aplicación
st.title("Aplicación de Reconocimiento de Peleas")

# Barra lateral para la navegación
menu = ["Carátula", "Reconocimiento"]
choice = st.sidebar.selectbox("Selecciona una opción", menu)

if choice == "Carátula":
    st.subheader("Carátula")
    st.write("""
    ### Bienvenido a la aplicación de reconocimiento de peleas.
    
    Esta aplicación está diseñada para reconocer y analizar peleas en videos utilizando un modelo de YOLO.
    
    - **Carátula**: Información sobre la aplicación.
    - **Reconocimiento**: Cargar un video para que el modelo detecte peleas.
    """)
    st.image("Images/Peleas download.jpeg", caption="Aplicación de Reconocimiento de Peleas", use_column_width=True)
elif choice == "Reconocimiento":
    st.subheader("Reconocimiento")
    st.write("Carga un video para que el modelo de YOLO reconozca peleas.")
    video_file = st.file_uploader("Cargar video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        st.video(video_file)
        st.write("El reconocimiento de peleas estará disponible aquí después de integrar el modelo de YOLO.")