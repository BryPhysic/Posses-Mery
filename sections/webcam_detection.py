import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Configura la página para usar el modo ancho
st.set_page_config(layout="wide")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Aquí puedes integrar tu modelo YOLO para el análisis
        # Por ahora, solo devolvemos el frame original
        return img

def show_webcam_detection():
    st.title("Detección de Peleas con Cámara Web")

    # Configuración CSS para hacer el video más grande
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
        }
        .streamlit-expanderHeader {
            display: none;
        }
        .streamlit-expanderContent {
            overflow: hidden;
        }
        .stVideo {
            width: 100%;
            height: 80vh;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        </style>
    """, unsafe_allow_html=True)

    # Usa todo el ancho de la página
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={
            "video": {
                "width": 1920,
                "height": 1080,
                "frameRate": {"ideal": 60}
            },
            "audio": True,  # Desactivamos el audio para centrarnos en el video
        },
        async_processing=True,
    )

    if webrtc_ctx.video_transformer:
        st.write("Iniciando detección de peleas...")

if __name__ == "__main__":
    show_webcam_detection()