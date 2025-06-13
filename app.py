import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Cargar el modelo
modelo = load_model('modelo_mejorado_cnn_original.h5')
TAMANO_IMG = 150  # <-- pon el tamaño exacto en el que entrenaste el modelo

# Preprocesamiento (escala de grises, resize y normalización)
def preprocesar_imagen(imagen_cv2):
    imagen_gris = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2GRAY)
    imagen_redimensionada = cv2.resize(imagen_gris, (TAMANO_IMG, TAMANO_IMG))
    imagen_normalizada = imagen_redimensionada.astype('float32') / 255.0
    imagen_final = imagen_normalizada.reshape(1, TAMANO_IMG, TAMANO_IMG, 1)
    return imagen_final

# Título
st.title("🧠 Clasificación Humanos vs Caballos en Vivo")

# Inicializar la cámara
camera = st.camera_input("Captura una imagen")

# Cuando haya foto capturada:
if camera is not None:
    image = Image.open(camera)
    image_np = np.array(image)

    st.image(image_np, caption="Imagen capturada", use_column_width=True)

    imagen_procesada = preprocesar_imagen(image_np)
    prediccion = modelo.predict(imagen_procesada)[0][0]

    if prediccion > 0.5:
        st.success(f"Predicción: HUMANO ({prediccion:.2f})")
    else:
        st.success(f"Predicción: CABALLO ({1 - prediccion:.2f})")
