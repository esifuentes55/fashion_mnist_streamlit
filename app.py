import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fashion_mnist_resnet.h5")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title(" Clasificador de Prendas - Fashion MNIST")
option = st.radio("¿Cómo deseas ingresar la imagen?", ["Subir archivo", "Tomar foto con cámara"])

if option == "Subir archivo":
    uploaded_file = st.file_uploader("📷 Sube una imagen", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
elif option == "Tomar foto con cámara":
    camera_image = st.camera_input("Toma una foto")
    if camera_image:
        image = Image.open(camera_image).convert("L").resize((28, 28))

if 'image' in locals():
    st.image(image, caption="Imagen usada", width=150)
    img_array = np.expand_dims(np.array(image) / 255.0, axis=(0, -1))
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.markdown(f"###  Predicción: **{class_names[class_index]}**")
    st.markdown(f"Confianza: **{prediction[0][class_index]:.2%}**")
