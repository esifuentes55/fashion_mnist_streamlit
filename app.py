import cv2
import numpy as np
import tensorflow as tf

# Cargar modelo CNN
model = tf.keras.models.load_model("modelo_cnn_tf_flowers.h5")

# Tamaño de entrada esperado
IMG_SIZE = 180

# Etiquetas
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Función de preprocesamiento
def preprocess_frame(frame):
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# Inicializar webcam
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predecir clase
    input_image = preprocess_frame(frame)
    prediction = model.predict(input_image)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"

    # Mostrar etiqueta en pantalla
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Clasificación en Vivo - CNN", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

