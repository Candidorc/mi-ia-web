import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Digit Recognizer")
st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro.")

# 1. Cargar el modelo guardado
@st.cache_resource
def load_my_model():
    # Asegúrate de que el archivo 'modelo_mnist.keras' esté en la misma carpeta que este script
    return tf.keras.models.load_model('modelo_mnist.keras')

model = load_my_model()

# 2. Crear el lienzo (Canvas) para dibujar
canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=20,
    stroke_color="white",
    background_color="black", 
    height=280, 
    width=280,
    drawing_mode="freedraw", 
    key="canvas",
)

# 3. Procesar el dibujo y predecir
if canvas_result.image_data is not None:
    # Convertir el dibujo a 28x28 píxeles (formato que usa el modelo MNIST)
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    
    # Pasar a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalizar los píxeles (0 a 1)
    img = img / 255.0 
    
    # Cambiar la forma para que la IA la entienda: (1 muestra, 28px, 28px, 1 canal de color)
    img_input = img.reshape(1, 28, 28, 1)

    # Predicción
    # Predicción
    pred = model.predict(img_input)
    clase = np.argmax(pred) # Esto busca qué posición tiene el valor más alto (0, 1, 2...)
    confianza = pred[0][clase] # Esto saca el porcentaje exacto de esa posición
    

    # 4. Mostrar resultados con Umbral de Seguridad
    # Basado en tu análisis, un umbral del 80% o 95% ayuda a reducir errores[cite: 1]
    st.subheader(f"Resultado: {clase}")
    
    if confianza < 0.80:
        st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
    else:
        st.success(f"Confianza alta: {confianza:.2%}")
        
    # Visualización de probabilidades para cada dígito (0-9)
    st.bar_chart(pred[0])