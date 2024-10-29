from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

loaded_model = tf.keras.models.load_model('trained_model.keras')
classes_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post('/predict')
def predict():
    # Obter o arquivo de image enviado pelo formulário
    print(f"{request.files}")
    if 'image' not in request.files:
        return render_template('erro.html')

    new_image = Image.open(request.files["image"].stream)
    new_image = new_image.resize((32, 32))

    new_image_array = np.array(new_image) / 255.0
    new_image_array = np.expand_dims(new_image_array, axis = 0)

    prediction = loaded_model.predict(new_image_array)

    predicted_classes = np.argmax(prediction)
    predicted_class_name = classes_names[predicted_classes]

    
    # Retornar o resultado em HTML
    return render_template('resultado.html', image=predicted_class_name)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="localhost")
