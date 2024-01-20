from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('model2.h5')
labels = ['animal', 'deforestation', 'fire', 'forest']

# Define a function to preprocess the image before feeding it to the model
# def preprocess_image(image):
#     image = image.resize((64, 64))
#     image_array = np.asarray(image)

#     normalized_image_array = (image_array.astype(np.float32) / 255.0)[np.newaxis, ...]
#     return normalized_image_array

def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.asarray(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    filename = secure_filename(image_file.filename)
    folder_path = os.path.join('static', filename.rsplit('.', 1)[0])
    os.makedirs(folder_path, exist_ok=True)
    image_path = os.path.join(folder_path, filename)
    image_file.save(image_path)

    image = Image.open(image_path)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)

    # animals_prob, deforestation_prob, fire_prob, forest_prob = predictions[0]
    output = np.argmax(predictions[0])  
    output = labels[output]

    response = {
        'prediction': output
    }

    print(response)
    





    # response = {
    #     'fire_probability': float(fire_prob),
    #     'deforestation_probability': float(deforestation_prob),
    #     'animals_probability': float(animals_prob), 
    #     'forest_probability': float(forest_prob),
    # }



    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
