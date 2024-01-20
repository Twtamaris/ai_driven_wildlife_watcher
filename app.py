# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# import os
# import cv2

# app = Flask(__name__)

# # Load your pre-trained model
# model = tf.keras.models.load_model('model2.h5')
# labels = ['animal', 'deforestation', 'fire', 'forest']

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (64, 64))
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     image_file = request.files['image']

#     filename = secure_filename(image_file.filename)
#     folder_path = os.path.join('static', filename.rsplit('.', 1)[0])
#     os.makedirs(folder_path, exist_ok=True)
#     image_path = os.path.join(folder_path, filename)
#     image_file.save(image_path)

#     processed_image = preprocess_image(image_path)

#     predictions = model.predict(processed_image)

#     output = np.argmax(predictions[0])  
#     output = labels[output]

#     response = {
#         'prediction': output
#     }

#     print(response)

#     return jsonify(response), 200

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5002)


from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import cv2
import io

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('model2.h5')
labels = ['animal', 'deforestation', 'fire', 'forest']

def preprocess_image(image_stream):
    # Convert the binary data to a numpy array
    npimg = np.fromstring(image_stream.read(), np.uint8)

    # Convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    processed_image = preprocess_image(image_file)

    predictions = model.predict(processed_image)

    output = np.argmax(predictions[0])  
    output = labels[output]

    response = {
        'prediction': output
    }

    print(response)

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)



