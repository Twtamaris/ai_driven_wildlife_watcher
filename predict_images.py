from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

model = load_model('model2.h5')

labels = ['animal', 'deforestation', 'fire', 'forest']

image_path = cv2.imread('3.jpg')
img = cv2.resize(image_path, (64, 64))
img = image.img_to_array(img)
img = img.astype("float32")/255.0
img = np.expand_dims(img, axis=0)

predict_output = model.predict(img)

print(labels[np.argmax(predict_output)])
