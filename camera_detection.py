import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your trained model
model = load_model('model2.h5')

# Define the labels for your classes
labels = ['animal', 'deforestation', 'fire', 'forest']

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    

    # Preprocess the frame for your model
    img = cv2.resize(frame, (64, 64))
    img = image.img_to_array(img)
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = labels[class_index]

    # Display the results on the frame
    cv2.putText(frame, f'Class: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
