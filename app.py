from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
# Load the emotion detection model
model = load_model('model.h5')


# Function to preprocess image
# Function to preprocess image
def preprocess_image(image):
    img = load_img(image, target_size=(48, 48), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


app = Flask(__name__)


# Define routes
@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_page():
    if request.method == 'POST':
        img_file = request.files['image']
        temp_image_path = 'temp_image.jpg'
        img_file.save(temp_image_path)
        img_array = preprocess_image(temp_image_path)
        emotion_probabilities = model.predict(img_array)
        predicted_emotion_index = np.argmax(emotion_probabilities)
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = emotions[predicted_emotion_index]
        print(predicted_emotion)
        os.remove(temp_image_path)
        return render_template('result.html', emotion=predicted_emotion)
if __name__ == '__main__':
    app.run(debug=True)
