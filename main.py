from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Correct model path
MODEL_PATH = r"C:\Users\Suraj Vishwakarma\Desktop\MAJOR PROJECT\NewModel\saved_model.h5"

# Load the model properly
try:
    print(" Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

# Define labels
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/')
def home():
    return "🧠 Brain Tumor Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # Ensure correct input size
        img = img.convert('RGB')

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]) * 100)

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
