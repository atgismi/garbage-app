from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Konfigurasi
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Label
LABELS = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

LABEL_NAMES = {
    'battery': 'Baterai', 'biological': 'Sampah Organik/Biologis', 'brown-glass': 'Kaca Coklat',
    'cardboard': 'Kardus', 'clothes': 'Pakaian Bekas', 'green-glass': 'Kaca Hijau',
    'metal': 'Logam/Metal', 'paper': 'Kertas', 'plastic': 'Plastik', 'shoes': 'Sepatu Bekas',
    'trash': 'Sampah Umum', 'white-glass': 'Kaca Putih/Bening'
}

RECYCLABLE_INFO = {
    'battery': {'recyclable': False, 'category': 'Berbahaya', 'disposal': 'Tempat khusus baterai bekas'},
    'biological': {'recyclable': True, 'category': 'Organik', 'disposal': 'Kompos atau tempat sampah organik'},
    'brown-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca coklat'},
    'cardboard': {'recyclable': True, 'category': 'Kertas', 'disposal': 'Tempat sampah kertas/kardus'},
    'clothes': {'recyclable': True, 'category': 'Tekstil', 'disposal': 'Donasi atau daur ulang tekstil'},
    'green-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca hijau'},
    'metal': {'recyclable': True, 'category': 'Logam', 'disposal': 'Tempat sampah logam'},
    'paper': {'recyclable': True, 'category': 'Kertas', 'disposal': 'Tempat sampah kertas'},
    'plastic': {'recyclable': True, 'category': 'Plastik', 'disposal': 'Tempat sampah plastik'},
    'shoes': {'recyclable': False, 'category': 'Campuran', 'disposal': 'Tempat sampah umum atau donasi'},
    'trash': {'recyclable': False, 'category': 'Umum', 'disposal': 'Tempat sampah umum'},
    'white-glass': {'recyclable': True, 'category': 'Kaca', 'disposal': 'Tempat sampah kaca putih/bening'}
}

# Variabel global
model = None

# Load model TFLite
def load_model():
    global model
    try:
        model_path = "garbage_model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        model = interpreter
        logger.info("✅ TFLite model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading TFLite model: {str(e)}")
        return False

# Preprocessing
def preprocess_image(image, target_size=(150, 150)):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return None

# Prediction
def predict_image(image):
    global model
    try:
        input_data = preprocess_image(image)
        if input_data is None:
            return None

        if isinstance(model, tf.lite.Interpreter):
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], input_data)
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])[0]
        else:
            logger.error("Invalid model type")
            return None

        results = []
        for label, confidence in zip(LABELS, predictions):
            if confidence > 0.05:
                info = RECYCLABLE_INFO[label]
                results.append({
                    'class': label,
                    'class_name': LABEL_NAMES[label],
                    'confidence': float(confidence),
                    'recyclable': info['recyclable'],
                    'category': info['category'],
                    'disposal_method': info['disposal']
                })

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:3]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/categories')
def get_categories():
    data = []
    for label in LABELS:
        info = RECYCLABLE_INFO[label]
        data.append({
            'id': label,
            'name': LABEL_NAMES[label],
            'recyclable': info['recyclable'],
            'category': info['category'],
            'disposal_method': info['disposal']
        })
    return jsonify({'categories': data})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No filename'}), 400

        image = Image.open(io.BytesIO(file.read()))
        predictions = predict_image(image)
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Simpan file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.stream.seek(0)
        file.save(file_path)

        recyclable = [p for p in predictions if p['recyclable']]
        non_recyclable = [p for p in predictions if not p['recyclable']]

        return jsonify({
            'success': True,
            'predictions': predictions,
            'summary': {
                'total_detected': len(predictions),
                'recyclable_items': len(recyclable),
                'non_recyclable_items': len(non_recyclable),
                'highest_confidence': predictions[0]['confidence'] if predictions else 0
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File terlalu besar (maks 16MB)'}), 413

# Run
if __name__ == '__main__':
    logger.info("🚀 Starting Garbage Detection API...")
    if load_model():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Model gagal dimuat.")
