from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pytorch_predict import PyTorchPredictor
import os
from werkzeug.utils import secure_filename

# Initialize Flask
app = Flask(__name__)
CORS(app, origins=["https://garbageclassification.insaash.space"])




UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

predictor = PyTorchPredictor(model_path='models/pytorch_garbage_model.pth')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve frontend page
@app.route('/')
def index():
    return jsonify({"status": "Backend running", "message": "Use /predict endpoint with POST"})


# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predicted_class, confidence, probabilities = predictor.predict(file_path)
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
