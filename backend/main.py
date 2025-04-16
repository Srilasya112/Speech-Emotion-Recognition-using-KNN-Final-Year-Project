from flask import Flask, request, jsonify
import pickle
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename
import tempfile
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature(file_name):
    """Extract audio features from a file"""
    try:
        X, sample_rate = librosa.load(file_name, sr=16000)

        if len(X) < 512:
            app.logger.warning(f"Audio file too short: {len(X)} samples")
            return None

        # Adjust n_fft dynamically
        n_fft = min(512, len(X) // 2)

        stft = np.abs(librosa.stft(X, n_fft=n_fft))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    except Exception as e:
        app.logger.error(f"Error processing {file_name}: {e}")
        return None

def predict_emotion(audio_file, model_path="emotion_classification_model.sav"):
    """Predict emotion from audio file using saved model"""
    try:
        # Load the saved model
        loaded_model = pickle.load(open(model_path, "rb"))
        app.logger.info(f"Model loaded successfully from {model_path}")
        
        # Extract features from the audio file
        app.logger.info(f"Extracting features from {audio_file}")
        test_feature = extract_feature(audio_file)
        
        if test_feature is not None:
            # Reshape feature for prediction
            test_feature = np.array(test_feature).reshape(1, -1)
            
            # Make prediction
            prediction = loaded_model.predict(test_feature)
            probabilities = loaded_model.predict_proba(test_feature)[0]
            
            # Get emotion labels
            emotion_labels = loaded_model.classes_
            
            # Create a dict of emotions and their probabilities
            emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_labels, probabilities)}
            
            return {
                "prediction": prediction[0],
                "probabilities": emotion_probs
            }
        else:
            app.logger.error("Feature extraction failed")
            return None
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get model path from form data or use default
        model_path = request.form.get('model_path', 'emotion_classification_model.sav')
        
        # Predict emotion
        result = predict_emotion(filepath, model_path)
        
        # Remove temporary file
        os.remove(filepath)
        
        if result:
            return jsonify({
                'success': True,
                'emotion': result['prediction'],
                'probabilities': result['probabilities']
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)