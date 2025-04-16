# predict.py - Load model and predict audio emotion

import pickle
import librosa
import numpy as np
import sys

def extract_feature(file_name):
    """Extract audio features from a file"""
    try:
        X, sample_rate = librosa.load(file_name, sr=16000)

        if len(X) < 512:
            print(f"⚠️ Audio file too short: {len(X)} samples")
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
        print(f"❌ Error processing {file_name}: {e}")
        return None

def predict_emotion(model_path, audio_file):
    """Predict emotion from audio file using saved model"""
    try:
        # Load the saved model
        loaded_model = pickle.load(open(model_path, "rb"))
        print(f"✅ Model loaded successfully from {model_path}")
        
        # Extract features from the audio file
        print(f"📊 Extracting features from {audio_file}")
        test_feature = extract_feature(audio_file)
        
        if test_feature is not None:
            # Reshape feature for prediction
            test_feature = np.array(test_feature).reshape(1, -1)
            
            # Make prediction
            prediction = loaded_model.predict(test_feature)
            print(f"🎯 Predicted emotion: {prediction[0]}")
            return prediction[0]
        else:
            print("❌ Feature extraction failed")
            return None
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Default paths
    default_model_path = "emotion_classification_model.sav"
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_path> [model_path]")
        print("Example: python predict.py test_audio.wav")
        sys.exit(1)
    
    # Get audio file path from command line argument
    audio_file = sys.argv[1]
    
    # Get model path from command line argument or use default
    model_path = sys.argv[2] if len(sys.argv) > 2 else default_model_path
    
    # Predict emotion
    emotion = predict_emotion(model_path, audio_file)
    
    if emotion:
        print(f"The audio file {audio_file} expresses the emotion: {emotion}")