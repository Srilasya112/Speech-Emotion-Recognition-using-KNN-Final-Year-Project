# model.py - Train and save the audio emotion classification model

import glob
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def extract_feature(file_name):
    """Extract audio features from a file"""
    try:
        X, sample_rate = librosa.load(file_name, sr=16000)

        if len(X) < 512:
            print(f"⚠️ Skipping {file_name} (Too Short: {len(X)} samples)")
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

def parse_audio_files(path):
    """Parse all audio files in the given path"""
    features, labels = [], []

    for fn in glob.glob(path):
        print(f"🔍 Processing file: {fn}")
        feature = extract_feature(fn)

        if feature is not None:
            features.append(feature)

            # Extract last word before ".wav" as the label
            label = fn.split("/")[-1].split("_")[-1].replace(".wav", "")
            labels.append(label)

    return np.array(features), np.array(labels)

def train_model(audio_path):
    """Train KNN model on audio files"""
    # Load and extract features
    tr_features, tr_labels = parse_audio_files(audio_path)

    # Check if data is available
    if len(tr_features) == 0:
        print("🚨 No valid audio files processed. Exiting...")
        return None, None, None, None

    # Convert features & labels to correct types
    X = np.array(tr_features, dtype=np.float32)
    y = np.array(tr_labels, dtype=str)

    # Split dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train K-Nearest Neighbors (KNN) Classifier
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train)

    # Save the trained model
    model_filename = 'emotion_classification_model.sav'
    pickle.dump(neigh, open(model_filename, 'wb'), protocol=2)
    print('✅ Model Saved Successfully!')

    # Print Model Score
    print('🔹 Training Score:', neigh.score(X_train, y_train))
    print('🔹 Testing Score:', neigh.score(X_test, y_test))
    
    return neigh, X_train, X_test, y_train, y_test

def plot_learning_curve(model, X, y):
    """Plot learning curve to show model performance"""
    # Set seed for reproducibility
    seed = 7

    # Add shuffle=True to KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, n_jobs=-1, cv=kfold,
        train_sizes=np.linspace(0.1, 1.0, 5), verbose=1
    )

    # Compute mean and standard deviation
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.figure()
    plt.title("KNN Model Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Fill areas to show standard deviation
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Plot mean scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.ylim(-0.1, 1.1)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix for test results"""
    # Generate predictions on the test set
    test_predicted = model.predict(X_test)  # Model predictions
    test_true = y_test  # Actual labels

    # Compute confusion matrix
    matrix = confusion_matrix(test_true, test_predicted)

    # Get unique class labels and sort them
    classes = sorted(set(y_test))  # Ensure correct order

    # Convert to DataFrame for heatmap
    df = pd.DataFrame(matrix, columns=classes, index=classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.title('Test Accuracy using KNN')
    sn.heatmap(df, annot=True, cmap="Blues", fmt='d')

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

if __name__ == "__main__":
    # Path to your audio files (update this to your actual file path)
    audio_path = r"data\*.wav"  
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(audio_path)
    
    if model is not None:
        # Plot learning curve
        plot_learning_curve(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
        
        # Plot confusion matrix
        plot_confusion_matrix(model, X_test, y_test)