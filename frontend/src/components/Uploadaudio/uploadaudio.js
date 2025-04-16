import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import "./uploadAudio.css"; // Import your CSS file for styling
const Upload = () => {
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioRecorder, setAudioRecorder] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Validate file type
      const fileType = selectedFile.type;
      if (!fileType.includes('audio')) {
        setError('Please select an audio file');
        setFile(null);
        return;
      }
      
      setFile(selectedFile);
      setError('');
    }
  };

  // Start recording
  const startRecording = async () => {
    try {
      setError('');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.addEventListener('dataavailable', (event) => {
        audioChunks.push(event.data);
      });

      mediaRecorder.addEventListener('stop', () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
        setFile(audioFile);
        setRecording(false);
      });

      mediaRecorder.start();
      setAudioRecorder(mediaRecorder);
      setRecording(true);
    } catch (err) {
      setError('Microphone access denied or not available');
      console.error('Error accessing microphone:', err);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (audioRecorder && recording) {
      audioRecorder.stop();
      // Stop all audio tracks
      audioRecorder.stream.getTracks().forEach(track => track.stop());
    }
  };

  // Handle upload and prediction
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select an audio file or record audio first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze audio');
      }
      console.log("🎯 Emotion:", data.emotion);
      console.log("📊 Probabilities:", data.probabilities);

      // Navigate to results page with prediction data
      navigate('/result', { state: { result: data } });
    } catch (err) {
      setError(err.message || 'An error occurred during upload');
      console.error('Upload error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Audio Emotion Analyzer</h1>
      
      <form onSubmit={handleSubmit} className="form">
        {/* File Upload */}
        <div className="upload-box">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="file-input"
            id="audio-file"
          />
          <label htmlFor="audio-file" className="file-label">
            <svg className="upload-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <span className="file-text">
              {file ? file.name : 'Click to upload audio file'}
            </span>
            <span className="file-formats">
              Supported formats: WAV, MP3, OGG
            </span>
          </label>
        </div>
        
        {/* Recording Controls */}
        <div className="recording-section">
          <p className="record-text">Or record audio directly:</p>
          {!recording ? (
            <button
              type="button"
              onClick={startRecording}
              className="record-button"
              disabled={isLoading}
            >
              <svg className="mic-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
              </svg>
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecording}
              className="stop-button"
            >
              <span className="stop-icon"></span>
            </button>
          )}
          {recording && <p className="recording-indicator">Recording...</p>}
        </div>
        
        {/* Error Message */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        {/* Submit Button */}
        <button
          type="submit"
          className="submit-button"
          disabled={!file || isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Emotion'}
        </button>
      </form>
    </div>
  );
};

export default Upload;