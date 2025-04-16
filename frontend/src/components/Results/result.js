import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './result.css';

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  
  // Color mapping for different emotions
  const emotionColors = {
    happy: 'emotion-happy',
    sad: 'emotion-sad',
    angry: 'emotion-angry',
    fear: 'emotion-fear',
    neutral: 'emotion-neutral',
    disgust: 'emotion-disgust',
    surprise: 'emotion-surprise',
    // Add more emotions as needed
  };
  
  // Emoji mapping for different emotions
  const emotionEmojis = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    fear: '😨',
    neutral: '😐',
    disgust: '🤢',
    surprise: '😲',
    // Add more emotions as needed
  };

  useEffect(() => {
    // Check if we have result data from the navigation state
    if (location.state && location.state.result) {
      setResult(location.state.result);
    } else {
      setError('No analysis results found. Please upload an audio file first.');
    }
  }, [location]);

  // Navigate back to upload
  const handleBack = () => {
    navigate('/');
  };

  // Get color class for emotion
  const getEmotionColor = (emotion) => {
    return emotionColors[emotion.toLowerCase()] || 'emotion-neutral';
  };

  // Get emoji for emotion
  const getEmotionEmoji = (emotion) => {
    return emotionEmojis[emotion.toLowerCase()] || '🤔';
  };

  // Calculate percentage for progress bar
  const calculatePercentage = (value) => {
    return (value * 100).toFixed(1);
  };

  if (error) {
    return (
      <div className="result-container">
        <div className="error-message">
          {error}
        </div>
        <button
          onClick={handleBack}
          className="back-link"
        >
          <svg className="back-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          Back to Upload
        </button>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-container">
        <div className="loader-container">
          <div className="loader"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="result-container">
      <h1 className="result-title">Emotion Analysis Result</h1>
      
      {/* Primary Emotion */}
      <div className="primary-emotion">
        <div className="emotion-emoji">
          {getEmotionEmoji(result.emotion)}
        </div>
        <h2 className="emotion-name">
          {result.emotion}
        </h2>
        <p className="emotion-label">Primary detected emotion</p>
      </div>
      
      {/* Emotion Probabilities */}
      {result.probabilities && (
        <div className="confidence-section">
          <h3 className="section-title">Confidence Levels</h3>
          <div className="confidence-bars">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([emotion, probability]) => (
                <div key={emotion} className="confidence-item">
                  <div className="confidence-header">
                    <span className="emotion-label">{emotion}</span>
                    <span className="percentage">{calculatePercentage(probability)}%</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div 
                      className={`progress-bar-fill ${getEmotionColor(emotion)}`} 
                      style={{ width: `${calculatePercentage(probability)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
      
      {/* What this means */}
      <div className="info-panel">
        <h3 className="info-title">What this means</h3>
        <p className="info-text">
          The audio appears to convey <strong className="emotion-highlight">{result.emotion}</strong> emotion. 
          This analysis is based on acoustic features extracted from your audio recording, 
          including tone, pitch variations, and spectral characteristics.
        </p>
      </div>
      
      {/* Back Button */}
      <button
        onClick={handleBack}
        className="analyze-button"
      >
        Analyze Another Audio
      </button>
    </div>
  );
};

export default Result;