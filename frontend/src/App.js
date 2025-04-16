import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import UploadAudioPage from './components/Uploadaudio/uploadaudio';
import Result from './components/Results/result';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UploadAudioPage />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </Router>
  );
}

export default App;
