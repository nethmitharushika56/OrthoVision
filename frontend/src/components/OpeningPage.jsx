import React from 'react';
import { useNavigate } from 'react-router-dom';

function OpeningPage() {
  const navigate = useNavigate();

  return (
    <div className="opening-page">
      <div className="opening-grid" aria-hidden="true" />
      <div className="opening-orb opening-orb--one" aria-hidden="true" />
      <div className="opening-orb opening-orb--two" aria-hidden="true" />

      <div className="opening-content">
        <div className="opening-logo-wrap">
          <img src="/ortho-vision-logo.jpeg" alt="OrthoVision logo" className="opening-logo" />
        </div>

        <p className="opening-kicker">AI Radiology Assistant</p>
        <h1 className="opening-title">OrthoVision AI</h1>
        <p className="opening-subtitle">
          Detect fractures faster with deep learning and 3D visual intelligence.
        </p>

        <div className="opening-loader" role="status" aria-label="Preparing experience">
          <span className="opening-loader__dot" />
          <span className="opening-loader__dot" />
          <span className="opening-loader__dot" />
        </div>

        <div className="opening-actions">
          <button className="button button--primary" onClick={() => navigate('/landing')}>
            Get Started
          </button>
          <button className="button button--ghost" onClick={() => navigate('/login')}>
            Go to Login
          </button>
        </div>
      </div>
    </div>
  );
}

export default OpeningPage;
