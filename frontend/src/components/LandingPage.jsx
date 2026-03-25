import React from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="landing-content">
        <div className="landing-logo-container">
          <img 
            src="/ortho-vision-logo.jpeg" 
            alt="OrthoVision AI Logo" 
            className="landing-logo"
          />
        </div>
        
        <h1 className="landing-title">
          <span className="landing-gradient-text">OrthoVision AI</span>
        </h1>
        
        <p className="landing-subtitle">
          Advanced AI-Powered Bone Fracture Detection & AR Visualization
        </p>

        <div className="landing-features">
          <div className="landing-feature-card">
            <div className="landing-feature-icon">🔍</div>
            <h3>Fracture Detection</h3>
            <p>AI detects and classifies 10 types of bone fractures with 90%+ accuracy</p>
          </div>
          
          <div className="landing-feature-card">
            <div className="landing-feature-icon">📊</div>
            <h3>Real-time Analysis</h3>
            <p>Instant results with confidence scores and detailed heatmaps</p>
          </div>
          
          <div className="landing-feature-card">
            <div className="landing-feature-icon">🦴</div>
            <h3>3D Visualization</h3>
            <p>Interactive 3D bone models with AR capabilities</p>
          </div>
        </div>

        <div className="landing-stats">
          <div className="landing-stat-item">
            <div className="landing-stat-number">393</div>
            <div className="landing-stat-label">Neural Layers</div>
          </div>
          <div className="landing-stat-item">
            <div className="landing-stat-number">11.2M</div>
            <div className="landing-stat-label">Parameters</div>
          </div>
          <div className="landing-stat-item">
            <div className="landing-stat-number">90%+</div>
            <div className="landing-stat-label">Accuracy</div>
          </div>
          <div className="landing-stat-item">
            <div className="landing-stat-number">10</div>
            <div className="landing-stat-label">Fracture Types</div>
          </div>
        </div>

        <div className="landing-actions">
          <button 
            className="button button--primary landing-button--large landing-pulse-glow"
            onClick={() => navigate('/login')}
          >
            Get Started
          </button>
          <button 
            className="button button--secondary landing-button--large landing-button--secondary"
            onClick={() => navigate('/signup')}
          >
            Create Account
          </button>
        </div>

        <div className="landing-tech">
          <p className="landing-tech-label">Powered by</p>
          <div className="landing-tech-badges">
            <span className="landing-tech-badge">TensorFlow</span>
            <span className="landing-tech-badge">EfficientNetB3</span>
            <span className="landing-tech-badge">React</span>
            <span className="landing-tech-badge">Three.js</span>
          </div>
        </div>
      </div>

      <div className="landing-background">
        <div className="bg-orb orb-1"></div>
        <div className="bg-orb orb-2"></div>
        <div className="bg-orb orb-3"></div>
        <div className="bg-grid"></div>
      </div>
    </div>
  );
};

export default LandingPage;
