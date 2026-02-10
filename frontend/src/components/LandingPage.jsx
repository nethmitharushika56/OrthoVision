import React from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="landing-content">
        <div className="landing-logo-container">
          <img 
            src="/ortho-vision-logo.svg" 
            alt="OrthoVision AI Logo" 
            className="landing-logo"
          />
        </div>
        
        <h1 className="landing-title">
          <span className="gradient-text">OrthoVision AI</span>
        </h1>
        
        <p className="landing-subtitle">
          Advanced AI-Powered Bone Fracture Detection & AR Visualization
        </p>

        <div className="landing-features">
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3>Fracture Detection</h3>
            <p>AI detects and classifies 10 types of bone fractures with 90%+ accuracy</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Real-time Analysis</h3>
            <p>Instant results with confidence scores and detailed heatmaps</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">🦴</div>
            <h3>3D Visualization</h3>
            <p>Interactive 3D bone models with AR capabilities</p>
          </div>
        </div>

        <div className="landing-stats">
          <div className="stat-item">
            <div className="stat-number">393</div>
            <div className="stat-label">Neural Layers</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">11.2M</div>
            <div className="stat-label">Parameters</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">90%+</div>
            <div className="stat-label">Accuracy</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">10</div>
            <div className="stat-label">Fracture Types</div>
          </div>
        </div>

        <div className="landing-actions">
          <button 
            className="button button--primary button--large pulse-glow"
            onClick={() => navigate('/login')}
          >
            Get Started
          </button>
          <button 
            className="button button--secondary button--large"
            onClick={() => navigate('/signup')}
          >
            Create Account
          </button>
        </div>

        <div className="landing-tech">
          <p className="tech-label">Powered by</p>
          <div className="tech-badges">
            <span className="tech-badge">TensorFlow</span>
            <span className="tech-badge">EfficientNetB3</span>
            <span className="tech-badge">React</span>
            <span className="tech-badge">Three.js</span>
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
