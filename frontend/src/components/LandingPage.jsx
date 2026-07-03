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
          Advanced AI-powered fracture detection, guided review, and 3D visualization in one clinical workflow.
        </p>

        <div className="landing-features">
          <div className="landing-feature-card">
            <div className="landing-feature-icon">🔍</div>
            <h3>Fracture Detection</h3>
            <p>AI detects and classifies 10 types of bone fractures with confidence scoring.</p>
          </div>
          
          <div className="landing-feature-card">
            <div className="landing-feature-icon">📊</div>
            <h3>Real-time Analysis</h3>
            <p>Instant results with confidence scores, heatmaps, and a clear next-step summary.</p>
          </div>
          
          <div className="landing-feature-card">
            <div className="landing-feature-icon">🦴</div>
            <h3>3D Visualization</h3>
            <p>Interactive 3D bone models with AR-ready visualization for bedside review.</p>
          </div>

          <div className="landing-feature-card">
            <div className="landing-feature-icon">📝</div>
            <h3>Case Notes</h3>
            <p>Capture observations and keep every study organized for follow-up review.</p>
          </div>

          <div className="landing-feature-card">
            <div className="landing-feature-icon">🔒</div>
            <h3>Secure Access</h3>
            <p>Keep patient workflows private with authenticated access and saved sessions.</p>
          </div>
        </div>

        <section className="landing-education" aria-labelledby="education-heading">
          <div className="landing-education__header">
            <p className="opening-kicker">Learn as you analyze</p>
            <h2 id="education-heading" className="landing-education__title">Built-in teaching tools for every scan</h2>
            <p className="landing-education__subtitle">
              OrthoVision is designed to help students, residents, and clinicians understand why a fracture looks the way it does, not just what label it receives.
            </p>
          </div>

          <div className="landing-education__grid">
            <article className="landing-education__card">
              <div className="landing-feature-icon">🧠</div>
              <h3>Case library</h3>
              <p>Browse curated cases by bone, age, difficulty, and fracture type with X-ray, diagnosis, explanation, and learning points.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">✏️</div>
              <h3>Annotation practice</h3>
              <p>Draw fracture lines, mark abnormalities, and compare your notes against expert references.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🔎</div>
              <h3>Differential diagnosis</h3>
              <p>Understand why fracture, osteoarthritis, osteomyelitis, tumor, cyst, or normal variant are more or less likely.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">💬</div>
              <h3>Clinical correlation</h3>
              <p>Use age, mechanism, and symptoms alongside the image to sharpen interpretation and reasoning.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🏆</div>
              <h3>Progress tracking</h3>
              <p>Track completed cases, accuracy, weak bone regions, and improvement over time with badges and achievements.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🤖</div>
              <h3>AI tutor chat</h3>
              <p>Ask why a case is spiral rather than oblique, or which view best supports the finding, in educational language.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">📏</div>
              <h3>Measurement tools</h3>
              <p>Measure Cobb angle, bone length, fracture displacement, joint space, and calibration ruler references.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🖼️</div>
              <h3>Multi-view analysis</h3>
              <p>Combine AP, lateral, and oblique views so the model can make a more complete assessment.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">📚</div>
              <h3>Learning resources</h3>
              <p>Review anatomy, injury mechanism, demographics, treatment overview, and key exam points for each fracture.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🔬</div>
              <h3>Image quality checks</h3>
              <p>Teach exposure, blur, rotation, and missing anatomy problems so students know if an image is fit for interpretation.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">🧭</div>
              <h3>PACS-style viewer</h3>
              <p>Support zoom, pan, windowing, brightness, magnifier, and distance tools for professional viewing habits.</p>
            </article>

            <article className="landing-education__card">
              <div className="landing-feature-icon">📈</div>
              <h3>Research dashboard</h3>
              <p>Show confidence, precision, recall, confusion matrix results, and misclassified examples for academic review.</p>
            </article>
          </div>

          <div className="landing-workflow">
            <div className="landing-workflow__header">
              <p className="opening-kicker">Complete workflow</p>
              <h3 className="landing-workflow__title">A learning loop from upload to feedback</h3>
            </div>

            <ol className="landing-workflow__list">
              <li>Upload an X-ray.</li>
              <li>AI checks image quality and identifies the bone.</li>
              <li>AI localizes and classifies the fracture.</li>
              <li>Explainability tools highlight the region of interest.</li>
              <li>Students review anatomy labels, notes, and quiz questions.</li>
              <li>Progress, badges, and case history update automatically.</li>
            </ol>
          </div>
        </section>

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
