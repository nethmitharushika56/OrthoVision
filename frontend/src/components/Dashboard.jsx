import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Uploader from './Uploader.jsx';
import AnalysisView from './AnalysisView.jsx';
import Bone3D from './Bone3D.jsx';

function Dashboard({ user, onLogout }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleAnalyze = async (uploadedFile) => {
    setFile(uploadedFile);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('image', uploadedFile);
      
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      console.log('Analysis result:', data);
      
      const transformedResult = {
        prediction: data.predicted_class || (data.is_fractured ? 'Fracture' : 'Normal'),
        predicted_class: data.predicted_class,
        fracture_type: data.fracture_type,
        fracture_type_confidence: data.fracture_type_confidence,
        confidence: data.confidence,
        fracture_probability: data.fracture_probability || data.confidence,
        is_fractured: data.is_fractured,
        bone_part: data.bone_part,
        bbox: data.bbox,
        all_probabilities: data.all_probabilities || {},
        type_probabilities: data.type_probabilities || {},
        image_url: URL.createObjectURL(uploadedFile),
        heatmap_url: (data.heatmap_url
          ? `http://127.0.0.1:5000${data.heatmap_url}?t=${Date.now()}`
          : `http://127.0.0.1:5000/get_heatmap?t=${Date.now()}`)
      };
      
      setResult(transformedResult);
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Make sure the backend server is running on http://127.0.0.1:5000');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('orthovision_user');
    onLogout();
    navigate('/login');
  };

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <img src="/ortho-vision-logo.svg" alt="OrthoVision Logo" className="brand__logo" style={{width: '46px', height: 'auto'}} />
          <div>
            <h1 className="brand__name">OrthoVision AI</h1>
            <p className="brand__subtitle">Fracture Detection & Analysis</p>
          </div>
        </div>
        
        <div className="header-actions">
          <div className="user-menu">
            <div className="user-avatar">{user?.name?.charAt(0).toUpperCase() || 'U'}</div>
            <div className="user-info">
              <div className="user-name">{user?.name || 'User'}</div>
              <div className="user-email">{user?.email || ''}</div>
            </div>
            <button onClick={handleLogout} className="button button--logout">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
              Logout
            </button>
          </div>
        </div>
      </header>
      
      <main className="app__main">
        <section className="layout layout--full-width">
          {!result && (
            <div className="card card--center">
              <div className="card__header">
                <h2 className="card__title">Upload X-Ray Image</h2>
                <p className="card__description">Upload an X-ray image to detect and classify fractures</p>
              </div>
              <Uploader onAnalyze={handleAnalyze} loading={loading} />
            </div>
          )}

          {result && (
            <div className="results-row">
              <div className="card">
                <AnalysisView result={result} />
                
                <div style={{ padding: '20px', borderTop: '1px solid var(--border)' }}>
                  <button
                    onClick={() => setResult(null)}
                    className="button button--secondary"
                    style={{ width: '100%' }}
                  >
                    Analyze Another X-Ray
                  </button>
                </div>
              </div>

              <div className="card card--large">
                <div className="card__header">
                  <h2 className="card__title">3D Bone Model</h2>
                  {result.is_fractured && <span className="fracture-badge">Fracture Detected</span>}
                </div>
                <Bone3D 
                  modelUrl={'/models/handbone.glb'}
                  isFractured={result.is_fractured}
                  bbox={result.bbox}
                />
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default Dashboard;
