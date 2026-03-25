import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Uploader from './Uploader.jsx';
import AnalysisView from './AnalysisView.jsx';
import Bone3D from './Bone3D.jsx';

function Dashboard({ user, onLogout }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (result) {
      saveAnalysisToHistory(result);
    }
  }, [result]);

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

  const saveAnalysisToHistory = (result) => {
    try {
      const analysis = {
        ...result,
        timestamp: new Date().toISOString(),
        patientId: localStorage.getItem('orthovision_patient_id') || 'Unknown',
        id: Date.now() // Add unique ID for each analysis
      };

      const savedAnalyses = localStorage.getItem('orthovision_analyses');
      const analyses = savedAnalyses ? JSON.parse(savedAnalyses) : [];
      analyses.push(analysis);
      localStorage.setItem('orthovision_analyses', JSON.stringify(analyses));
      
      console.log('✅ Analysis saved to history:', analysis);
      console.log('📊 Total analyses:', analyses.length);
    } catch (error) {
      console.error('❌ Error saving analysis to history:', error);
      alert('Warning: Analysis result was not saved to history. Please check your browser storage.');
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <img src="/ortho-vision-logo.jpeg" alt="OrthoVision Logo" className="brand__logo" style={{width: '46px', height: 'auto'}} />
          <div>
            <h1 className="brand__name brand__name--animated">OrthoVision AI</h1>
            <p className="brand__subtitle">Fracture Detection & Analysis</p>
          </div>
        </div>
        
        <div className="header-actions">
          <div className="nav-buttons">
            <button 
              onClick={() => navigate('/history')} 
              className="button button--nav"
              title="View Analysis History"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              History
            </button>
            <button 
              onClick={() => navigate('/settings')} 
              className="button button--nav"
              title="Settings & Profile"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="1"/>
                <path d="M12 1v6m0 6v6M4.22 4.22l4.24 4.24m5.08 5.08l4.24 4.24M1 12h6m6 0h6M4.22 19.78l4.24-4.24m5.08-5.08l4.24-4.24"/>
              </svg>
              Settings
            </button>
          </div>
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
                
                <div style={{ padding: '20px', borderTop: '1px solid var(--border)', display: 'flex', gap: '12px', flexDirection: 'column' }}>
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
