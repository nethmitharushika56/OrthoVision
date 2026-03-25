import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Uploader from './Uploader.jsx';
import AnalysisView from './AnalysisView.jsx';
import Bone3D from './Bone3D.jsx';

const DEFAULT_MODEL_URL = '/models/handbone.glb';
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

const apiUrl = (path = '') => {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${normalizedPath}` : normalizedPath;
};

const withTimestamp = (url) => `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;

// Add additional mappings here as you add new model files under /public/models.
const FRACTURE_MODEL_MAP = {
  'avulsion fracture': '/models/Avulsion_fracture_shaded.glb',
  'avulsionfracture': '/models/Avulsion_fracture_shaded.glb',
  avulsion: '/models/Avulsion_fracture_shaded.glb',
  'comminuted fracture': '/models/Comminuted_fracture_shaded.glb',
  'comminutedfracture': '/models/Comminuted_fracture_shaded.glb',
  'communited fracture': '/models/Comminuted_fracture_shaded.glb',
  'communitedfracture': '/models/Comminuted_fracture_shaded.glb',
  comminuted: '/models/Comminuted_fracture_shaded.glb',
  'fracture dislocation': '/models/Fracture_dislocation_shaded.glb',
  'fracturedislocation': '/models/Fracture_dislocation_shaded.glb',
};

const BUILT_IN_MODELS = [
  '/models/handbone.glb',
  '/models/Avulsion_fracture_shaded.glb',
  '/models/Comminuted_fracture_shaded.glb',
  '/models/Fracture_dislocation_shaded.glb',
];

const normalizeKey = (value = '') =>
  value
    .toString()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

const toModelLabel = (modelUrl) => {
  const fileName = modelUrl.split('/').pop() || modelUrl;
  return fileName.replace(/\.glb$/i, '');
};

const getFinalFractureType = (analysisResult) => {
  if (!analysisResult) {
    return '';
  }
  return String(analysisResult.fracture_type || '').trim();
};

const resolveAutoModelUrl = (analysisResult, models) => {
  if (!analysisResult?.is_fractured) {
    return DEFAULT_MODEL_URL;
  }

  const rawFracture = getFinalFractureType(analysisResult);
  if (!rawFracture) {
    return DEFAULT_MODEL_URL;
  }

  const fractureKey = rawFracture.toString().trim().toLowerCase();
  const normalizedFracture = normalizeKey(rawFracture);

  const mapped = FRACTURE_MODEL_MAP[fractureKey] || FRACTURE_MODEL_MAP[normalizedFracture];
  if (mapped) {
    return mapped;
  }

  const fuzzyMatched = (models || []).find((modelUrl) => {
    const modelNameNormalized = normalizeKey(toModelLabel(modelUrl));
    return modelNameNormalized.includes(normalizedFracture) || normalizedFracture.includes(modelNameNormalized);
  });

  return fuzzyMatched || DEFAULT_MODEL_URL;
};

function Dashboard({ user, onLogout }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState(BUILT_IN_MODELS);
  const [selectedModelUrl, setSelectedModelUrl] = useState('auto');
  const navigate = useNavigate();

  useEffect(() => {
    if (result) {
      saveAnalysisToHistory(result);
    }
  }, [result]);

  useEffect(() => {
    let active = true;

    const loadModels = async () => {
      try {
        const response = await fetch('/models/models.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('Model manifest not found');
        }

        const files = await response.json();
        if (!Array.isArray(files)) {
          throw new Error('Invalid models.json format');
        }

        const glbUrls = files
          .filter((name) => typeof name === 'string' && name.toLowerCase().endsWith('.glb'))
          .map((name) => `/models/${name}`);

        if (active && glbUrls.length > 0) {
          setAvailableModels(Array.from(new Set([...BUILT_IN_MODELS, ...glbUrls])));
        }
      } catch (error) {
        console.warn('Using built-in model list:', error.message);
      }
    };

    loadModels();

    return () => {
      active = false;
    };
  }, []);

  const handleAnalyze = async (uploadedFile) => {
    setFile(uploadedFile);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('image', uploadedFile);
      
      const response = await fetch(apiUrl('/analyze'), {
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
        heatmap_url: withTimestamp(apiUrl(data.heatmap_url || '/get_heatmap')),
      };
      
      setResult(transformedResult);
      // Keep auto mode synced to the latest analysis fracture type.
      setSelectedModelUrl('auto');
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Make sure the backend server is running.');
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

  const getModelUrlForResult = (analysisResult) => {
    if (selectedModelUrl !== 'auto') {
      return selectedModelUrl;
    }
    return resolveAutoModelUrl(analysisResult, availableModels);
  };

  const autoMatchedModelUrl = resolveAutoModelUrl(result, availableModels);
  const autoSourceFractureType = getFinalFractureType(result) || 'N/A';

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
                  <div className="model-picker-row">
                    <label className="model-picker-label" htmlFor="model-picker">Model</label>
                    <select
                      id="model-picker"
                      className="model-picker-select"
                      value={selectedModelUrl}
                      onChange={(e) => setSelectedModelUrl(e.target.value)}
                    >
                      <option value="auto">
                        Auto (from output type: {autoSourceFractureType}) to {toModelLabel(autoMatchedModelUrl)}
                      </option>
                      {availableModels.map((modelUrl) => (
                        <option key={modelUrl} value={modelUrl}>
                          {toModelLabel(modelUrl)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <Bone3D 
                  modelUrl={getModelUrlForResult(result)}
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
