import React from 'react';

const AnalysisView = ({ result }) => {
  if (!result) return null;

  const fracture_probability = result.fracture_probability || 0;
  const prediction = result.prediction || 'Unknown';
  const predicted_class = result.predicted_class || prediction;
  const confidence = result.confidence || fracture_probability;
  const image_url = result.image_url || '';
  const heatmap_url = result.heatmap_url || '';
  const bbox = result.bbox || null;
  const all_probabilities = result.all_probabilities || {};
  const fracture_type = result.fracture_type || null;
  const type_probabilities = result.type_probabilities || {};

  return (
    <div className="analysis">
      <div className="analysis__header">
        <h2 className="card__title">Analysis Results</h2>
      </div>

      <div className="analysis__content">
        {/* Image with Heatmap Overlay */}
        {image_url && (
          <div className="image-section">
            <h3 className="section-label">Original X-Ray</h3>
            <div className="image-container-with-bbox">
              <img src={image_url} alt="X-Ray" className="result-image" />
              {bbox && result.is_fractured && (
                <div 
                  className="bbox-overlay"
                  style={{
                    left: `${bbox.x * 100}%`,
                    top: `${bbox.y * 100}%`,
                    width: `${bbox.w * 100}%`,
                    height: `${bbox.h * 100}%`
                  }}
                >
                  <span className="bbox-label">Fracture</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Fracture Status */}
        <div className="result-item">
          <span className="result-label">Fractured:</span>
          <span className={`result-value ${result.is_fractured ? 'danger' : 'ok'}`}>
            {result.is_fractured ? 'Yes' : 'No'}
          </span>
        </div>

        {/* Fracture Type (when fractured) */}
        {result.is_fractured && (
          <div className="result-item">
            <span className="result-label">Fracture Type:</span>
            <span className="result-value">{fracture_type || predicted_class}</span>
          </div>
        )}

        {/* Confidence Score */}
        <div className="result-item">
          <span className="result-label">Prediction Confidence:</span>
          <span className="result-value">{(confidence * 100).toFixed(1)}%</span>
        </div>

        {/* Progress Bar */}
        <div className="progress-container">
          <div className="progress-label">Fracture Probability</div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${fracture_probability * 100}%` }}
            />
          </div>
          <div className="progress-value">{(fracture_probability * 100).toFixed(1)}%</div>
        </div>

        {/* Heatmap */}
        {heatmap_url && (
          <div className="heatmap-container">
            <h3 className="section-label">Attention Heatmap</h3>
            <img 
              src={heatmap_url} 
              alt="Attention Heatmap" 
              className="heatmap-image"
              crossOrigin="anonymous"
            />
          </div>
        )}

        {/* Bounding Box Info */}
        {bbox && result.is_fractured && (
          <div className="bbox-info">
            <h3 className="section-label">Fracture Location</h3>
            <div className="bbox-details">
              <div>X: {(bbox.x * 100).toFixed(1)}%</div>
              <div>Y: {(bbox.y * 100).toFixed(1)}%</div>
              <div>Width: {(bbox.w * 100).toFixed(1)}%</div>
              <div>Height: {(bbox.h * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}

        {/* All Class Probabilities */}
        {Object.keys(all_probabilities).length > 0 && (
          <div className="probabilities-section">
            <h3 className="section-label">All Predictions</h3>
            <div className="probability-list">
              {Object.entries(all_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, prob]) => (
                  <div key={className} className="probability-item">
                    <span className="class-name">{className}</span>
                    <div className="probability-bar-container">
                      <div 
                        className="probability-bar" 
                        style={{ width: `${prob * 100}%` }}
                      />
                      <span className="probability-text">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Fracture Type Probabilities */}
        {Object.keys(type_probabilities).length > 0 && (
          <div className="probabilities-section">
            <h3 className="section-label">Fracture Type Probabilities</h3>
            <div className="probability-list">
              {Object.entries(type_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([className, prob]) => (
                  <div key={className} className="probability-item">
                    <span className="class-name">{className}</span>
                    <div className="probability-bar-container">
                      <div
                        className="probability-bar"
                        style={{ width: `${prob * 100}%` }}
                      />
                      <span className="probability-text">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisView;
