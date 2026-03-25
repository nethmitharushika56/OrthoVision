import React, { useMemo, useState } from 'react';

const DEFAULT_MODEL = '/models/handbone.glb';

const formatBbox = (bbox) => {
  if (!bbox) {
    return 'N/A';
  }
  if (Array.isArray(bbox) && bbox.length >= 4) {
    return `[${bbox.slice(0, 4).join(', ')}]`;
  }
  if (typeof bbox === 'object') {
    try {
      return JSON.stringify(bbox);
    } catch (e) {
      return 'N/A';
    }
  }
  return String(bbox);
};

const Bone3D = ({ modelUrl, isFractured = false, bbox = null }) => {
  const activeModel = modelUrl || DEFAULT_MODEL;
  const [failed, setFailed] = useState(false);

  const bboxLabel = useMemo(() => formatBbox(bbox), [bbox]);

  return (
    <div className={`viewer ${isFractured ? 'viewer--fractured' : ''}`}>
      {!failed ? (
        <model-viewer
          src={activeModel}
          camera-controls
          auto-rotate
          auto-rotate-delay="0"
          shadow-intensity="1"
          exposure="1"
          interaction-prompt="none"
          className="viewer__model"
          style={{ width: '100%', height: '100%', minHeight: '450px' }}
          onError={() => setFailed(true)}
        ></model-viewer>
      ) : (
        <div className="viewer__empty">Failed to load 3D model.</div>
      )}

      {isFractured && (
        <>
          <div className="viewer__fracture-box" aria-hidden="true"></div>
          <div className="viewer__fracture-info">Fracture Focus Box: {bboxLabel}</div>
        </>
      )}

      <div className="viewer__controls-hint">Drag to rotate • Scroll to zoom</div>
    </div>
  );
};

export default Bone3D;
