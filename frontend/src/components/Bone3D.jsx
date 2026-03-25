import React, { useState } from 'react';

const DEFAULT_MODEL = '/models/handbone.glb';

const Bone3D = ({ modelUrl }) => {
  const activeModel = modelUrl || DEFAULT_MODEL;
  const [failed, setFailed] = useState(false);

  return (
    <div className="viewer">
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

      <div className="viewer__controls-hint">Drag to rotate • Scroll to zoom</div>
    </div>
  );
};

export default Bone3D;
