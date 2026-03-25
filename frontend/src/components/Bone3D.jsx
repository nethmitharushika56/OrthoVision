import React, { useEffect, useState } from 'react';

const Bone3D = ({ modelUrl }) => {
  const activeModel = modelUrl || '';
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [activeModel]);

  return (
    <div className="viewer">
      {activeModel && !failed ? (
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
        <div className="viewer__empty">
          {failed ? 'Failed to load 3D model.' : 'No fracture detected. No 3D model auto-selected.'}
        </div>
      )}

      {activeModel && !failed && <div className="viewer__controls-hint">Drag to rotate • Scroll to zoom</div>}
    </div>
  );
};

export default Bone3D;
