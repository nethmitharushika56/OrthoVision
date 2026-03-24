import React, { Suspense, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

const DEFAULT_MODEL = '/models/handbone.glb';

class ViewerErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    console.error('3D Viewer error:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="viewer__empty">
          Failed to load 3D model. Showing fallback model.
        </div>
      );
    }
    return this.props.children;
  }
}

function Model({ url, isFractured }) {
  const { scene } = useGLTF(url);

  const fittedScene = useMemo(() => {
    const cloned = scene.clone(true);

    const box = new THREE.Box3().setFromObject(cloned);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    cloned.position.sub(center);

    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const targetSize = 4.0;
    const uniformScale = targetSize / maxDim;
    cloned.scale.setScalar(uniformScale);

    return cloned;
  }, [scene]);

  useEffect(() => {
    if (fittedScene) {
      const targetColor = isFractured ? 0xff6b6b : 0x48b7ff;
      const emissiveColor = isFractured ? 0xff3b30 : 0x000000;
      const emissiveIntensity = isFractured ? 0.35 : 0;

      fittedScene.traverse((node) => {
        if (node.isMesh) {
          node.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(targetColor),
            metalness: 0.6,
            roughness: 0.4,
            emissive: new THREE.Color(emissiveColor),
            emissiveIntensity: emissiveIntensity
          });
        }
      });
    }
  }, [fittedScene, isFractured]);

  return <primitive object={fittedScene} position={[0, 0, 0]} />;
}

function ModelWithFallback({ url, isFractured }) {
  try {
    return <Model url={url} isFractured={isFractured} />;
  } catch (error) {
    console.error('Primary model load failed, trying fallback:', error);
    return <Model url={DEFAULT_MODEL} isFractured={isFractured} />;
  }
}

const Bone3D = ({ modelUrl, isFractured = false }) => {
  const activeModel = modelUrl || DEFAULT_MODEL;

  return (
    <div className="viewer">
      <ViewerErrorBoundary>
        <Suspense fallback={<div className="viewer__empty">Loading 3D model...</div>}>
          <Canvas
            camera={{ position: [0, 0, 8], fov: 45 }}
            gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
            style={{ width: '100%', height: '100%', background: 'transparent' }}
          >
            <ambientLight intensity={0.9} />
            <directionalLight position={[5, 5, 10]} intensity={1.5} />
            <pointLight position={[-10, -10, 5]} intensity={0.8} color={0x88ccff} />

            <ModelWithFallback url={activeModel} isFractured={isFractured} />

            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              minDistance={3}
              maxDistance={15}
              autoRotate={false}
            />
          </Canvas>
        </Suspense>
      </ViewerErrorBoundary>

      <div className="viewer__controls-hint">
        Drag to rotate • Scroll to zoom
      </div>
    </div>
  );
};

useGLTF.preload(DEFAULT_MODEL);

export default Bone3D;
