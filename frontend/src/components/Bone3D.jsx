import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

function Model({ url, isFractured, bbox }) {
  const { scene } = useGLTF(url);
  const modelRef = useRef();
  const [colorized, setColorized] = useState(false);

  useEffect(() => {
    if (scene && !colorized) {
      const targetColor = isFractured ? 0xff6b6b : 0x48b7ff;
      const emissiveColor = isFractured ? 0xff3b30 : 0x000000;
      const emissiveIntensity = isFractured ? 0.5 : 0;

      scene.traverse((node) => {
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
      setColorized(true);
    }
  }, [scene, isFractured, colorized]);

  return <primitive ref={modelRef} object={scene} scale={1.8} position={[0, 0, 0]} />;
}

const Bone3D = ({ modelUrl, isFractured = false, bbox = null }) => {
  const defaultModel = '/models/handbone.glb';
  const activeModel = modelUrl || defaultModel;

  return (
    <div className="viewer">
      <Suspense fallback={<div className="viewer__empty">Loading 3D Hand Model...</div>}>
        <Canvas
          camera={{ position: [0, 0, 8], fov: 45 }}
          gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
          style={{ width: '100%', height: '100%', background: 'transparent' }}
        >
          <ambientLight intensity={0.9} />
          <directionalLight position={[5, 5, 10]} intensity={1.5} />
          <pointLight position={[-10, -10, 5]} intensity={0.8} color={0x88ccff} />
          
          <Model url={activeModel} isFractured={isFractured} bbox={bbox} />
          
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

      <div className="viewer__controls-hint">
        Drag to rotate • Scroll to zoom
      </div>
    </div>
  );
};

export default Bone3D;
