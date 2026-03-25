export default function BoneViewer() {
  return (
    <model-viewer
      src="/models/Comminuted_fracture_shaded.glb"
      auto-rotate
      camera-controls
      style={{ width: '100%', height: '500px' }}
    ></model-viewer>
  );
}