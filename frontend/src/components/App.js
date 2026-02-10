import React, { useState } from 'react';
import axios from 'axios';
import Bone3D from './Bone3D';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    const response = await axios.post(`${API_BASE}/analyze`, formData);
    setResult(response.data);
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center p-10 bg-gray-900 text-white min-h-screen">
      <h1 className="text-4xl font-bold mb-8">Orthovision AI</h1>
      
      <div className="grid grid-cols-2 gap-10 w-full max-w-6xl">
        {/* Left Side: Upload & 2D Result */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-4" />
          <button onClick={handleUpload} className="bg-blue-600 px-6 py-2 rounded">
            {loading ? "Analyzing..." : "Analyze Fracture"}
          </button>
          
          {result && (
            <div className="mt-6">
              <h2 className="text-xl">Result: {result.is_fractured ? "FRACTURE DETECTED" : "Normal"}</h2>
              <img src={`${API_BASE}/get_heatmap?t=${new Date().getTime()}`} 
                   alt="Analysis" className="mt-4 rounded border-2 border-red-500" />
            </div>
          )}
        </div>

        {/* Right Side: 3D AR Model */}
        <div className="bg-gray-800 p-6 rounded-lg h-[500px]">
          <h2 className="text-xl mb-4">AR 3D Visualization</h2>
          {result ? (
            <Bone3D result={result} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              Upload an image to see 3D Highlight
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;