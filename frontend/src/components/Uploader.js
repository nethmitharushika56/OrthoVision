import React, { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

const Uploader = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file)); // Create local URL for preview
    }
  };

  const uploadToBackend = async () => {
    if (!selectedFile) return alert("Please select an image first!");

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      onUploadSuccess(data, preview); // Pass result and image back to App.js
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div className="flex flex-col items-center p-4 border-2 border-dashed border-gray-600 rounded-lg bg-gray-800">
      <h3 className="text-lg font-semibold mb-4">Upload X-Ray Scan</h3>
      
      {preview && (
        <img src={preview} alt="Preview" className="w-48 h-48 object-cover mb-4 rounded-md border" />
      )}

      <input 
        type="file" 
        onChange={handleFileChange} 
        className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 mb-4"
      />

      <button 
        onClick={uploadToBackend}
        className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition duration-200"
      >
        Analyze Scan
      </button>
    </div>
  );
};

export default Uploader;