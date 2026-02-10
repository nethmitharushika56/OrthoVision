import React, { useState } from 'react';

const Uploader = ({ onAnalyze, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedFile) {
      alert('Please select an image first!');
      return;
    }
    onAnalyze(selectedFile);
  };

  return (
    <div className="upload">
      {preview && (
        <div className="upload__preview">
          <img src={preview} alt="Preview" className="upload__image" />
        </div>
      )}

      <div className="upload__input-wrapper">
        <label className="upload__label">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={loading}
            className="upload__input"
          />
          <span className="upload__text">
            {preview ? 'Choose Another Image' : 'Choose Image'}
          </span>
        </label>
      </div>

      <button
        onClick={handleAnalyzeClick}
        disabled={!selectedFile || loading}
        className="button button--primary"
      >
        {loading ? 'Analyzing...' : 'Analyze X-Ray'}
      </button>
    </div>
  );
};

export default Uploader;
