import React from 'react';
import { jsPDF } from 'jspdf';

const loadImageElement = (src, crossOrigin = 'anonymous') =>
  new Promise((resolve, reject) => {
    const img = new Image();
    if (crossOrigin) {
      img.crossOrigin = crossOrigin;
    }
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });

const buildAnnotatedImageDataUrl = async (src, bbox = null) => {
  const img = await loadImageElement(src, null);
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (bbox && Number.isFinite(bbox.x) && Number.isFinite(bbox.y) && Number.isFinite(bbox.w) && Number.isFinite(bbox.h)) {
    const x = bbox.x * canvas.width;
    const y = bbox.y * canvas.height;
    const w = bbox.w * canvas.width;
    const h = bbox.h * canvas.height;

    ctx.strokeStyle = 'rgba(255, 59, 48, 0.95)';
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
    ctx.fillStyle = 'rgba(255, 59, 48, 0.12)';
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();

    const label = 'Fracture';
    ctx.font = `${Math.max(16, Math.round(canvas.width / 30))}px Arial`;
    const textWidth = ctx.measureText(label).width;
    const padX = 12;
    const padY = 8;
    const labelX = x;
    const labelY = Math.max(0, y - 32);

    ctx.fillStyle = 'rgba(255, 59, 48, 0.95)';
    ctx.fillRect(labelX, labelY, textWidth + padX * 2, 30);
    ctx.fillStyle = 'white';
    ctx.fillText(label, labelX + padX, labelY + 22);
  }

  return canvas.toDataURL('image/jpeg', 0.95);
};

const buildImageDataUrl = async (src) => {
  const img = await loadImageElement(src, 'anonymous');
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.95);
};

const addPdfImageBlock = (pdf, title, dataUrl, yPosition, pageWidth, pageHeight) => {
  const margin = 15;
  const maxW = pageWidth - margin * 2;
  const imgProps = pdf.getImageProperties(dataUrl);
  const ratio = imgProps.height / imgProps.width;
  let drawW = maxW;
  let drawH = drawW * ratio;

  const maxBlockHeight = pageHeight - yPosition - 25;
  if (drawH > maxBlockHeight) {
    drawH = Math.max(40, maxBlockHeight);
    drawW = drawH / ratio;
  }

  pdf.setFontSize(11);
  pdf.setFont('helvetica', 'bold');
  pdf.setTextColor(0, 0, 0);
  pdf.text(title, margin, yPosition);
  yPosition += 5;

  if (yPosition + drawH > pageHeight - 15) {
    pdf.addPage();
    yPosition = 15;
    pdf.setFontSize(11);
    pdf.setFont('helvetica', 'bold');
    pdf.text(title, margin, yPosition);
    yPosition += 5;
  }

  pdf.addImage(dataUrl, 'JPEG', margin, yPosition, drawW, drawH, undefined, 'FAST');
  return yPosition + drawH + 8;
};

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
  const localization_hidden_reason = result.localization_hidden_reason || '';

  const handleDownloadReport = async () => {
    try {
      console.log('Starting PDF download...');
      const pdf = new jsPDF();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const pageWidth = pdf.internal.pageSize.getWidth();
      let yPosition = 15;

      // Header
      pdf.setFillColor(31, 107, 255);
      pdf.rect(0, 0, pageWidth, 30, 'F');

      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('ORTHOVISION AI', pageWidth / 2, 12, { align: 'center' });
      
      pdf.setFontSize(10);
      pdf.text('Fracture Detection Report', pageWidth / 2, 22, { align: 'center' });

      pdf.setTextColor(0, 0, 0);
      yPosition = 40;

      // Report Info
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Generated: ${new Date().toLocaleString()}`, 15, yPosition);
      yPosition += 6;

      // Status Badge
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      const statusText = result.is_fractured ? 'FRACTURE DETECTED' : 'NO FRACTURE';
      const statusColor = result.is_fractured ? [255, 59, 48] : [52, 199, 89];
      pdf.setFillColor(...statusColor);
      pdf.rect(15, yPosition, 80, 10, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.text(statusText, 55, yPosition + 6, { align: 'center' });
      yPosition += 20;

      // Results Section
      pdf.setTextColor(0, 0, 0);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('ANALYSIS RESULTS', 15, yPosition);
      yPosition += 8;

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      
      // Simple text-based results
      pdf.text(`Confidence: ${(confidence * 100).toFixed(2)}%`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Status: ${result.is_fractured ? 'Detected' : 'Not Detected'}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Prediction: ${prediction}`, 15, yPosition);
      yPosition += 6;
      
      if (result.is_fractured) {
        pdf.text(`Fracture Type: ${fracture_type || predicted_class}`, 15, yPosition);
        yPosition += 6;
      }

      yPosition += 8;

      // Include annotated 2D image in report.
      if (image_url) {
        try {
          const annotatedImage = await buildAnnotatedImageDataUrl(image_url, result.is_fractured ? bbox : null);
          yPosition = addPdfImageBlock(pdf, 'ANNOTATED 2D X-RAY', annotatedImage, yPosition, pageWidth, pageHeight);
        } catch (imgErr) {
          console.warn('Could not embed annotated X-ray image in PDF:', imgErr);
        }
      }

      // Include heatmap image when available for fractured cases.
      if (result.is_fractured && heatmap_url) {
        try {
          const heatmapImage = await buildImageDataUrl(heatmap_url);
          yPosition = addPdfImageBlock(pdf, 'ATTENTION HEATMAP', heatmapImage, yPosition, pageWidth, pageHeight);
        } catch (hmErr) {
          console.warn('Could not embed heatmap image in PDF:', hmErr);
        }
      }

      // All Classification Probabilities
      if (Object.keys(all_probabilities).length > 0) {
        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'bold');
        pdf.text('CLASSIFICATION PROBABILITIES', 15, yPosition);
        yPosition += 8;

        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'normal');
        
        Object.entries(all_probabilities)
          .sort(([, a], [, b]) => b - a)
          .forEach(([className, prob]) => {
            if (yPosition > pageHeight - 40) {
              pdf.addPage();
              yPosition = 15;
            }
            pdf.text(`${className}: ${(prob * 100).toFixed(2)}%`, 15, yPosition);
            yPosition += 5;
          });

        yPosition += 5;
      }

      // Fracture Type Probabilities
      if (Object.keys(type_probabilities).length > 0) {
        if (yPosition > pageHeight - 80) {
          pdf.addPage();
          yPosition = 15;
        }

        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'bold');
        pdf.text('FRACTURE TYPE PROBABILITIES', 15, yPosition);
        yPosition += 8;

        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'normal');
        
        Object.entries(type_probabilities)
          .sort(([, a], [, b]) => b - a)
          .forEach(([className, prob]) => {
            if (yPosition > pageHeight - 40) {
              pdf.addPage();
              yPosition = 15;
            }
            pdf.text(`${className}: ${(prob * 100).toFixed(2)}%`, 15, yPosition);
            yPosition += 5;
          });
      }

      // Disclaimer Page
      pdf.addPage();
      yPosition = 20;

      pdf.setFontSize(13);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(220, 38, 38);
      pdf.text('⚠  MEDICAL DISCLAIMER', 15, yPosition);
      yPosition += 10;

      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(0, 0, 0);

      const disclaimerText = `This report is generated by OrthoVision AI, an artificial intelligence system designed to assist in fracture detection and analysis.

IMPORTANT NOTES:
• This tool is NOT a substitute for professional medical diagnosis
• Results MUST be reviewed by qualified radiologists or specialists
• Accuracy depends on image quality, positioning, and resolution
• AI systems may produce false positives or false negatives
• Clinical decisions require professional medical evaluation
• In emergencies, seek immediate medical attention

For clinical confirmation, always consult appropriate medical professionals.`;

      const disclaimerLines = pdf.splitTextToSize(disclaimerText, pageWidth - 30);
      pdf.text(disclaimerLines, 15, yPosition);

      // Footer
      const totalPages = pdf.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setTextColor(128, 128, 128);
        pdf.text(
          `Page ${i} of ${totalPages}`,
          pageWidth / 2,
          pageHeight - 10,
          { align: 'center' }
        );
      }

      pdf.save(`orthovision_report_${Date.now()}.pdf`);
      console.log('PDF download completed successfully');
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error downloading PDF: ' + error.message);
    }
  };

  return (
    <div className="analysis">
      <div className="analysis__header">
        <h2 className="card__title">Analysis Results</h2>
        <button 
          onClick={handleDownloadReport}
          className="button button--primary button--download"
          title="Download analysis report as PDF"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download PDF
        </button>
      </div>

      <div className="analysis__content">
        {/* Image with Heatmap Overlay */}
        {image_url && (
          <div className="image-section">
            <h3 className="section-label">Original X-Ray</h3>
            <div className="image-container-with-bbox">
              <div className="image-bbox-frame">
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
        {result.is_fractured && (
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
        )}

        {/* Heatmap */}
        {result.is_fractured && heatmap_url && (
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

        {result.is_fractured && !heatmap_url && !bbox && (
          <div className="result-item">
            <span className="result-label">Localization:</span>
            <span className="result-value">
              {localization_hidden_reason || 'Heatmap/annotation unavailable for this image confidence level.'}
            </span>
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
        {result.is_fractured && Object.keys(type_probabilities).length > 0 && (
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
