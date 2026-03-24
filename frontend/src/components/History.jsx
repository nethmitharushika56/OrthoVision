import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';

function History({ user }) {
  const [analyses, setAnalyses] = useState([]);
  const [filterType, setFilterType] = useState('all');
  const [loading, setLoading] = useState(true);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Load analyses from localStorage
    const savedAnalyses = localStorage.getItem('orthovision_analyses');
    if (savedAnalyses) {
      try {
        setAnalyses(JSON.parse(savedAnalyses));
      } catch (e) {
        console.error('Error loading analyses:', e);
      }
    }
    setLoading(false);
  }, []);

  const getStatusColor = (isFractured) => {
    return isFractured ? 'status-fractured' : 'status-normal';
  };

  const filteredAnalyses = analyses.filter(analysis => {
    if (filterType === 'all') return true;
    if (filterType === 'fractured') return analysis.is_fractured;
    if (filterType === 'normal') return !analysis.is_fractured;
    return true;
  });

  const handleDownloadReport = (analysis) => {
    try {
      console.log('Downloading individual report...');
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
      pdf.text('Individual Analysis Report', pageWidth / 2, 22, { align: 'center' });

      pdf.setTextColor(0, 0, 0);
      yPosition = 40;

      // Report Info
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Generated: ${new Date().toLocaleString()}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Analysis Date: ${new Date(analysis.timestamp).toLocaleString()}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Patient ID: ${analysis.patientId || 'N/A'}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Analyzed By: ${user?.name || 'Unknown'}`, 15, yPosition);
      yPosition += 12;

      // Status Badge
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      const statusText = analysis.is_fractured ? 'FRACTURE DETECTED' : 'NO FRACTURE';
      const statusColor = analysis.is_fractured ? [255, 59, 48] : [52, 199, 89];
      pdf.setFillColor(...statusColor);
      pdf.rect(15, yPosition, 80, 10, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.text(statusText, 55, yPosition + 6, { align: 'center' });
      yPosition += 20;

      // Results
      pdf.setTextColor(0, 0, 0);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('ANALYSIS RESULTS', 15, yPosition);
      yPosition += 8;

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Status: ${analysis.is_fractured ? 'Detected' : 'Not Detected'}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Confidence: ${(analysis.confidence * 100).toFixed(2)}%`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Prediction: ${analysis.prediction || 'N/A'}`, 15, yPosition);
      yPosition += 6;
      
      if (analysis.fracture_type) {
        pdf.text(`Fracture Type: ${analysis.fracture_type}`, 15, yPosition);
        yPosition += 6;
      }

      yPosition += 8;

      // Probabilities
      if (analysis.all_probabilities && Object.keys(analysis.all_probabilities).length > 0) {
        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'bold');
        pdf.text('CLASSIFICATION PROBABILITIES', 15, yPosition);
        yPosition += 8;

        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'normal');

        Object.entries(analysis.all_probabilities)
          .sort(([, a], [, b]) => b - a)
          .forEach(([key, value]) => {
            if (yPosition > pageHeight - 40) {
              pdf.addPage();
              yPosition = 15;
            }
            pdf.text(`${key}: ${(value * 100).toFixed(2)}%`, 15, yPosition);
            yPosition += 5;
          });

        yPosition += 5;
      }

      // Fracture Type Probabilities
      if (analysis.type_probabilities && Object.keys(analysis.type_probabilities).length > 0) {
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

        Object.entries(analysis.type_probabilities)
          .sort(([, a], [, b]) => b - a)
          .forEach(([key, value]) => {
            if (yPosition > pageHeight - 40) {
              pdf.addPage();
              yPosition = 15;
            }
            pdf.text(`${key}: ${(value * 100).toFixed(2)}%`, 15, yPosition);
            yPosition += 5;
          });
      }

      pdf.save(`orthovision_report_${new Date(analysis.timestamp).getTime()}.pdf`);
      console.log('Individual report PDF downloaded');
    } catch (error) {
      console.error('Error downloading individual report:', error);
      alert('Error downloading report: ' + error.message);
    }
  };

  const handleDownloadAllReports = () => {
    try {
      console.log('Downloading all reports...');
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
      pdf.text('Analysis History Report', pageWidth / 2, 22, { align: 'center' });

      pdf.setTextColor(0, 0, 0);
      yPosition = 40;

      // Summary Info
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Generated: ${new Date().toLocaleString()}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`User: ${user?.name || 'Unknown'}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Total Analyses: ${filteredAnalyses.length}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Fracture Cases: ${filteredAnalyses.filter(a => a.is_fractured).length}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Normal Cases: ${filteredAnalyses.filter(a => !a.is_fractured).length}`, 15, yPosition);
      yPosition += 12;

      // Summary List
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('ANALYSIS SUMMARY', 15, yPosition);
      yPosition += 8;

      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'normal');

      filteredAnalyses.forEach((analysis, index) => {
        if (yPosition > pageHeight - 40) {
          pdf.addPage();
          yPosition = 15;
        }

        const status = analysis.is_fractured ? 'FRACTURE' : 'NORMAL';
        const date = new Date(analysis.timestamp).toLocaleDateString();
        const confidence = `${(analysis.confidence * 100).toFixed(1)}%`;
        const type = analysis.fracture_type || analysis.prediction || 'N/A';

        pdf.text(`${index + 1}. [${date}] ${status} - ${type} (${confidence})`, 15, yPosition);
        yPosition += 5;
      });

      yPosition += 10;

      // Statistics
      if (yPosition > pageHeight - 60) {
        pdf.addPage();
        yPosition = 15;
      }

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('STATISTICS', 15, yPosition);
      yPosition += 8;

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');

      const totalAnalyses = filteredAnalyses.length;
      const fracturedCount = filteredAnalyses.filter(a => a.is_fractured).length;
      const normalCount = filteredAnalyses.filter(a => !a.is_fractured).length;
      const avgConfidence = filteredAnalyses.length > 0
        ? (filteredAnalyses.reduce((sum, a) => sum + a.confidence, 0) / filteredAnalyses.length * 100).toFixed(2)
        : 0;

      pdf.text(`Total Analyses: ${totalAnalyses}`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Fracture Detected: ${fracturedCount} (${totalAnalyses > 0 ? ((fracturedCount / totalAnalyses) * 100).toFixed(1) : 0}%)`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Normal Cases: ${normalCount} (${totalAnalyses > 0 ? ((normalCount / totalAnalyses) * 100).toFixed(1) : 0}%)`, 15, yPosition);
      yPosition += 6;
      pdf.text(`Average Confidence: ${avgConfidence}%`, 15, yPosition);

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

      pdf.save(`orthovision_history_${new Date().getTime()}.pdf`);
      console.log('All reports PDF downloaded');
    } catch (error) {
      console.error('Error downloading all reports:', error);
      alert('Error downloading reports: ' + error.message);
    }
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <button onClick={() => navigate('/dashboard')} className="back-button">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          Back to Dashboard
        </button>
        <h1>Analysis History</h1>
      </div>

      <div className="history-controls">
        <div className="filter-group">
          <button
            className={`filter-button ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All ({analyses.length})
          </button>
          <button
            className={`filter-button ${filterType === 'fractured' ? 'active' : ''}`}
            onClick={() => setFilterType('fractured')}
          >
            Fractured ({analyses.filter(a => a.is_fractured).length})
          </button>
          <button
            className={`filter-button ${filterType === 'normal' ? 'active' : ''}`}
            onClick={() => setFilterType('normal')}
          >
            Normal ({analyses.filter(a => !a.is_fractured).length})
          </button>
        </div>

        {analyses.length > 0 && (
          <button onClick={handleDownloadAllReports} className="button button--primary">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download All Reports
          </button>
        )}
      </div>

      <div className="history-content">
        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading your analysis history...</p>
          </div>
        ) : filteredAnalyses.length === 0 ? (
          <div className="empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
            </svg>
            <h3>No analysis records found</h3>
            <p>Your analysis history will appear here after you complete your first fracture analysis.</p>
          </div>
        ) : (
          <div className="history-list">
            {filteredAnalyses.map((analysis, index) => (
              <div key={index} className={`history-card ${getStatusColor(analysis.is_fractured)}`}>
                <div className="card-header">
                  <div className="status-badge">
                    {analysis.is_fractured ? '🔴 Fracture' : '✅ No Fracture'}
                  </div>
                  <span className="timestamp">
                    {new Date(analysis.timestamp).toLocaleString()}
                  </span>
                </div>

                <div className="card-content">
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="label">Fracture Type:</span>
                      <span className="value">{analysis.fracture_type || 'N/A'}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Bone Part:</span>
                      <span className="value">{analysis.bone_part || 'N/A'}</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Confidence:</span>
                      <span className="value">{(analysis.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="info-item">
                      <span className="label">Prediction:</span>
                      <span className="value">{analysis.prediction || 'N/A'}</span>
                    </div>
                  </div>
                </div>

                <div className="card-actions">
                  <button
                    onClick={() => setSelectedAnalysis(analysis)}
                    className="button button--secondary button--small"
                  >
                    View Details
                  </button>
                  <button
                    onClick={() => handleDownloadReport(analysis)}
                    className="button button--primary button--small"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedAnalysis && (
        <div className="modal-overlay" onClick={() => setSelectedAnalysis(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Analysis Details</h2>
              <button
                onClick={() => setSelectedAnalysis(null)}
                className="close-button"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <div className="detail-section">
                <h3>Basic Information</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Status:</span>
                    <span className="value">
                      {selectedAnalysis.is_fractured ? 'Fracture Detected' : 'No Fracture'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Date & Time:</span>
                    <span className="value">
                      {new Date(selectedAnalysis.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Fracture Type:</span>
                    <span className="value">{selectedAnalysis.fracture_type || 'N/A'}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Bone Part:</span>
                    <span className="value">{selectedAnalysis.bone_part || 'N/A'}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3>Probabilities</h3>
                <div className="probability-chart">
                  {Object.entries(selectedAnalysis.all_probabilities || {}).map(([key, value]) => (
                    <div key={key} className="probability-bar">
                      <div className="bar-label">{key}</div>
                      <div className="bar-container">
                        <div
                          className="bar-fill"
                          style={{ width: `${value * 100}%` }}
                        ></div>
                      </div>
                      <div className="bar-value">{(value * 100).toFixed(2)}%</div>
                    </div>
                  ))}
                </div>
              </div>

              {selectedAnalysis.type_probabilities && Object.keys(selectedAnalysis.type_probabilities).length > 0 && (
                <div className="detail-section">
                  <h3>Fracture Type Probabilities</h3>
                  <div className="probability-chart">
                    {Object.entries(selectedAnalysis.type_probabilities).map(([key, value]) => (
                      <div key={key} className="probability-bar">
                        <div className="bar-label">{key}</div>
                        <div className="bar-container">
                          <div
                            className="bar-fill"
                            style={{ width: `${value * 100}%` }}
                          ></div>
                        </div>
                        <div className="bar-value">{(value * 100).toFixed(2)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                onClick={() => handleDownloadReport(selectedAnalysis)}
                className="button button--primary"
              >
                Download Report
              </button>
              <button
                onClick={() => setSelectedAnalysis(null)}
                className="button button--secondary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default History;

