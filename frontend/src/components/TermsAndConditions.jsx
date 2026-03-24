import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function TermsAndConditions() {
  const [activeSection, setActiveSection] = useState('terms');
  const navigate = useNavigate();

  return (
    <div className="legal-container">
      <div className="legal-header">
        <button onClick={() => navigate(-1)} className="back-button">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          Back
        </button>
        <h1>Legal Documents</h1>
      </div>

      <div className="legal-tabs">
        <button
          className={`tab-button ${activeSection === 'terms' ? 'active' : ''}`}
          onClick={() => setActiveSection('terms')}
        >
          Terms of Service
        </button>
        <button
          className={`tab-button ${activeSection === 'privacy' ? 'active' : ''}`}
          onClick={() => setActiveSection('privacy')}
        >
          Privacy Policy
        </button>
        <button
          className={`tab-button ${activeSection === 'disclaimer' ? 'active' : ''}`}
          onClick={() => setActiveSection('disclaimer')}
        >
          Medical Disclaimer
        </button>
      </div>

      <div className="legal-content">
        {activeSection === 'terms' && (
          <div className="legal-section">
            <h2>Terms of Service</h2>

            <div className="section">
              <h3>1. Acceptance of Terms</h3>
              <p>
                By accessing and using OrthoVision AI, you accept and agree to be bound by the terms and provision of this agreement.
                If you do not agree to abide by the above, please do not use this service.
              </p>
            </div>

            <div className="section">
              <h3>2. Use License</h3>
              <p>
                Permission is granted to temporarily download one copy of the materials (information or software) on OrthoVision AI's
                web and mobile applications for personal, non-commercial transitory viewing only. This is the grant of a license, not a
                transfer of title, and under this license you may not:
              </p>
              <ul>
                <li>Modifying or copying the materials</li>
                <li>Using the materials for any commercial purpose or for any public display</li>
                <li>Attempting to decompile or reverse engineer any software contained on OrthoVision AI</li>
                <li>Removing any copyright or other proprietary notations from the materials</li>
                <li>Transferring the materials to another person or &quot;mirroring&quot; the materials on any other server</li>
              </ul>
            </div>

            <div className="section">
              <h3>3. Disclaimer</h3>
              <p>
                The materials on OrthoVision AI's web and mobile applications are provided on an 'as is' basis. OrthoVision AI makes no
                warranties, expressed or implied, and hereby disclaims and negates all other warranties including, without limitation,
                implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of
                intellectual property or other violation of rights.
              </p>
            </div>

            <div className="section">
              <h3>4. Limitations</h3>
              <p>
                In no event shall OrthoVision AI or its suppliers be liable for any damages (including, without limitation, damages for
                loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on
                OrthoVision AI's web and mobile applications, even if OrthoVision AI or an authorized representative has been notified
                orally or in writing of the possibility of such damage.
              </p>
            </div>

            <div className="section">
              <h3>5. User Responsibilities</h3>
              <p>
                Users are responsible for:
              </p>
              <ul>
                <li>Maintaining the confidentiality of their account information</li>
                <li>All activities that occur under their account</li>
                <li>Notifying OrthoVision AI immediately of any unauthorized uses of their account</li>
                <li>Ensuring that all information provided is accurate and current</li>
              </ul>
            </div>

            <div className="section">
              <h3>6. Modifications to Terms</h3>
              <p>
                OrthoVision AI may revise these terms of service for our web and mobile applications at any time without notice. By
                using this service, you are agreeing to be bound by the then current version of these terms of service.
              </p>
            </div>
          </div>
        )}

        {activeSection === 'privacy' && (
          <div className="legal-section">
            <h2>Privacy Policy</h2>

            <div className="section">
              <h3>1. Information Collection</h3>
              <p>
                OrthoVision AI collects information you provide directly to us, such as when you create an account, upload images,
                or contact us for support. This may include:
              </p>
              <ul>
                <li>Name and contact information</li>
                <li>Email address</li>
                <li>Medical images and related data</li>
                <li>Usage patterns and analytics</li>
              </ul>
            </div>

            <div className="section">
              <h3>2. Use of Information</h3>
              <p>
                We use the information we collect to:
              </p>
              <ul>
                <li>Provide, maintain, and improve our services</li>
                <li>Process and analyze medical images for fracture detection</li>
                <li>Communicate with you about updates and changes</li>
                <li>Monitor and enhance the performance of our AI models</li>
                <li>Comply with legal obligations</li>
              </ul>
            </div>

            <div className="section">
              <h3>3. Data Security</h3>
              <p>
                We implement appropriate technical and organizational measures to protect your personal data against unauthorized access,
                alteration, disclosure, or destruction. However, no method of transmission over the internet or electronic storage is
                completely secure.
              </p>
            </div>

            <div className="section">
              <h3>4. Data Retention</h3>
              <p>
                We retain your personal information for as long as necessary to provide our services and comply with legal obligations.
                You may request deletion of your account and associated data at any time.
              </p>
            </div>

            <div className="section">
              <h3>5. Third-Party Services</h3>
              <p>
                Our application may use third-party services for analytics and other purposes. These third parties may collect information
                used to identify you. Please review their privacy policies as we are not responsible for their practices.
              </p>
            </div>

            <div className="section">
              <h3>6. Changes to Privacy Policy</h3>
              <p>
                We may update this privacy policy from time to time. We will notify you of any changes by posting the new privacy policy
                on this page and updating the &quot;last updated&quot; date.
              </p>
            </div>

            <div className="section">
              <h3>7. Contact Us</h3>
              <p>
                If you have questions about this privacy policy or our privacy practices, please contact us at privacy@orthovision.ai
              </p>
            </div>
          </div>
        )}

        {activeSection === 'disclaimer' && (
          <div className="legal-section">
            <h2>Medical Disclaimer</h2>

            <div className="section warning-section">
              <h3>⚠️ Important Medical Disclaimer</h3>
              <p>
                <strong>
                  OrthoVision AI is a diagnostic assistance tool and NOT a substitute for professional medical advice, diagnosis, or
                  treatment by a qualified healthcare professional.
                </strong>
              </p>
            </div>

            <div className="section">
              <h3>1. Limitations of AI Analysis</h3>
              <p>
                While OrthoVision AI utilizes advanced machine learning algorithms for fracture detection and classification, the
                system has limitations including:
              </p>
              <ul>
                <li>Potential for false positives or false negatives</li>
                <li>Difficulty with complex or uncommon fracture types</li>
                <li>Dependence on image quality and proper positioning</li>
                <li>Inability to consider patient history or clinical context</li>
              </ul>
            </div>

            <div className="section">
              <h3>2. Professional Consultation Required</h3>
              <p>
                All results from OrthoVision AI should be reviewed and verified by qualified radiologists or orthopedic specialists
                before making any clinical decisions or initiating treatment.
              </p>
            </div>

            <div className="section">
              <h3>3. Not for Emergency Use</h3>
              <p>
                This application should not be used in emergency situations. Patients requiring immediate medical attention should
                seek care from emergency services.
              </p>
            </div>

            <div className="section">
              <h3>4. No Guarantee of Accuracy</h3>
              <p>
                OrthoVision AI does not guarantee the accuracy, completeness, or reliability of any analysis. The system is provided
                &quot;as is&quot; without warranties of any kind, whether express or implied.
              </p>
            </div>

            <div className="section">
              <h3>5. User Acknowledgment</h3>
              <p>
                By using OrthoVision AI, you acknowledge that you understand and accept these limitations and agree that you will not
                rely solely on this application for medical decisions.
              </p>
            </div>

            <div className="section">
              <h3>6. Liability Limitations</h3>
              <p>
                OrthoVision AI and its developers shall not be liable for any injury, loss, or damage resulting from the use or misuse
                of this application, including but not limited to incorrect diagnoses or delayed treatment due to reliance on this tool.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TermsAndConditions;
