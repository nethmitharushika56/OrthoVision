import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

const SignUp = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    specialty: '',
    acceptTerms: false
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!formData.acceptTerms) {
      setError('Please accept the terms and conditions');
      return;
    }

    setLoading(true);

    // Simulate registration (replace with actual API call)
    setTimeout(() => {
      const user = { 
        email: formData.email, 
        name: formData.fullName,
        specialty: formData.specialty 
      };
      localStorage.setItem('orthovision_user', JSON.stringify(user));
      onLogin(user);
      navigate('/dashboard');
      setLoading(false);
    }, 800);
  };

  return (
    <div className="auth-container auth-container--split">
      <aside className="auth-panel">
        <div className="auth-panel__logo-wrap">
          <img src="/ortho-vision-logo.svg" alt="OrthoVision Logo" className="auth-panel__logo" />
        </div>
        <h1 className="auth-panel__title">OrthoVision AI</h1>
        <p className="auth-panel__subtitle">Precision fracture detection powered by deep learning</p>

        <div className="auth-panel__features">
          <div className="auth-panel__item">
            <span className="auth-panel__icon">🦴</span>
            <div>
              <h3>Fracture Detection</h3>
              <p>AI-assisted X-ray and CT analysis in seconds</p>
            </div>
          </div>
          <div className="auth-panel__item">
            <span className="auth-panel__icon">📊</span>
            <div>
              <h3>Advanced Analytics</h3>
              <p>Track outcomes and build patient history</p>
            </div>
          </div>
          <div className="auth-panel__item">
            <span className="auth-panel__icon">👥</span>
            <div>
              <h3>Team Collaboration</h3>
              <p>Share cases and discuss with colleagues</p>
            </div>
          </div>
          <div className="auth-panel__item">
            <span className="auth-panel__icon">🔒</span>
            <div>
              <h3>HIPAA Compliant</h3>
              <p>Enterprise-grade data security</p>
            </div>
          </div>
        </div>

        <blockquote className="auth-panel__quote">
          "OrthoVision cut our diagnosis time by 40%."
          <cite>Dr. Priya Sharma, Senior Radiologist</cite>
        </blockquote>
      </aside>

      <div className="auth-card auth-card--signup-modern">
        <div className="auth-steps" aria-label="Signup progress">
          <div className="auth-steps__item auth-steps__item--done">
            <span className="auth-steps__dot">✓</span>
            <span className="auth-steps__label">Profile</span>
          </div>
          <span className="auth-steps__line" />
          <div className="auth-steps__item auth-steps__item--active">
            <span className="auth-steps__dot">2</span>
            <span className="auth-steps__label">Security</span>
          </div>
        </div>

        <div className="auth-header auth-header--left">
          <h1 className="auth-title">Secure your account</h1>
          <p className="auth-subtitle">Set up your login credentials</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form auth-form--compact">
          {error && <div className="auth-error">{error}</div>}

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="fullName" className="form-label">Full Name</label>
              <input
                id="fullName"
                name="fullName"
                type="text"
                value={formData.fullName}
                onChange={handleChange}
                className="form-input"
                placeholder="Dr. John Smith"
                required
                autoComplete="name"
              />
            </div>

            <div className="form-group">
              <label htmlFor="specialty" className="form-label">Specialty</label>
              <select
                id="specialty"
                name="specialty"
                value={formData.specialty}
                onChange={handleChange}
                className="form-input"
                required
              >
                <option value="">Select specialty</option>
                <option value="orthopedic">Orthopedic Surgeon</option>
                <option value="radiologist">Radiologist</option>
                <option value="emergency">Emergency Medicine</option>
                <option value="sports">Sports Medicine</option>
                <option value="general">General Practitioner</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="email" className="form-label">Email Address</label>
            <input
              id="email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              className="form-input"
              placeholder="doctor@hospital.com"
              required
              autoComplete="email"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="password" className="form-label">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                className="form-input"
                placeholder="Enter password"
                required
                minLength="8"
                autoComplete="new-password"
              />
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword" className="form-label">Confirm Password</label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="form-input"
                placeholder="Confirm password"
                required
                minLength="8"
                autoComplete="new-password"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="checkbox-label checkbox-label--block">
              <input
                type="checkbox"
                name="acceptTerms"
                className="checkbox"
                checked={formData.acceptTerms}
                onChange={handleChange}
                required
              />
              <span>I agree to the <a href="#" className="link-text">Terms of Service</a> and <a href="#" className="link-text">Privacy Policy</a></span>
            </label>
          </div>

          <button
            type="submit"
            className="button button--primary button--full"
            disabled={loading}
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>

          <button type="button" className="button button--ghost button--full" onClick={() => navigate('/login')}>
            {'<- Go back'}
          </button>
        </form>

        <div className="auth-footer">
          <p>Already have an account? <Link to="/login" className="link-primary">Sign in</Link></p>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
