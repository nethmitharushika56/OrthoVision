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
    <div className="auth-container">
      <div className="auth-card auth-card--wide">
        <div className="auth-header">
          <img src="/ortho-vision-logo.svg" alt="OrthoVision Logo" className="brand__logo auth-logo" style={{width: '80px', height: 'auto'}} />
          <h1 className="auth-title">Create Account</h1>
          <p className="auth-subtitle">Join OrthoVision AI platform</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
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
                placeholder="Min. 8 characters"
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
                placeholder="Re-enter password"
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

          <div className="auth-divider">
            <span>or sign up with</span>
          </div>

          <div className="social-buttons">
            <button type="button" className="button button--social">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Google
            </button>
            <button type="button" className="button button--social">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              GitHub
            </button>
          </div>
        </form>

        <div className="auth-footer">
          <p>Already have an account? <Link to="/login" className="link-primary">Sign in</Link></p>
        </div>
      </div>

      <div className="auth-features">
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Advanced Analytics</h3>
          <p>Track patient outcomes and analysis history</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">👥</div>
          <h3>Team Collaboration</h3>
          <p>Share and discuss cases with colleagues</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📱</div>
          <h3>Mobile Access</h3>
          <p>Access from any device, anywhere</p>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
