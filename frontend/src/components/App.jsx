import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './LandingPage.jsx';
import Login from './Login.jsx';
import SignUp from './SignUp.jsx';
import Dashboard from './Dashboard.jsx';
import Settings from './Settings.jsx';
import History from './History.jsx';
import TermsAndConditions from './TermsAndConditions.jsx';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    // Check if user is already logged in
    const savedUser = localStorage.getItem('orthovision_user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (e) {
        localStorage.removeItem('orthovision_user');
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const savedTheme = localStorage.getItem('orthovision_theme');
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem('orthovision_theme', theme);
  }, [theme]);

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  const handleUpdateUser = (updatedUser) => {
    setUser(updatedUser);
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading OrthoVision AI...</p>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={<LandingPage />} 
        />
        <Route 
          path="/login" 
          element={user ? <Navigate to="/dashboard" /> : <Login onLogin={handleLogin} />} 
        />
        <Route 
          path="/signup" 
          element={user ? <Navigate to="/dashboard" /> : <SignUp onLogin={handleLogin} />} 
        />
        <Route 
          path="/dashboard" 
          element={user ? (
            <Dashboard
              user={user}
              onLogout={handleLogout}
            />
          ) : (
            <Navigate to="/login" />
          )}
        />
        <Route 
          path="/settings" 
          element={user ? <Settings user={user} onLogout={handleLogout} onUpdateUser={handleUpdateUser} /> : <Navigate to="/login" />} 
        />
        <Route 
          path="/history" 
          element={user ? <History user={user} /> : <Navigate to="/login" />} 
        />
        <Route 
          path="/terms" 
          element={<TermsAndConditions />} 
        />
        <Route 
          path="/privacy" 
          element={<TermsAndConditions />} 
        />
      </Routes>
      <footer className="app-footer">
        <div className="app-footer__content">
          <span className="app-footer__text">OrthoVision AI</span>
          <button
            type="button"
            className="button button--secondary button--small"
            onClick={handleToggleTheme}
            aria-label="Toggle color theme"
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </footer>
    </Router>
  );
}

export default App;