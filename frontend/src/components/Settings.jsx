import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function Settings({ user, onLogout, onUpdateUser }) {
  const [activeTab, setActiveTab] = useState('profile');
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    phone: user?.phone || '',
    institution: user?.institution || '',
  });
  const [editMode, setEditMode] = useState(false);
  const [password, setPassword] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const navigate = useNavigate();

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveProfile = () => {
    // Save profile data
    const updatedUser = { ...user, ...profileData };
    localStorage.setItem('orthovision_user', JSON.stringify(updatedUser));
    onUpdateUser(updatedUser);
    setEditMode(false);
    alert('Profile updated successfully!');
  };

  const handleDeleteProfile = async () => {
    if (!password) {
      setDeleteError('Please enter your password to confirm deletion');
      return;
    }

    try {
      // In a real app, this would call a backend endpoint
      console.log('Deleting profile...');
      
      // Simulate deletion (replace with actual API call)
      localStorage.removeItem('orthovision_user');
      onLogout();
      navigate('/');
    } catch (error) {
      setDeleteError('Failed to delete profile. Please try again.');
    }
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <button onClick={() => navigate('/dashboard')} className="button button--nav">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          Back to Dashboard
        </button>
        <h1>Settings</h1>
      </div>

      <div className="settings-content">
        <div className="settings-tabs">
          <button
            className={`tab-button ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
              <circle cx="12" cy="7" r="4"/>
            </svg>
            Profile
          </button>
          <button
            className={`tab-button ${activeTab === 'security' ? 'active' : ''}`}
            onClick={() => setActiveTab('security')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            Security
          </button>
          <button
            className={`tab-button ${activeTab === 'preferences' ? 'active' : ''}`}
            onClick={() => setActiveTab('preferences')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="1"/>
              <path d="M12 1v6m0 6v6M4.22 4.22l4.24 4.24m5.08 5.08l4.24 4.24M1 12h6m6 0h6M4.22 19.78l4.24-4.24m5.08-5.08l4.24-4.24"/>
            </svg>
            Preferences
          </button>
        </div>

        <div className="settings-panel">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="settings-section">
              <div className="section-header">
                <h2>Profile Information</h2>
                {!editMode && (
                  <button onClick={() => setEditMode(true)} className="button button--primary">
                    Edit Profile
                  </button>
                )}
              </div>

              <div className="profile-card">
                <div className="profile-avatar-section">
                  <div className="profile-avatar">
                    {user?.name?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div className="profile-avatar-info">
                    <h3>{user?.name || 'User'}</h3>
                    <p>{user?.email || 'No email'}</p>
                  </div>
                </div>

                {editMode ? (
                  <div className="profile-edit-form">
                    <div className="form-group">
                      <label>Full Name</label>
                      <input
                        type="text"
                        name="name"
                        value={profileData.name}
                        onChange={handleProfileChange}
                        placeholder="Enter your full name"
                      />
                    </div>

                    <div className="form-group">
                      <label>Email Address</label>
                      <input
                        type="email"
                        name="email"
                        value={profileData.email}
                        onChange={handleProfileChange}
                        placeholder="Enter your email"
                      />
                    </div>

                    <div className="form-group">
                      <label>Phone Number</label>
                      <input
                        type="tel"
                        name="phone"
                        value={profileData.phone}
                        onChange={handleProfileChange}
                        placeholder="Enter your phone number"
                      />
                    </div>

                    <div className="form-group">
                      <label>Institution/Hospital</label>
                      <input
                        type="text"
                        name="institution"
                        value={profileData.institution}
                        onChange={handleProfileChange}
                        placeholder="Enter your institution"
                      />
                    </div>

                    <div className="form-actions">
                      <button onClick={handleSaveProfile} className="button button--primary">
                        Save Changes
                      </button>
                      <button onClick={() => setEditMode(false)} className="button button--secondary">
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="profile-view">
                    <div className="profile-item">
                      <span className="label">Name:</span>
                      <span className="value">{profileData.name || 'Not set'}</span>
                    </div>
                    <div className="profile-item">
                      <span className="label">Email:</span>
                      <span className="value">{profileData.email || 'Not set'}</span>
                    </div>
                    <div className="profile-item">
                      <span className="label">Phone:</span>
                      <span className="value">{profileData.phone || 'Not set'}</span>
                    </div>
                    <div className="profile-item">
                      <span className="label">Institution:</span>
                      <span className="value">{profileData.institution || 'Not set'}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="settings-section">
              <div className="section-header">
                <h2>Security Settings</h2>
              </div>

              <div className="security-card">
                <h3>Change Password</h3>
                <div className="form-group">
                  <label>Current Password</label>
                  <input type="password" placeholder="Enter current password" />
                </div>
                <div className="form-group">
                  <label>New Password</label>
                  <input type="password" placeholder="Enter new password" />
                </div>
                <div className="form-group">
                  <label>Confirm Password</label>
                  <input type="password" placeholder="Confirm new password" />
                </div>
                <button className="button button--primary">Update Password</button>
              </div>

              <div className="danger-zone">
                <h3>Danger Zone</h3>
                <div className="delete-account-section">
                  <div className="delete-info">
                    <h4>Delete Account</h4>
                    <p>Permanently delete your account and all associated data. This action cannot be undone.</p>
                  </div>
                  <button
                    onClick={() => setShowDeleteConfirm(true)}
                    className="button button--danger"
                  >
                    Delete Account
                  </button>
                </div>

                {showDeleteConfirm && (
                  <div className="delete-confirmation">
                    <div className="confirmation-modal">
                      <h3>Delete Account Confirmation</h3>
                      <p>This action is irreversible. All your data will be permanently deleted.</p>

                      <div className="form-group">
                        <label>Enter your password to confirm:</label>
                        <input
                          type="password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          placeholder="Enter your password"
                        />
                      </div>

                      {deleteError && <p className="error-message">{deleteError}</p>}

                      <div className="modal-actions">
                        <button
                          onClick={handleDeleteProfile}
                          className="button button--danger"
                        >
                          Confirm Deletion
                        </button>
                        <button
                          onClick={() => {
                            setShowDeleteConfirm(false);
                            setPassword('');
                            setDeleteError('');
                          }}
                          className="button button--secondary"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Preferences Tab */}
          {activeTab === 'preferences' && (
            <div className="settings-section">
              <div className="section-header">
                <h2>Preferences & Notifications</h2>
              </div>

              <div className="preferences-card">
                <div className="preference-item">
                  <div className="preference-info">
                    <h4>Email Notifications</h4>
                    <p>Receive email updates about your analyses and account activity</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>

                <div className="preference-item">
                  <div className="preference-info">
                    <h4>Analysis History</h4>
                    <p>Keep a detailed history of all your analyses</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>

                <div className="preference-item">
                  <div className="preference-info">
                    <h4>Dark Mode</h4>
                    <p>Use dark theme for the application</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>

                <div className="preference-item">
                  <div className="preference-info">
                    <h4>Data Privacy</h4>
                    <p>Share usage data to help improve OrthoVision AI</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="legal-links">
                <h3>Legal & Compliance</h3>
                <button
                  onClick={() => navigate('/terms')}
                  className="link-button"
                >
                  Terms and Conditions
                </button>
                <button
                  onClick={() => navigate('/privacy')}
                  className="link-button"
                >
                  Privacy Policy
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Settings;
