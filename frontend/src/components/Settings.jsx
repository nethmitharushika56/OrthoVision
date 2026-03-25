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
  const [profileImage, setProfileImage] = useState(user?.profileImage || '');
  const [imageError, setImageError] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [password, setPassword] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    setProfileData({
      name: user?.name || '',
      email: user?.email || '',
      phone: user?.phone || '',
      institution: user?.institution || '',
    });
    setProfileImage(user?.profileImage || '');
  }, [user]);

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveProfile = async () => {
    // Save profile data
    const updatedUser = { ...user, ...profileData, profileImage };
    localStorage.setItem('orthovision_user', JSON.stringify(updatedUser));
    onUpdateUser(updatedUser);

    try {
      await fetch('/users/upsert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedUser),
      });
    } catch (error) {
      console.warn('Failed to sync profile to backend:', error);
    }

    setEditMode(false);
  };

  const handleProfileImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      setImageError('Please select a valid image file.');
      return;
    }

    if (file.size > 2 * 1024 * 1024) {
      setImageError('Image size must be less than 2MB.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setProfileImage(typeof reader.result === 'string' ? reader.result : '');
      setImageError('');
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveProfileImage = () => {
    setProfileImage('');
    setImageError('');
  };

  const userInitial = (profileData.name || user?.name || 'U').charAt(0).toUpperCase();

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
        <button onClick={() => navigate('/dashboard')} className="back-button">
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
            People
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
            Appearance
          </button>
        </div>

        <div className="settings-panel">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="settings-section settings-section--structured">
              <div className="settings-group">
                <h3 className="settings-group-title">People</h3>
                <div className="settings-list-card">
                  <div className="settings-row settings-row--profile">
                    <div className="settings-row-main">
                      <div className="profile-avatar-wrap">
                        {profileImage ? (
                          <img src={profileImage} alt="Profile" className="profile-avatar-image" />
                        ) : (
                          <div className="profile-avatar">{userInitial}</div>
                        )}
                      </div>

                      <div className="profile-avatar-info">
                        <h3>{profileData.name || 'User'}</h3>
                        <p>{profileData.email || 'No email'}</p>
                      </div>
                    </div>

                    <div className="settings-row-actions">
                      {!editMode ? (
                        <button onClick={() => setEditMode(true)} className="button button--primary">
                          Edit Profile
                        </button>
                      ) : (
                        <>
                          <label className="button button--secondary profile-upload-label">
                            Upload Photo
                            <input
                              type="file"
                              accept="image/*"
                              onChange={handleProfileImageChange}
                              className="profile-upload-input"
                            />
                          </label>
                          {profileImage && (
                            <button onClick={handleRemoveProfileImage} className="button button--ghost" type="button">
                              Remove Photo
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                  {imageError && <p className="error-message settings-inline-error">{imageError}</p>}
                </div>
              </div>

              <div className="settings-group">
                <h3 className="settings-group-title">Personal information</h3>
                <div className="settings-list-card">
                  {editMode ? (
                    <div className="profile-edit-form profile-edit-form--rows">
                      <div className="settings-row settings-row--field">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Full Name</span>
                        </div>
                        <div className="settings-row-value">
                          <input
                            type="text"
                            name="name"
                            value={profileData.name}
                            onChange={handleProfileChange}
                            placeholder="Enter your full name"
                          />
                        </div>
                      </div>

                      <div className="settings-row settings-row--field">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Email Address</span>
                        </div>
                        <div className="settings-row-value">
                          <input
                            type="email"
                            name="email"
                            value={profileData.email}
                            onChange={handleProfileChange}
                            placeholder="Enter your email"
                          />
                        </div>
                      </div>

                      <div className="settings-row settings-row--field">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Phone Number</span>
                        </div>
                        <div className="settings-row-value">
                          <input
                            type="tel"
                            name="phone"
                            value={profileData.phone}
                            onChange={handleProfileChange}
                            placeholder="Enter your phone number"
                          />
                        </div>
                      </div>

                      <div className="settings-row settings-row--field">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Institution</span>
                        </div>
                        <div className="settings-row-value">
                          <input
                            type="text"
                            name="institution"
                            value={profileData.institution}
                            onChange={handleProfileChange}
                            placeholder="Enter your institution"
                          />
                        </div>
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
                    <>
                      <div className="settings-row">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Name</span>
                        </div>
                        <div className="settings-row-value">{profileData.name || 'Not set'}</div>
                      </div>
                      <div className="settings-row">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Email</span>
                        </div>
                        <div className="settings-row-value">{profileData.email || 'Not set'}</div>
                      </div>
                      <div className="settings-row">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Phone</span>
                        </div>
                        <div className="settings-row-value">{profileData.phone || 'Not set'}</div>
                      </div>
                      <div className="settings-row">
                        <div className="settings-row-main">
                          <span className="settings-row-label">Institution</span>
                        </div>
                        <div className="settings-row-value">{profileData.institution || 'Not set'}</div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="settings-section settings-section--structured">
              <div className="settings-group">
                <h3 className="settings-group-title">Security</h3>
                <div className="settings-list-card">
                  <div className="settings-row settings-row--field">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Current Password</span>
                    </div>
                    <div className="settings-row-value">
                      <input type="password" placeholder="Enter current password" />
                    </div>
                  </div>
                  <div className="settings-row settings-row--field">
                    <div className="settings-row-main">
                      <span className="settings-row-label">New Password</span>
                    </div>
                    <div className="settings-row-value">
                      <input type="password" placeholder="Enter new password" />
                    </div>
                  </div>
                  <div className="settings-row settings-row--field">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Confirm Password</span>
                    </div>
                    <div className="settings-row-value">
                      <input type="password" placeholder="Confirm new password" />
                    </div>
                  </div>
                  <div className="form-actions">
                    <button className="button button--primary">Update Password</button>
                  </div>
                </div>
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
            <div className="settings-section settings-section--structured">
              <div className="settings-group">
                <h3 className="settings-group-title">Appearance & Preferences</h3>
                <div className="settings-list-card">
                  <div className="settings-row settings-row--toggle">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Email Notifications</span>
                      <p className="settings-row-description">Receive email updates about your analyses and account activity</p>
                    </div>
                    <label className="toggle-switch">
                      <input type="checkbox" defaultChecked />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>

                  <div className="settings-row settings-row--toggle">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Analysis History</span>
                      <p className="settings-row-description">Keep a detailed history of all your analyses</p>
                    </div>
                    <label className="toggle-switch">
                      <input type="checkbox" defaultChecked />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>

                  <div className="settings-row settings-row--toggle">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Dark Mode</span>
                      <p className="settings-row-description">Use dark theme for the application</p>
                    </div>
                    <label className="toggle-switch">
                      <input type="checkbox" defaultChecked />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>

                  <div className="settings-row settings-row--toggle">
                    <div className="settings-row-main">
                      <span className="settings-row-label">Data Privacy</span>
                      <p className="settings-row-description">Share usage data to help improve OrthoVision AI</p>
                    </div>
                    <label className="toggle-switch">
                      <input type="checkbox" defaultChecked />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="legal-links">
                <h3 className="settings-group-title">Legal & Compliance</h3>
                <div className="settings-list-card">
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
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Settings;
