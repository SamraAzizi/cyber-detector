import React, { useState, useEffect, useCallback } from 'react';
import { 
  FiActivity, 
  FiAlertTriangle, 
  FiCpu, 
  FiShield, 
  FiClock, 
  FiBarChart2,
  FiSettings,
  FiUser,
  FiLogOut,
  FiRefreshCw,
  FiAlertCircle
} from 'react-icons/fi';
import { MdSecurity, MdNetworkCheck } from 'react-icons/md';
import { BsGraphUp, BsShieldLock } from 'react-icons/bs';
import RealTimeMonitor from './RealTimeMonitor';
import Alerts from './Alerts';
import ModelTraining from './ModelTraining';
import SystemSettings from './SystemSettings';
import UserProfile from './UserProfile';
import HealthMetrics from './HealthMetrics';
import NetworkMap from './NetworkMap';
import './Dashboard.css';

function Dashboard() {
  const [activeTab, setActiveTab] = useState('monitor');
  const [modelStatus, setModelStatus] = useState({
    threat_model: 'loading',
    anomaly_model: 'loading'
  });
  const [systemHealth, setSystemHealth] = useState({
    uptime: 0,
    cpu_usage: 0,
    memory_usage: 0
  });
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [unreadAlerts, setUnreadAlerts] = useState(0);
  const [darkMode, setDarkMode] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);

  // Toggle dark mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
  }, [darkMode]);

  // Fetch system status with error handling
  const fetchStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const [modelRes, healthRes, alertsRes] = await Promise.all([
        fetch('/api/model-status'),
        fetch('/api/health'),
        fetch('/api/alerts/unread')
      ]);
      
      if (!modelRes.ok || !healthRes.ok || !alertsRes.ok) {
        throw new Error('Failed to fetch system data');
      }
      
      const [modelData, healthData, alertsData] = await Promise.all([
        modelRes.json(),
        healthRes.json(),
        alertsRes.json()
      ]);
      
      setModelStatus(modelData);
      setSystemHealth(healthData);
      setUnreadAlerts(alertsData.count || 0);
      setLastUpdated(new Date());
      
      // Add success notification
      addNotification('System data updated successfully', 'success');
    } catch (err) {
      console.error("Failed to fetch system status:", err);
      setError(err.message);
      addNotification('Failed to fetch system data', 'error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Add notification to queue
  const addNotification = (message, type = 'info') => {
    const newNotification = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date()
    };
    
    setNotifications(prev => [newNotification, ...prev].slice(0, 5)); // Keep only 5 latest
  };

  // Initial fetch and setup interval
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Manual refresh
  const handleRefresh = () => {
    addNotification('Manual refresh initiated', 'info');
    fetchStatus();
  };

  // Format uptime
  const formatUptime = (seconds) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    return `${days > 0 ? `${days}d ` : ''}${hours}h ${minutes}m`;
  };

  // Calculate health percentage
  const calculateHealthPercentage = () => {
    const { cpu_usage = 0, memory_usage = 0 } = systemHealth.system_info || {};
    const modelHealth = modelStatus.threat_model === 'loaded' && modelStatus.anomaly_model === 'loaded' ? 100 : 50;
    const cpuHealth = 100 - cpu_usage;
    const memoryHealth = 100 - memory_usage;
    
    return Math.round((modelHealth + cpuHealth + memoryHealth) / 3);
  };

  // Handle logout
  const handleLogout = () => {
    addNotification('Logging out...', 'info');
    // Implement actual logout logic here
  };

  return (
    <div className={`dashboard-container ${darkMode ? 'dark-mode' : ''}`}>
      {/* Sidebar Navigation */}
      <div className="sidebar">
        <div className="logo-container">
          <FiShield className="logo-icon" />
          <h1 className="logo-text">CyberShield</h1>
          <button 
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
            aria-label="Toggle dark mode"
          >
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>
        
        <nav className="nav-menu">
          <button 
            className={`nav-btn ${activeTab === 'monitor' ? 'active' : ''}`}
            onClick={() => setActiveTab('monitor')}
            aria-current={activeTab === 'monitor'}
          >
            <FiActivity className="nav-icon" />
            <span>Real-Time Monitor</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'alerts' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('alerts');
              setUnreadAlerts(0); // Mark as read
            }}
            aria-current={activeTab === 'alerts'}
          >
            <FiAlertTriangle className="nav-icon" />
            <span>Threat Alerts</span>
            {unreadAlerts > 0 && (
              <span className="alert-badge">
                {unreadAlerts > 9 ? '9+' : unreadAlerts}
              </span>
            )}
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'network' ? 'active' : ''}`}
            onClick={() => setActiveTab('network')}
            aria-current={activeTab === 'network'}
          >
            <MdNetworkCheck className="nav-icon" />
            <span>Network Map</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
            aria-current={activeTab === 'metrics'}
          >
            <BsGraphUp className="nav-icon" />
            <span>Health Metrics</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
            aria-current={activeTab === 'training'}
          >
            <FiCpu className="nav-icon" />
            <span>Model Training</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
            aria-current={activeTab === 'settings'}
          >
            <FiSettings className="nav-icon" />
            <span>System Settings</span>
          </button>
        </nav>
        
        <div className="system-info">
          <div className="health-indicator">
            <div className="health-bar">
              <div 
                className="health-progress" 
                style={{ width: `${calculateHealthPercentage()}%` }}
                data-health={calculateHealthPercentage()}
              ></div>
            </div>
            <span>System Health: {calculateHealthPercentage()}%</span>
          </div>
          
          <div className="info-item">
            <FiClock className="info-icon" />
            <span>Uptime: {systemHealth.uptime ? formatUptime(systemHealth.uptime) : '--'}</span>
          </div>
          
          <div className="info-item">
            <FiBarChart2 className="info-icon" />
            <span>Last updated: {lastUpdated.toLocaleTimeString()}</span>
            <button 
              className="refresh-btn"
              onClick={handleRefresh}
              disabled={isLoading}
              aria-label="Refresh data"
            >
              <FiRefreshCw className={`refresh-icon ${isLoading ? 'spinning' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Top Header Bar */}
        <header className="top-header">
          <h2 className="content-title">
            {activeTab === 'monitor' && <><FiActivity /> Real-Time Network Monitor</>}
            {activeTab === 'alerts' && <><FiAlertTriangle /> Security Alerts ({unreadAlerts} new)</>}
            {activeTab === 'network' && <><MdNetworkCheck /> Network Map</>}
            {activeTab === 'metrics' && <><BsGraphUp /> System Health Metrics</>}
            {activeTab === 'training' && <><FiCpu /> Model Training</>}
            {activeTab === 'settings' && <><FiSettings /> System Settings</>}
          </h2>
          
          <div className="header-controls">
            <div className="status-indicators">
              <div className={`status-indicator ${modelStatus.threat_model === 'loaded' ? 'online' : 'offline'}`}>
                <div className="status-light"></div>
                <BsShieldLock className="status-icon" />
                <span>Threat Detection</span>
              </div>
              <div className={`status-indicator ${modelStatus.anomaly_model === 'loaded' ? 'online' : 'offline'}`}>
                <div className="status-light"></div>
                <MdSecurity className="status-icon" />
                <span>Anomaly Detection</span>
              </div>
            </div>
            
            <div className="user-menu-container">
              <button 
                className="user-btn"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                aria-expanded={userMenuOpen}
                aria-label="User menu"
              >
                <FiUser className="user-icon" />
              </button>
              
              {userMenuOpen && (
                <div className="user-menu">
                  <button 
                    className="user-menu-item"
                    onClick={() => {
                      setActiveTab('profile');
                      setUserMenuOpen(false);
                    }}
                  >
                    <FiUser className="menu-icon" />
                    <span>Profile</span>
                  </button>
                  <button 
                    className="user-menu-item"
                    onClick={handleLogout}
                  >
                    <FiLogOut className="menu-icon" />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
