import React, { useState, useEffect } from 'react';
import { FiActivity, FiAlertTriangle, FiCpu, FiShield, FiClock, FiBarChart2 } from 'react-icons/fi';
import RealTimeMonitor from './RealTimeMonitor';
import Alerts from './Alerts';
import ModelTraining from './ModelTraining';
import './Dashboard.css';

function Dashboard() {
  const [activeTab, setActiveTab] = useState('monitor');
  const [modelStatus, setModelStatus] = useState({});
  const [systemHealth, setSystemHealth] = useState({});
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isLoading, setIsLoading] = useState(true);

  // Fetch system status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setIsLoading(true);
        const [modelRes, healthRes] = await Promise.all([
          fetch('/model-status'),
          fetch('/health')
        ]);
        
        const modelData = await modelRes.json();
        const healthData = await healthRes.json();
        
        setModelStatus(modelData);
        setSystemHealth(healthData);
        setLastUpdated(new Date());
      } catch (error) {
        console.error("Failed to fetch system status:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Format uptime
  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <div className="sidebar">
        <div className="logo-container">
          <FiShield className="logo-icon" />
          <h1 className="logo-text">CyberShield</h1>
        </div>
        
        <nav className="nav-menu">
          <button 
            className={`nav-btn ${activeTab === 'monitor' ? 'active' : ''}`}
            onClick={() => setActiveTab('monitor')}
          >
            <FiActivity className="nav-icon" />
            <span>Real-Time Monitor</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'alerts' ? 'active' : ''}`}
            onClick={() => setActiveTab('alerts')}
          >
            <FiAlertTriangle className="nav-icon" />
            <span>Threat Alerts</span>
            <span className="alert-badge">3</span>
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            <FiCpu className="nav-icon" />
            <span>Model Training</span>
          </button>
        </nav>
        
        <div className="system-info">
          <div className="info-item">
            <FiClock className="info-icon" />
            <span>Uptime: {systemHealth.uptime ? formatUptime(systemHealth.uptime) : '--'}</span>
          </div>
          <div className="info-item">
            <FiBarChart2 className="info-icon" />
            <span>Last updated: {lastUpdated.toLocaleTimeString()}</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Status Header */}
        <header className="content-header">
          <h2 className="content-title">
            {activeTab === 'monitor' && <><FiActivity /> Real-Time Network Monitor</>}
            {activeTab === 'alerts' && <><FiAlertTriangle /> Security Alerts</>}
            {activeTab === 'training' && <><FiCpu /> Model Training</>}
          </h2>
          
          <div className="status-indicators">
            <div className={`status-indicator ${modelStatus.threat_model === 'loaded' ? 'online' : 'offline'}`}>
              <div className="status-light"></div>
              Threat Detection
            </div>
            <div className={`status-indicator ${modelStatus.anomaly_model === 'loaded' ? 'online' : 'offline'}`}>
              <div className="status-light"></div>
              Anomaly Detection
            </div>
          </div>
        </header>

        {/* Loading State */}
        {isLoading ? (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Loading system status...</p>
          </div>
        ) : (
          <>
            {/* Content Sections */}
            <div className="content-section">
              {activeTab === 'monitor' && <RealTimeMonitor />}
              {activeTab === 'alerts' && <Alerts />}
              {activeTab === 'training' && <ModelTraining />}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
