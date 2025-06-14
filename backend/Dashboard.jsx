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
