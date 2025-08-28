# Real-Time Analytics Dashboard - Setup and Usage Guide

## Overview

The Real-Time Analytics Dashboard provides comprehensive monitoring and visualization of your cybersecurity system's performance, including cryptographic operations, intrusion detection metrics, and system health monitoring.

## Features

### 🔄 Real-Time Monitoring
- Live updates every 2 seconds
- System resource monitoring (CPU, Memory)
- Real-time threat detection metrics
- Interactive charts and visualizations

### 📊 Key Metrics Displayed

#### IDS (Intrusion Detection System) Metrics
- **Accuracy**: Overall detection accuracy percentage
- **Precision**: True positive rate (precision score)
- **Recall**: Sensitivity/true positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Rate of incorrectly flagged normal activities
- **Detection Latency**: Average time to detect threats (milliseconds)
- **Throughput**: Number of samples processed per second

#### Cryptographic Metrics
- **Key Generation Time**: Average time to generate keys
- **Security Score**: Attack resistance rating (0-100)
- **Encoding/Decoding Time**: Combined average processing time

#### System Health
- **CPU Usage**: Real-time processor utilization
- **Memory Usage**: RAM consumption percentage
- **System Uptime**: Time since dashboard initialization
- **Module Status**: Availability of key system components

### 📈 Visualizations

#### 1. Performance Trend Chart (Chart.js)
- Real-time line chart showing accuracy, precision, recall, and F1 score over time
- Maintains last 20 data points for smooth visualization
- Color-coded metrics with fill gradients

#### 2. Threat Detection Doughnut Chart (Chart.js)
- Shows distribution of true positives, false positives, true negatives, false negatives
- Real-time updates based on current detection results

#### 3. Advanced Plotly Visualizations
- **Detection Distribution Pie Chart**: Statistical breakdown of detection results
- **24-Hour Performance Trends**: Dual-axis chart showing accuracy trends and threat counts

### 🎯 Per-Attack Detection Matrix
Interactive table showing detailed metrics for different attack types:
- SQL Injection, XSS Attack, Phishing, Malware, DDoS, CSRF, Path Traversal
- Metrics include precision, recall, F1 score, accuracy, false positive rate
- Color-coded performance indicators:
  - 🟢 Excellent (≥95%)
  - 🟡 Good (85-94%)
  - 🟠 Warning (70-84%)
  - 🔴 Poor (<70%)

### 🤖 AI-Generated Recommendations
Smart system recommendations based on:
- CPU and memory usage patterns
- Detection accuracy performance
- System resource utilization
- Real-time performance analysis

## Setup Instructions

### 1. Files Required
Ensure these files are in your project structure:
```
Cyber/
├── app_flask.py                           # Main Flask application
├── templates/
│   └── realtime_analytics.html           # Dashboard template
├── crypto_metrics.py                     # Cryptographic metrics collector
├── ids_metrics.py                        # IDS metrics collector
└── modules/
    └── simple_intrusion_detector.py      # Core detection system
```

### 2. Dependencies
Install required Python packages:
```bash
pip install flask plotly chart.js psutil
```

### 3. Starting the Application
```bash
python app_flask.py
```

The Flask app will start on `http://localhost:5000`

### 4. Accessing the Dashboard
Navigate to: `http://localhost:5000/realtime-analytics`

## API Endpoints

The dashboard uses these real-time API endpoints:

### System Status
- **GET** `/api/system-status`
- Returns: CPU usage, memory usage, module availability

### IDS Metrics
- **GET** `/api/ids-realtime-metrics`
- Returns: Real-time intrusion detection performance metrics

### Crypto Metrics  
- **GET** `/api/crypto-realtime-metrics`
- Returns: Cryptographic operation performance metrics

### Attack Matrix
- **GET** `/api/attack-matrix-metrics`
- Returns: Per-attack-type detection performance matrix

### System Recommendations
- **GET** `/api/system-recommendations`
- Returns: AI-generated performance recommendations

## Configuration Options

### Dashboard Refresh Rate
```javascript
let refreshInterval = 2000; // 2 seconds (in milliseconds)
```

### Chart Data Points
```javascript
const maxDataPoints = 20; // Number of historical points to display
```

### Mock Data Generation
If real metrics modules aren't available, the system automatically generates realistic mock data to demonstrate functionality.

## Troubleshooting

### Common Issues

1. **Dashboard shows "Connection Error"**
   - Ensure Flask app is running on port 5000
   - Check console for JavaScript errors

2. **Metrics show zeros or don't update**
   - Mock data generation will activate automatically
   - Check if `crypto_metrics.py` and `ids_metrics.py` are properly imported

3. **Charts not displaying**
   - Ensure CDN libraries (Chart.js, Plotly) are loaded
   - Check browser console for errors

### Debug Mode
Enable Flask debug mode for detailed error messages:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Customization

### Adding New Metrics
1. Add metric collection in appropriate collector class
2. Update API endpoint to return new metric
3. Add UI elements in `realtime_analytics.html`
4. Update JavaScript to handle new data

### Modifying Chart Appearance
Customize Chart.js configurations in the `initializeCharts()` function:
```javascript
// Example: Change chart colors
datasets: [{
    borderColor: '#your-color',
    backgroundColor: 'rgba(your-color, 0.1)',
    // ... other properties
}]
```

### Adding New Visualizations
1. Create new chart container in HTML
2. Initialize chart in JavaScript
3. Add data update function
4. Call update function in main refresh cycle

## Performance Considerations

- Dashboard updates every 2 seconds - adjust `refreshInterval` as needed
- Charts maintain limited historical data to prevent memory issues
- Real-time updates use efficient Chart.js animation: `update('none')`
- Plotly charts are optimized for responsive display

## Security Notes

- Dashboard displays system performance metrics - ensure appropriate access controls
- Real-time data includes system resource information
- Consider implementing authentication for production deployments

## Support

For issues or questions:
1. Check console logs in browser developer tools
2. Review Flask application logs
3. Verify all required dependencies are installed
4. Ensure proper file structure and imports

---

*Last updated: January 2024*
*Version: 1.0.0*
