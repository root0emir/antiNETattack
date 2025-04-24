# antiNETattack

A modern web-based cybersecurity tool with AI-powered network attack detection and visualization.

## Features

### Core Functionality
- Real-time network traffic monitoring and analysis
- Advanced AI-powered attack detection using RandomForest and GradientBoosting
- Anomaly detection for potential zero-day attacks
- Detection of 20 different network-based attack types
- Comprehensive packet analysis and visualization

### User Interface
- Modern, responsive web interface with Bootstrap 5
- Interactive dashboard with real-time visualizations using Chart.js
- Dark mode support with persistent settings
- Customizable monitoring and detection settings
- Real-time alerts and notifications

### Attack Detection
- **Supported Attack Types**:
  - DDoS (Distributed Denial of Service)
  - DoS (Denial of Service)
  - Port Scanning
  - SYN Flood
  - UDP Flood
  - Ping Flood (ICMP Flood)
  - TCP Reset Attack
  - ARP Spoofing
  - DNS Spoofing
  - IP Spoofing
  - MAC Spoofing
  - DHCP Spoofing
  - Session Hijacking
  - Replay Attack
  - Smurf Attack
  - ICMP Redirect Attack
  - Deauth Attack
  - NetBIOS Enumeration
  - Network Worm
  - Man-in-the-Middle

## Technical Details

### AI Model
- RandomForest and GradientBoosting classifiers
- Feature selection and hyperparameter tuning
- Advanced feature engineering with 24 network traffic features
- Anomaly detection for unknown attack patterns

### Architecture
- Flask-based web application (backend)
- RESTful API for monitoring and statistics
- Bootstrap 5 and Chart.js (frontend)
- Persistent storage for attack logs and detection history

## Requirements

- Python 3.8+
- Flask and Flask-CORS
- scikit-learn 1.0+
- numpy
- joblib
- Modern web browser with JavaScript enabled

## Installation

```bash
# Clone the repository
git clone https://github.com/root0emir/antiNETattack.git
cd antiNETattack

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser to http://localhost:5000
   ```
   python app.py
   ```

## Usage

1. Access the web interface at http://localhost:5000
2. Start the network monitoring
3. View real-time attack detection results

## License

MIT License
