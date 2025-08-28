# ML-Enhanced Cybersecurity System - Flask Version

## Overview
This is the Flask version of the ML-Enhanced Cybersecurity System, designed to replace the Streamlit version and remove all numpy dependencies. The system provides cryptographic key generation and intrusion detection capabilities through a modern web interface.

## Key Features
- **No numpy dependencies** - Pure Python implementation
- **Flask web framework** - Better performance than Streamlit
- **Responsive web interface** - Bootstrap-based UI
- **Cryptographic key generation** - AES, RSA, ECC, and Hybrid keys
- **Intrusion detection system** - Rule-based threat detection
- **Analytics dashboard** - Performance metrics and visualizations
- **Real-time charts** - Chart.js integration

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install Flask requirements:
```bash
pip install -r requirements_flask.txt
```

## Quick Start

1. Run the Flask application:
```bash
python app_flask.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the navigation menu to access different modules:
   - **Dashboard**: System overview and quick actions
   - **Cryptographic Keys**: Generate and analyze encryption keys
   - **Intrusion Detection**: Train models and detect threats
   - **Analytics**: View performance metrics and charts

## Architecture

### Backend Components
- `app_flask.py` - Main Flask application
- `modules/crypto_key_generator.py` - Hash-based key generation (no ML dependencies)
- `modules/intrusion_detection.py` - Rule-based threat detection
- `modules/data_processor.py` - Data processing without pandas
- `modules/visualization_flask.py` - Chart data generation
- `utils/metrics_flask.py` - Metrics calculation without numpy

### Frontend Components
- `templates/dashboard.html` - Main dashboard interface
- Bootstrap 5.1.3 for responsive design
- Chart.js for interactive visualizations
- Font Awesome for icons

## API Endpoints

### System Status
- `GET /api/system-status` - Get current system status

### Cryptographic Operations
- `POST /api/generate-keys` - Generate cryptographic keys
- `POST /api/upload-file` - Upload files for key generation

### Intrusion Detection
- `POST /api/train-ids` - Train IDS models
- `POST /api/run-detection` - Run threat detection

### Analytics
- `GET /api/get-chart-data/<chart_type>` - Get chart data
- `GET /api/export-csv` - Export metrics to CSV
- `GET /api/generate-report` - Generate HTML report

### Quick Actions
- `GET /api/quick-action/<action>` - Execute dashboard quick actions

## Key Improvements Over Streamlit Version

1. **Performance**: Flask is more lightweight and faster than Streamlit
2. **No numpy dependencies**: Pure Python implementation reduces complexity
3. **Better UI/UX**: Modern Bootstrap interface with responsive design
4. **Modular architecture**: Clean separation of concerns
5. **API-first design**: RESTful endpoints for better integration
6. **Real-time updates**: AJAX-based dynamic content updates

## Removed Dependencies
The following packages have been completely removed:
- `numpy`
- `pandas` 
- `matplotlib`
- `seaborn`
- `streamlit`
- `plotly` (replaced with Chart.js)
- `tensorflow`
- `scikit-learn`

## Cryptographic Key Generation
The system now uses a hash-based approach instead of machine learning:
- Multiple SHA-256 rounds for entropy enhancement
- Statistical feature extraction from input data
- Pure Python randomness testing
- Support for AES, RSA, ECC, and Hybrid keys

## Intrusion Detection
Rule-based detection system with:
- Heuristic threat analysis
- Statistical anomaly detection
- Network traffic simulation
- Performance metrics calculation

## File Upload Support
- Text files (.txt, .csv, .json)
- Image files (.png, .jpg, .jpeg) for key generation
- Secure file handling with validation

## Configuration
Edit `app_flask.py` to modify:
- Server port (default: 5000)
- Upload folder location
- File size limits
- Debug mode settings

## Security Notes
- Change the secret key in production
- Implement proper authentication for production use
- Validate all file uploads
- Use HTTPS in production environments

## Troubleshooting

### Common Issues
1. **Port already in use**: Change the port in `app_flask.py`
2. **Missing templates**: Ensure the `templates` folder exists
3. **File upload errors**: Check upload folder permissions

### Debug Mode
Enable debug mode by setting `debug=True` in `app.run()` for detailed error messages.

## License
This project is provided as-is for educational and research purposes.
