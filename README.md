# AI-Driven LCA Cycle Gas Leakage Detection System

A comprehensive machine learning system for predicting gas leakage in mining and metallurgy industries, with IoT sensor integration and environmental impact assessment.

## 🚀 Features

- **Gas Leakage Prediction**: ML models to predict gas leakage with probability scores
- **Severity Classification**: Categorize leakage severity (low, medium, high, critical)
- **Warning System**: Multi-level warning system (safe, caution, warning, critical, emergency)
- **Location Prediction**: Identify probable leakage locations using clustering
- **IoT Sensor Integration**: Real-time sensor data processing and monitoring
- **Environmental Impact Assessment**: LCA calculations for carbon footprint and environmental costs
- **Real-time Monitoring**: Continuous monitoring with alert notifications
- **API Integration**: RESTful API for frontend integration

## 📁 Project Structure

```
sih-ml/
├── data_parser.py              # Dataset parsing and preprocessing
├── gas_leakage_predictor.py    # Main ML prediction models
├── iot_sensor_system.py        # IoT sensor simulation and integration
├── lca_environmental_calculator.py  # Environmental impact calculations
├── model_manager.py            # Simplified model manager for frontend
├── api_server.py              # Flask API server
├── example_usage.py           # Usage examples
├── dataset.xlsx               # Training dataset
├── parsed_dataset.csv         # Parsed dataset
└── requirements.txt           # Python dependencies
```

## 🛠️ Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**:
   ```bash
   python example_usage.py
   ```

## 🔧 Usage

### Basic Usage with Model Manager

```python
from model_manager import GasLeakageModelManager

# Initialize the system
manager = GasLeakageModelManager()
manager.initialize()

# Single prediction
sensor_data = {
    'timestamp': '2024-01-15 14:30:00',
    'location_id': 'LOC001',
    'location_name': 'Extraction_Zone_A',
    'process_area': 'Open_Pit',
    'equipment_id': 'EQ_EX_001',
    'CO2_ppm': 1200,
    'CO_ppm': 15,
    'SO2_ppm': 2.5,
    'CH4_ppm': 800,
    'H2S_ppm': 5,
    'NOx_ppm': 30,
    'temperature_C': 35.5,
    'humidity_percent': 70,
    'pressure_kPa': 102.5,
    'wind_speed_ms': 2.8,
    'vibration_level_mm_s': 2.1,
    'equipment_age_months': 42,
    'maintenance_days_ago': 8,
    'production_rate_percent': 90
}

# Make prediction
result = manager.predict_single(sensor_data)
print(result)
```

### API Server Usage

```bash
# Start the API server
python api_server.py

# The server will run on http://localhost:5000
```

### API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Single prediction
- `POST /api/predict/batch` - Batch predictions
- `GET /api/sensors/status` - Sensor status
- `POST /api/sensors/start` - Start monitoring
- `POST /api/sensors/stop` - Stop monitoring
- `GET /api/predictions/latest` - Latest predictions
- `POST /api/environmental/calculate` - Environmental impact
- `POST /api/environmental/gas-impact` - Gas impact calculation
- `POST /api/environmental/process-impact` - Process impact calculation
- `POST /api/sensors/simulate-leakage` - Simulate leakage event
- `GET /api/data/historical` - Historical data
- `GET /api/data/statistics` - Data statistics
- `GET /api/models/info` - Model information

## 📊 Model Performance

- **Leakage Detection**: 100% accuracy on test set
- **Severity Classification**: 22% accuracy (limited by small dataset)
- **Warning Level**: 33% accuracy (limited by small dataset)
- **Location Prediction**: K-means clustering for leakage source identification

## 🔬 Technical Details

### Machine Learning Models

1. **Random Forest Classifier** for:
   - Leakage detection (binary classification)
   - Severity classification (multi-class)
   - Warning level prediction (multi-class)

2. **K-means Clustering** for:
   - Location prediction based on leakage patterns

### Features Used

- Gas concentrations (CO2, CO, SO2, CH4, H2S, NOx)
- Environmental factors (temperature, humidity, pressure, wind speed)
- Equipment data (age, maintenance, vibration)
- Process data (production rate, process area)
- Time-based features (hour, day of week)

### Gas Thresholds

- CO2: 1000 ppm
- CO: 50 ppm
- SO2: 5 ppm
- CH4: 1000 ppm
- H2S: 10 ppm
- NOx: 25 ppm

## 🌍 Environmental Impact

The system calculates:
- CO2 equivalent emissions
- Carbon footprint by process area
- Energy and water consumption
- Environmental costs and regulatory compliance
- Sustainability metrics

## 🔔 Alert System

- **Real-time monitoring** with configurable intervals
- **Multi-level alerts** based on severity
- **Gas concentration warnings** when thresholds are exceeded
- **Recommendations** for immediate action

## 📈 Frontend Integration

The system provides JSON-serializable responses for easy frontend integration:

```python
# Example response format
{
    "success": true,
    "prediction": {
        "leakage_detected": true,
        "leakage_probability": 0.76,
        "severity": "medium",
        "warning_level": "warning",
        "predicted_location": "Cluster_0",
        "gas_warnings": [...],
        "recommendations": [...]
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

## 🚨 Safety Features

- Immediate evacuation recommendations for high-risk situations
- Emergency ventilation system activation alerts
- Safety team and emergency services notifications
- Equipment shutdown recommendations for critical situations
- Preventive maintenance scheduling

## 📝 Notes

- The system is trained on a limited dataset (44 samples)
- Model performance may improve with more training data
- Real-time monitoring requires active sensor network
- Environmental calculations use industry-standard factors
- All predictions include confidence scores and recommendations

## 🔧 Customization

You can customize:
- Gas concentration thresholds
- Alert levels and criteria
- Environmental impact factors
- Process area configurations
- Sensor network parameters

## 📞 Support

For integration support or questions, refer to the example usage files and API documentation.
