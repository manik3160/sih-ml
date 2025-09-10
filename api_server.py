"""
API Server for Gas Leakage Detection System
Provides REST API endpoints for frontend integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import threading
import time
from gas_leakage_predictor import GasLeakagePredictor
from iot_sensor_system import IoTSensorNetwork, AlertHandler
from lca_environmental_calculator import LCAEnvironmentalCalculator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables
predictor = None
sensor_network = None
environmental_calculator = None
alert_handler = None
is_monitoring = False

def initialize_system():
    """Initialize the ML system components"""
    global predictor, sensor_network, environmental_calculator, alert_handler
    
    print("Initializing ML system...")
    
    # Initialize predictor
    predictor = GasLeakagePredictor()
    predictor.load_data()
    predictor.train_models()
    
    # Initialize sensor network
    sensor_network = IoTSensorNetwork(predictor)
    
    # Initialize environmental calculator
    environmental_calculator = LCAEnvironmentalCalculator()
    
    # Initialize alert handler
    alert_handler = AlertHandler()
    sensor_network.add_alert_callback(alert_handler.handle_alert)
    
    print("ML system initialized successfully!")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_initialized': predictor is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_leakage():
    """
    Predict gas leakage for given sensor data
    
    Expected JSON payload:
    {
        "timestamp": "2024-01-15 14:30:00",
        "location_id": "LOC001",
        "location_name": "Extraction_Zone_A",
        "process_area": "Open_Pit",
        "equipment_id": "EQ_EX_001",
        "CO2_ppm": 1200,
        "CO_ppm": 15,
        "SO2_ppm": 2.5,
        "CH4_ppm": 800,
        "H2S_ppm": 5,
        "NOx_ppm": 30,
        "temperature_C": 35.5,
        "humidity_percent": 70,
        "pressure_kPa": 102.5,
        "wind_speed_ms": 2.8,
        "vibration_level_mm_s": 2.1,
        "equipment_age_months": 42,
        "maintenance_days_ago": 8,
        "production_rate_percent": 90
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        prediction = predictor.predict_leakage(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict gas leakage for multiple sensor readings
    
    Expected JSON payload:
    {
        "readings": [
            {sensor_data_1},
            {sensor_data_2},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'readings' not in data:
            return jsonify({'error': 'No readings provided'}), 400
        
        predictions = []
        for reading in data['readings']:
            prediction = predictor.predict_leakage(reading)
            predictions.append({
                'sensor_data': reading,
                'prediction': prediction
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/sensors/status', methods=['GET'])
def get_sensor_status():
    """Get status of all IoT sensors"""
    try:
        status = sensor_network.get_sensor_status()
        return jsonify({
            'success': True,
            'sensor_status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/sensors/start', methods=['POST'])
def start_monitoring():
    """Start IoT sensor monitoring"""
    global is_monitoring
    
    try:
        data = request.get_json() or {}
        interval = data.get('interval', 5.0)
        
        if not is_monitoring:
            sensor_network.start_monitoring(interval=interval)
            is_monitoring = True
            
            return jsonify({
                'success': True,
                'message': f'Monitoring started with {interval}s interval',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Monitoring already active',
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/sensors/stop', methods=['POST'])
def stop_monitoring():
    """Stop IoT sensor monitoring"""
    global is_monitoring
    
    try:
        if is_monitoring:
            sensor_network.stop_monitoring()
            is_monitoring = False
            
            return jsonify({
                'success': True,
                'message': 'Monitoring stopped',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Monitoring not active',
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_predictions():
    """Get latest predictions from monitoring"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 10)
        
        predictions = sensor_network.get_latest_predictions(limit=limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/environmental/calculate', methods=['POST'])
def calculate_environmental_impact():
    """
    Calculate environmental impact for given data
    
    Expected JSON payload:
    {
        "prediction_data": {prediction_data},
        "process_data": {process_data},
        "historical_data": {historical_data}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction_data = data.get('prediction_data', {})
        process_data = data.get('process_data', {})
        historical_data = data.get('historical_data', {})
        
        # Convert historical data to DataFrame if provided
        if historical_data:
            historical_df = pd.DataFrame(historical_data)
            if 'timestamp' in historical_df.columns:
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        else:
            historical_df = pd.DataFrame()
        
        # Calculate environmental impact
        report = environmental_calculator.create_environmental_report(
            prediction_data, process_data, historical_df
        )
        
        return jsonify({
            'success': True,
            'environmental_report': report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/environmental/gas-impact', methods=['POST'])
def calculate_gas_impact():
    """
    Calculate gas emissions impact
    
    Expected JSON payload:
    {
        "gas_concentrations": {
            "CO2_ppm": 1200,
            "CO_ppm": 15,
            "SO2_ppm": 2.5,
            "CH4_ppm": 800,
            "H2S_ppm": 5,
            "NOx_ppm": 30
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'gas_concentrations' not in data:
            return jsonify({'error': 'No gas concentrations provided'}), 400
        
        gas_impacts = environmental_calculator.calculate_gas_emissions_impact(
            data['gas_concentrations']
        )
        
        return jsonify({
            'success': True,
            'gas_impacts': gas_impacts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/environmental/process-impact', methods=['POST'])
def calculate_process_impact():
    """
    Calculate process environmental impact
    
    Expected JSON payload:
    {
        "process_data": {
            "Open_Pit": 1000,
            "Processing": 800,
            "Smelting": 200,
            "Refining": 150,
            "Waste_Management": 500
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'process_data' not in data:
            return jsonify({'error': 'No process data provided'}), 400
        
        process_impacts = environmental_calculator.calculate_process_environmental_impact(
            data['process_data']
        )
        
        return jsonify({
            'success': True,
            'process_impacts': process_impacts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/sensors/simulate-leakage', methods=['POST'])
def simulate_leakage():
    """
    Simulate a gas leakage event
    
    Expected JSON payload:
    {
        "location": "Smelter_Unit_1",
        "gas_type": "CO2_ppm",
        "intensity": 3.0
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        location = data.get('location')
        gas_type = data.get('gas_type', 'CO2_ppm')
        intensity = data.get('intensity', 2.0)
        
        if not location:
            return jsonify({'error': 'Location not specified'}), 400
        
        sensor_network.simulate_leakage_event(location, gas_type, intensity)
        
        return jsonify({
            'success': True,
            'message': f'Simulated {gas_type} leakage at {location} with intensity {intensity}',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/historical', methods=['GET'])
def get_historical_data():
    """Get historical data for analysis"""
    try:
        # Get historical data from the dataset
        df = predictor.df
        
        # Convert to JSON-serializable format
        historical_data = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'historical_data': historical_data,
            'count': len(historical_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/data/statistics', methods=['GET'])
def get_data_statistics():
    """Get statistical analysis of the dataset"""
    try:
        df = predictor.df
        
        # Calculate statistics
        stats = {
            'total_records': len(df),
            'leakage_incidents': int(df['leakage_detected'].sum()),
            'leakage_rate': float(df['leakage_detected'].mean()),
            'locations': df['location_name'].nunique(),
            'process_areas': df['process_area'].nunique(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'gas_statistics': {
                'CO2_ppm': {
                    'mean': float(df['CO2_ppm'].mean()),
                    'max': float(df['CO2_ppm'].max()),
                    'min': float(df['CO2_ppm'].min())
                },
                'CO_ppm': {
                    'mean': float(df['CO_ppm'].mean()),
                    'max': float(df['CO_ppm'].max()),
                    'min': float(df['CO_ppm'].min())
                },
                'SO2_ppm': {
                    'mean': float(df['SO2_ppm'].mean()),
                    'max': float(df['SO2_ppm'].max()),
                    'min': float(df['SO2_ppm'].min())
                },
                'CH4_ppm': {
                    'mean': float(df['CH4_ppm'].mean()),
                    'max': float(df['CH4_ppm'].max()),
                    'min': float(df['CH4_ppm'].min())
                },
                'H2S_ppm': {
                    'mean': float(df['H2S_ppm'].mean()),
                    'max': float(df['H2S_ppm'].max()),
                    'min': float(df['H2S_ppm'].min())
                },
                'NOx_ppm': {
                    'mean': float(df['NOx_ppm'].mean()),
                    'max': float(df['NOx_ppm'].max()),
                    'min': float(df['NOx_ppm'].min())
                }
            },
            'warning_levels': df['warning_level'].value_counts().to_dict(),
            'severity_levels': df['leakage_severity'].value_counts().to_dict()
        }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get information about trained models"""
    try:
        model_info = {
            'leakage_detection': {
                'type': 'RandomForestClassifier',
                'features': 21,
                'trained': True
            },
            'severity_classification': {
                'type': 'RandomForestClassifier',
                'features': 21,
                'trained': True
            },
            'warning_level': {
                'type': 'RandomForestClassifier',
                'features': 21,
                'trained': True
            },
            'location_prediction': {
                'type': 'KMeans',
                'features': 21,
                'trained': True
            }
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    print("="*60)
    print("GAS LEAKAGE DETECTION API SERVER")
    print("="*60)
    
    # Initialize system
    initialize_system()
    
    print(f"Starting API server on {host}:{port}")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/predict - Single prediction")
    print("  POST /api/predict/batch - Batch predictions")
    print("  GET  /api/sensors/status - Sensor status")
    print("  POST /api/sensors/start - Start monitoring")
    print("  POST /api/sensors/stop - Stop monitoring")
    print("  GET  /api/predictions/latest - Latest predictions")
    print("  POST /api/environmental/calculate - Environmental impact")
    print("  POST /api/environmental/gas-impact - Gas impact")
    print("  POST /api/environmental/process-impact - Process impact")
    print("  POST /api/sensors/simulate-leakage - Simulate leakage")
    print("  GET  /api/data/historical - Historical data")
    print("  GET  /api/data/statistics - Data statistics")
    print("  GET  /api/models/info - Model information")
    print("="*60)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server(debug=True)
