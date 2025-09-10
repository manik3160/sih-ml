

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
from gas_leakage_predictor import GasLeakagePredictor
from iot_sensor_system import IoTSensorNetwork, AlertHandler
from lca_environmental_calculator import LCAEnvironmentalCalculator

class GasLeakageModelManager:
    """
    Simplified model manager for easy frontend integration
    """
    
    def __init__(self, data_path='parsed_dataset.csv'):
        """
        Initialize the model manager
        
        Args:
            data_path (str): Path to the parsed dataset
        """
        self.data_path = data_path
        self.predictor = None
        self.sensor_network = None
        self.environmental_calculator = None
        self.alert_handler = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all models and components"""
        try:
            print("Initializing Gas Leakage Detection System...")
            
            # Initialize predictor
            self.predictor = GasLeakagePredictor(self.data_path)
            self.predictor.load_data()
            self.predictor.train_models()
            
            # Initialize sensor network
            self.sensor_network = IoTSensorNetwork(self.predictor)
            
            # Initialize environmental calculator
            self.environmental_calculator = LCAEnvironmentalCalculator()
            
            # Initialize alert handler
            self.alert_handler = AlertHandler()
            self.sensor_network.add_alert_callback(self.alert_handler.handle_alert)
            
            self.is_initialized = True
            print("System initialized successfully!")
            
            return True
            
        except Exception as e:
            print(f"Error initializing system: {e}")
            return False
    
    def predict_single(self, sensor_data: Dict) -> Dict:
        """
        Make a single prediction for gas leakage
        
        Args:
            sensor_data (Dict): Sensor data dictionary
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            prediction = self.predictor.predict_leakage(sensor_data)
            return {
                'success': True,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, sensor_data_list: List[Dict]) -> Dict:
        """
        Make batch predictions for multiple sensor readings
        
        Args:
            sensor_data_list (List[Dict]): List of sensor data dictionaries
            
        Returns:
            Dict: Batch prediction results
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            predictions = []
            for sensor_data in sensor_data_list:
                prediction = self.predictor.predict_leakage(sensor_data)
                predictions.append({
                    'sensor_data': sensor_data,
                    'prediction': prediction
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'count': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_monitoring(self, interval: float = 5.0) -> Dict:
        """
        Start real-time monitoring
        
        Args:
            interval (float): Monitoring interval in seconds
            
        Returns:
            Dict: Status result
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            self.sensor_network.start_monitoring(interval=interval)
            return {
                'success': True,
                'message': f'Monitoring started with {interval}s interval',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def stop_monitoring(self) -> Dict:
        """
        Stop real-time monitoring
        
        Returns:
            Dict: Status result
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            self.sensor_network.stop_monitoring()
            return {
                'success': True,
                'message': 'Monitoring stopped',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_latest_predictions(self, limit: int = 10) -> Dict:
        """
        Get latest predictions from monitoring
        
        Args:
            limit (int): Maximum number of predictions to return
            
        Returns:
            Dict: Latest predictions
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            predictions = self.sensor_network.get_latest_predictions(limit=limit)
            return {
                'success': True,
                'predictions': predictions,
                'count': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_sensor_status(self) -> Dict:
        """
        Get status of all sensors
        
        Returns:
            Dict: Sensor status information
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            status = self.sensor_network.get_sensor_status()
            return {
                'success': True,
                'sensor_status': status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_environmental_impact(self, prediction_data: Dict, 
                                     process_data: Dict, 
                                     historical_data: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate environmental impact
        
        Args:
            prediction_data (Dict): Prediction data
            process_data (Dict): Process data
            historical_data (Optional[List[Dict]]): Historical data
            
        Returns:
            Dict: Environmental impact report
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            # Convert historical data to DataFrame if provided
            if historical_data:
                historical_df = pd.DataFrame(historical_data)
                if 'timestamp' in historical_df.columns:
                    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            else:
                historical_df = pd.DataFrame()
            
            # Calculate environmental impact
            report = self.environmental_calculator.create_environmental_report(
                prediction_data, process_data, historical_df
            )
            
            return {
                'success': True,
                'environmental_report': report,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_gas_impact(self, gas_concentrations: Dict) -> Dict:
        """
        Calculate gas emissions impact
        
        Args:
            gas_concentrations (Dict): Gas concentration data
            
        Returns:
            Dict: Gas impact calculations
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            gas_impacts = self.environmental_calculator.calculate_gas_emissions_impact(
                gas_concentrations
            )
            
            return {
                'success': True,
                'gas_impacts': gas_impacts,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_process_impact(self, process_data: Dict) -> Dict:
        """
        Calculate process environmental impact
        
        Args:
            process_data (Dict): Process data
            
        Returns:
            Dict: Process impact calculations
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            process_impacts = self.environmental_calculator.calculate_process_environmental_impact(
                process_data
            )
            
            return {
                'success': True,
                'process_impacts': process_impacts,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def simulate_leakage(self, location: str, gas_type: str = 'CO2_ppm', 
                        intensity: float = 2.0) -> Dict:
        """
        Simulate a gas leakage event
        
        Args:
            location (str): Location where leakage occurs
            gas_type (str): Type of gas leaking
            intensity (float): Intensity of the leakage
            
        Returns:
            Dict: Simulation result
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            self.sensor_network.simulate_leakage_event(location, gas_type, intensity)
            return {
                'success': True,
                'message': f'Simulated {gas_type} leakage at {location} with intensity {intensity}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_historical_data(self) -> Dict:
        """
        Get historical data for analysis
        
        Returns:
            Dict: Historical data
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            df = self.predictor.df
            historical_data = df.to_dict('records')
            
            return {
                'success': True,
                'historical_data': historical_data,
                'count': len(historical_data),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistical analysis of the dataset
        
        Returns:
            Dict: Data statistics
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            df = self.predictor.df
            
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
            
            return {
                'success': True,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict:
        """
        Get information about trained models
        
        Returns:
            Dict: Model information
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
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
            
            return {
                'success': True,
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Example usage
def main():
    """Example usage of the Model Manager"""
    print("="*60)
    print("GAS LEAKAGE DETECTION MODEL MANAGER")
    print("="*60)
    
    # Initialize model manager
    manager = GasLeakageModelManager()
    
    # Initialize system
    if not manager.initialize():
        print("Failed to initialize system")
        return
    
    # Example: Single prediction
    print("\n--- Single Prediction Example ---")
    sample_sensor_data = {
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
    
    prediction_result = manager.predict_single(sample_sensor_data)
    print(f"Prediction Result: {json.dumps(prediction_result, indent=2)}")
    
    # Example: Environmental impact calculation
    print("\n--- Environmental Impact Example ---")
    gas_concentrations = {
        'CO2_ppm': 1200,
        'CO_ppm': 15,
        'SO2_ppm': 2.5,
        'CH4_ppm': 800,
        'H2S_ppm': 5,
        'NOx_ppm': 30
    }
    
    gas_impact_result = manager.calculate_gas_impact(gas_concentrations)
    print(f"Gas Impact Result: {json.dumps(gas_impact_result, indent=2)}")
    
    # Example: Process impact calculation
    print("\n--- Process Impact Example ---")
    process_data = {
        'Open_Pit': 1000,
        'Processing': 800,
        'Smelting': 200,
        'Refining': 150,
        'Waste_Management': 500
    }
    
    process_impact_result = manager.calculate_process_impact(process_data)
    print(f"Process Impact Result: {json.dumps(process_impact_result, indent=2)}")
    
    # Example: Get statistics
    print("\n--- Data Statistics Example ---")
    stats_result = manager.get_data_statistics()
    print(f"Statistics Result: {json.dumps(stats_result, indent=2)}")
    
    print("\n" + "="*60)
    print("MODEL MANAGER READY FOR FRONTEND INTEGRATION")
    print("="*60)

if __name__ == "__main__":
    main()
