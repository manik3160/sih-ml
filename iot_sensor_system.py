

import pandas as pd
import numpy as np
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import queue
import requests
from gas_leakage_predictor import GasLeakagePredictor

class IoTSensor:
    """
    Individual IoT sensor class for gas monitoring
    """
    
    def __init__(self, sensor_id: str, location: str, sensor_type: str):
        self.sensor_id = sensor_id
        self.location = location
        self.sensor_type = sensor_type
        self.is_active = True
        self.last_reading = None
        self.reading_history = []
        
    def read_sensor(self) -> Dict:
        """
        Simulate sensor reading with realistic variations
        """
        if not self.is_active:
            return None
            
        # Base readings for different sensor types
        base_readings = {
            'gas_multi': {
                'CO2_ppm': random.uniform(300, 800),
                'CO_ppm': random.uniform(5, 25),
                'SO2_ppm': random.uniform(0.5, 3.0),
                'CH4_ppm': random.uniform(0.5, 500),
                'H2S_ppm': random.uniform(0.1, 2.0),
                'NOx_ppm': random.uniform(10, 40)
            },
            'environmental': {
                'temperature_C': random.uniform(15, 45),
                'humidity_percent': random.uniform(40, 90),
                'pressure_kPa': random.uniform(95, 105),
                'wind_speed_ms': random.uniform(0.5, 8.0)
            },
            'vibration': {
                'vibration_level_mm_s': random.uniform(0.5, 5.0)
            }
        }
        
        # Generate reading based on sensor type
        reading = {
            'sensor_id': self.sensor_id,
            'location': self.location,
            'timestamp': datetime.now().isoformat(),
            'sensor_type': self.sensor_type
        }
        
        if self.sensor_type in base_readings:
            reading.update(base_readings[self.sensor_type])
        else:
            # Default gas sensor - include all readings
            reading.update(base_readings['gas_multi'])
            reading.update(base_readings['environmental'])
            reading.update(base_readings['vibration'])
        
        # Add some realistic noise and drift
        for key in reading:
            if isinstance(reading[key], (int, float)) and key not in ['sensor_id', 'timestamp', 'sensor_type']:
                noise = random.uniform(-0.1, 0.1) * reading[key]
                reading[key] = max(0, reading[key] + noise)
        
        self.last_reading = reading
        self.reading_history.append(reading)
        
        # Keep only last 100 readings
        if len(self.reading_history) > 100:
            self.reading_history = self.reading_history[-100:]
            
        return reading
    
    def simulate_leakage(self, gas_type: str, intensity: float = 1.0):
        """
        Simulate a gas leakage event
        """
        if self.last_reading and gas_type in self.last_reading:
            # Increase gas concentration significantly
            self.last_reading[gas_type] *= (1 + intensity * random.uniform(2, 10))
            
    def deactivate(self):
        """Deactivate sensor (simulate sensor failure)"""
        self.is_active = False
        
    def activate(self):
        """Activate sensor"""
        self.is_active = True

class IoTSensorNetwork:
    """
    Network of IoT sensors for comprehensive monitoring
    """
    
    def __init__(self, predictor: GasLeakagePredictor):
        self.predictor = predictor
        self.sensors = {}
        self.data_queue = queue.Queue()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
        # Initialize sensor network
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize the sensor network with various sensor types"""
        sensor_configs = [
            # Gas monitoring sensors
            {'id': 'GAS_001', 'location': 'Extraction_Zone_A', 'type': 'gas_multi'},
            {'id': 'GAS_002', 'location': 'Crushing_Station', 'type': 'gas_multi'},
            {'id': 'GAS_003', 'location': 'Smelter_Unit_1', 'type': 'gas_multi'},
            {'id': 'GAS_004', 'location': 'Refinery_Section_B', 'type': 'gas_multi'},
            {'id': 'GAS_005', 'location': 'Tailings_Area', 'type': 'gas_multi'},
            
            # Environmental sensors
            {'id': 'ENV_001', 'location': 'Extraction_Zone_A', 'type': 'environmental'},
            {'id': 'ENV_002', 'location': 'Smelter_Unit_1', 'type': 'environmental'},
            {'id': 'ENV_003', 'location': 'Refinery_Section_B', 'type': 'environmental'},
            
            # Vibration sensors
            {'id': 'VIB_001', 'location': 'Crushing_Station', 'type': 'vibration'},
            {'id': 'VIB_002', 'location': 'Smelter_Unit_1', 'type': 'vibration'},
        ]
        
        for config in sensor_configs:
            sensor = IoTSensor(
                sensor_id=config['id'],
                location=config['location'],
                sensor_type=config['type']
            )
            self.sensors[config['id']] = sensor
    
    def add_alert_callback(self, callback):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start continuous monitoring of all sensors
        
        Args:
            interval (float): Monitoring interval in seconds
        """
        if self.is_monitoring:
            print("Monitoring already active")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect readings from all active sensors
                readings = []
                for sensor in self.sensors.values():
                    if sensor.is_active:
                        reading = sensor.read_sensor()
                        if reading:
                            readings.append(reading)
                
                # Process readings
                if readings:
                    self._process_readings(readings)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _process_readings(self, readings: List[Dict]):
        """
        Process sensor readings and make predictions
        
        Args:
            readings (List[Dict]): List of sensor readings
        """
        # Combine readings by location
        location_data = self._combine_readings_by_location(readings)
        
        # Process each location
        for location, data in location_data.items():
            try:
                # Make prediction
                prediction = self.predictor.predict_leakage(data)
                
                # Store prediction with metadata
                prediction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'location': location,
                    'sensor_data': data,
                    'prediction': prediction
                }
                
                # Add to queue for further processing
                self.data_queue.put(prediction_data)
                
                # Check for alerts
                if prediction['leakage_detected'] or prediction['gas_warnings']:
                    self._trigger_alert(prediction_data)
                    
            except Exception as e:
                print(f"Error processing readings for {location}: {e}")
    
    def _combine_readings_by_location(self, readings: List[Dict]) -> Dict[str, Dict]:
        """
        Combine readings from multiple sensors at the same location
        
        Args:
            readings (List[Dict]): List of sensor readings
            
        Returns:
            Dict[str, Dict]: Combined readings by location
        """
        location_data = {}
        
        for reading in readings:
            location = reading['location']
            
            if location not in location_data:
                location_data[location] = {
                    'timestamp': reading['timestamp'],
                    'location_id': f"LOC_{location.replace('_', '')}",
                    'location_name': location,
                    'process_area': self._get_process_area(location),
                    'equipment_id': f"EQ_{reading['sensor_id']}",
                    'equipment_age_months': random.randint(12, 60),
                    'maintenance_days_ago': random.randint(0, 30),
                    'production_rate_percent': random.randint(70, 100),
                    # Initialize with default values
                    'CO2_ppm': 400, 'CO_ppm': 10, 'SO2_ppm': 1.0, 'CH4_ppm': 100,
                    'H2S_ppm': 0.5, 'NOx_ppm': 20, 'temperature_C': 25.0,
                    'humidity_percent': 50, 'pressure_kPa': 101.3, 'wind_speed_ms': 2.0,
                    'vibration_level_mm_s': 1.0
                }
            
            # Merge sensor data
            for key, value in reading.items():
                if key not in ['sensor_id', 'location', 'timestamp', 'sensor_type']:
                    if key in location_data[location]:
                        # Average multiple readings for the same parameter
                        if isinstance(value, (int, float)):
                            location_data[location][key] = (location_data[location][key] + value) / 2
                    else:
                        location_data[location][key] = value
        
        return location_data
    
    def _get_process_area(self, location: str) -> str:
        """Map location to process area"""
        mapping = {
            'Extraction_Zone_A': 'Open_Pit',
            'Crushing_Station': 'Processing',
            'Smelter_Unit_1': 'Smelting',
            'Refinery_Section_B': 'Refining',
            'Tailings_Area': 'Waste_Management'
        }
        return mapping.get(location, 'Unknown')
    
    def _trigger_alert(self, prediction_data: Dict):
        """
        Trigger alert for leakage detection or warnings
        
        Args:
            prediction_data (Dict): Prediction data with alerts
        """
        alert = {
            'timestamp': prediction_data['timestamp'],
            'location': prediction_data['location'],
            'severity': prediction_data['prediction']['severity'],
            'warning_level': prediction_data['prediction']['warning_level'],
            'leakage_probability': prediction_data['prediction']['leakage_probability'],
            'gas_warnings': prediction_data['prediction']['gas_warnings'],
            'recommendations': prediction_data['prediction']['recommendations']
        }
        
        # Call all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_latest_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get latest predictions from the queue
        
        Args:
            limit (int): Maximum number of predictions to return
            
        Returns:
            List[Dict]: Latest predictions
        """
        predictions = []
        count = 0
        
        while not self.data_queue.empty() and count < limit:
            try:
                prediction = self.data_queue.get_nowait()
                predictions.append(prediction)
                count += 1
            except queue.Empty:
                break
                
        return predictions
    
    def simulate_leakage_event(self, location: str, gas_type: str, intensity: float = 2.0):
        """
        Simulate a leakage event at a specific location
        
        Args:
            location (str): Location where leakage occurs
            gas_type (str): Type of gas leaking
            intensity (float): Intensity of the leakage
        """
        print(f"Simulating {gas_type} leakage at {location} with intensity {intensity}")
        
        # Find sensors at the location
        location_sensors = [s for s in self.sensors.values() if s.location == location]
        
        for sensor in location_sensors:
            if sensor.sensor_type == 'gas_multi':
                sensor.simulate_leakage(gas_type, intensity)
    
    def get_sensor_status(self) -> Dict:
        """
        Get status of all sensors
        
        Returns:
            Dict: Sensor status information
        """
        status = {
            'total_sensors': len(self.sensors),
            'active_sensors': sum(1 for s in self.sensors.values() if s.is_active),
            'inactive_sensors': sum(1 for s in self.sensors.values() if not s.is_active),
            'sensors': {}
        }
        
        for sensor_id, sensor in self.sensors.items():
            status['sensors'][sensor_id] = {
                'location': sensor.location,
                'type': sensor.sensor_type,
                'is_active': sensor.is_active,
                'last_reading_time': sensor.last_reading['timestamp'] if sensor.last_reading else None
            }
        
        return status

class AlertHandler:
    """
    Handles alerts and notifications
    """
    
    def __init__(self):
        self.alerts = []
        self.alert_thresholds = {
            'leakage_probability': 0.7,
            'severity_high': True,
            'gas_warning_count': 2
        }
    
    def handle_alert(self, alert: Dict):
        """
        Handle incoming alert
        
        Args:
            alert (Dict): Alert data
        """
        self.alerts.append(alert)
        
        # Determine alert priority
        priority = self._determine_priority(alert)
        
        # Log alert
        print(f"\n🚨 ALERT - Priority: {priority}")
        print(f"Location: {alert['location']}")
        print(f"Severity: {alert['severity']}")
        print(f"Warning Level: {alert['warning_level']}")
        print(f"Leakage Probability: {alert['leakage_probability']:.4f}")
        
        if alert['gas_warnings']:
            print("Gas Warnings:")
            for warning in alert['gas_warnings']:
                print(f"  - {warning['message']}")
        
        print("Recommendations:")
        for i, rec in enumerate(alert['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        # Send notifications based on priority
        if priority == 'HIGH':
            self._send_emergency_notification(alert)
        elif priority == 'MEDIUM':
            self._send_warning_notification(alert)
        else:
            self._send_info_notification(alert)
    
    def _determine_priority(self, alert: Dict) -> str:
        """Determine alert priority"""
        if (alert['leakage_probability'] > 0.8 or 
            alert['severity'] in ['high', 'critical'] or
            len(alert['gas_warnings']) >= 3):
            return 'HIGH'
        elif (alert['leakage_probability'] > 0.5 or 
              alert['severity'] in ['medium', 'high'] or
              len(alert['gas_warnings']) >= 1):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _send_emergency_notification(self, alert: Dict):
        """Send emergency notification"""
        print("🚨 EMERGENCY NOTIFICATION SENT!")
        print("   - Emergency services notified")
        print("   - Safety team alerted")
        print("   - Management notified")
    
    def _send_warning_notification(self, alert: Dict):
        """Send warning notification"""
        print("⚠️  WARNING NOTIFICATION SENT!")
        print("   - Safety team notified")
        print("   - Maintenance team alerted")
    
    def _send_info_notification(self, alert: Dict):
        """Send info notification"""
        print("ℹ️  INFO NOTIFICATION SENT!")
        print("   - Monitoring team notified")

def main():
    """Main function to demonstrate IoT sensor system"""
    print("="*60)
    print("IoT SENSOR INTEGRATION SYSTEM")
    print("Gas Leakage Detection for Mining & Metallurgy")
    print("="*60)
    
    # Initialize predictor
    print("Loading ML models...")
    predictor = GasLeakagePredictor()
    predictor.load_data()
    predictor.train_models()
    
    # Initialize IoT sensor network
    print("Initializing IoT sensor network...")
    sensor_network = IoTSensorNetwork(predictor)
    
    # Initialize alert handler
    alert_handler = AlertHandler()
    sensor_network.add_alert_callback(alert_handler.handle_alert)
    
    # Start monitoring
    print("Starting sensor monitoring...")
    sensor_network.start_monitoring(interval=3.0)
    
    try:
        # Monitor for 30 seconds
        print("\nMonitoring for 30 seconds...")
        time.sleep(10)
        
        # Simulate a leakage event
        print("\nSimulating CO2 leakage at Smelter_Unit_1...")
        sensor_network.simulate_leakage_event('Smelter_Unit_1', 'CO2_ppm', intensity=3.0)
        
        # Continue monitoring
        time.sleep(20)
        
        # Get latest predictions
        print("\nLatest predictions:")
        predictions = sensor_network.get_latest_predictions(limit=5)
        for i, pred in enumerate(predictions, 1):
            print(f"\nPrediction {i}:")
            print(f"  Location: {pred['location']}")
            print(f"  Leakage: {pred['prediction']['leakage_detected']}")
            print(f"  Probability: {pred['prediction']['leakage_probability']:.4f}")
            print(f"  Warning Level: {pred['prediction']['warning_level']}")
        
        # Get sensor status
        print("\nSensor Network Status:")
        status = sensor_network.get_sensor_status()
        print(f"Total Sensors: {status['total_sensors']}")
        print(f"Active Sensors: {status['active_sensors']}")
        print(f"Inactive Sensors: {status['inactive_sensors']}")
        
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        sensor_network.stop_monitoring()
        print("IoT sensor system stopped.")

if __name__ == "__main__":
    main()
