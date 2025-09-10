"""
Example Usage of Gas Leakage Detection System
Demonstrates how to use the model manager for frontend integration
"""

from model_manager import GasLeakageModelManager
import json

def main():
    """Example usage of the Gas Leakage Detection System"""
    print("="*60)
    print("GAS LEAKAGE DETECTION SYSTEM - EXAMPLE USAGE")
    print("="*60)
    
    # Initialize the model manager
    print("Initializing model manager...")
    manager = GasLeakageModelManager()
    
    # Initialize the system
    if not manager.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Example 1: Single Prediction
    print("\n" + "="*50)
    print("EXAMPLE 1: SINGLE PREDICTION")
    print("="*50)
    
    sample_sensor_data = {
        'timestamp': '2024-01-15 14:30:00',
        'location_id': 'LOC001',
        'location_name': 'Extraction_Zone_A',
        'process_area': 'Open_Pit',
        'equipment_id': 'EQ_EX_001',
        'CO2_ppm': 1200,  # Above threshold
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
    
    if prediction_result['success']:
        pred = prediction_result['prediction']
        print(f"🔍 Leakage Detected: {pred['leakage_detected']}")
        print(f"📊 Leakage Probability: {pred['leakage_probability']:.4f}")
        print(f"⚠️  Severity: {pred['severity']}")
        print(f"🚨 Warning Level: {pred['warning_level']}")
        print(f"📍 Predicted Location: {pred['predicted_location']}")
        
        if pred['gas_warnings']:
            print("\nGas Warnings:")
            for warning in pred['gas_warnings']:
                print(f"  - {warning['message']} (Severity: {warning['severity']})")
        
        print("\nRecommendations:")
        for i, rec in enumerate(pred['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    else:
        print(f"❌ Prediction failed: {prediction_result['error']}")
    
    # Example 2: Environmental Impact Calculation
    print("\n" + "="*50)
    print("EXAMPLE 2: ENVIRONMENTAL IMPACT CALCULATION")
    print("="*50)
    
    gas_concentrations = {
        'CO2_ppm': 1200,
        'CO_ppm': 15,
        'SO2_ppm': 2.5,
        'CH4_ppm': 800,
        'H2S_ppm': 5,
        'NOx_ppm': 30
    }
    
    gas_impact_result = manager.calculate_gas_impact(gas_concentrations)
    
    if gas_impact_result['success']:
        impacts = gas_impact_result['gas_impacts']
        print(f"🌍 Total CO2 Equivalent: {impacts['total_co2e']:.4f} kg")
        print(f"📈 Global Warming Impact: {impacts['impact_categories']['global_warming']:.4f}")
        print(f"🌧️  Acidification Impact: {impacts['impact_categories']['acidification']:.4f}")
        print(f"🌫️  Smog Formation Impact: {impacts['impact_categories']['smog_formation']:.4f}")
        
        print("\nIndividual Gas Impacts:")
        for gas, impact in impacts['individual_impacts'].items():
            print(f"  {gas}: {impact['co2e']:.4f} kg CO2e")
    else:
        print(f"❌ Gas impact calculation failed: {gas_impact_result['error']}")
    
    # Example 3: Process Environmental Impact
    print("\n" + "="*50)
    print("EXAMPLE 3: PROCESS ENVIRONMENTAL IMPACT")
    print("="*50)
    
    process_data = {
        'Open_Pit': 1000,      # kg ore
        'Processing': 800,     # kg processed material
        'Smelting': 200,       # kg metal
        'Refining': 150,       # kg refined metal
        'Waste_Management': 500 # kg waste
    }
    
    process_impact_result = manager.calculate_process_impact(process_data)
    
    if process_impact_result['success']:
        impacts = process_impact_result['process_impacts']
        print(f"🌍 Total Carbon Footprint: {impacts['carbon_footprint']:.2f} kg CO2e")
        print(f"⚡ Total Energy Consumption: {impacts['energy_consumption']:.2f} kWh")
        print(f"💧 Total Water Consumption: {impacts['water_consumption']:.2f} L")
        
        print("\nProcess-wise Impacts:")
        for process, impact in impacts['process_impacts'].items():
            print(f"  {process}:")
            print(f"    - Carbon: {impact['carbon_footprint']:.2f} kg CO2e")
            print(f"    - Energy: {impact['energy_consumption']:.2f} kWh")
            print(f"    - Water: {impact['water_consumption']:.2f} L")
    else:
        print(f"❌ Process impact calculation failed: {process_impact_result['error']}")
    
    # Example 4: Data Statistics
    print("\n" + "="*50)
    print("EXAMPLE 4: DATA STATISTICS")
    print("="*50)
    
    stats_result = manager.get_data_statistics()
    
    if stats_result['success']:
        stats = stats_result['statistics']
        print(f"📊 Total Records: {stats['total_records']}")
        print(f"🚨 Leakage Incidents: {stats['leakage_incidents']}")
        print(f"📈 Leakage Rate: {stats['leakage_rate']:.4f}")
        print(f"🏭 Locations: {stats['locations']}")
        print(f"⚙️  Process Areas: {stats['process_areas']}")
        
        print(f"\n📅 Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        
        print("\nGas Statistics:")
        for gas, gas_stats in stats['gas_statistics'].items():
            print(f"  {gas}: Mean={gas_stats['mean']:.2f}, Max={gas_stats['max']:.2f}, Min={gas_stats['min']:.2f}")
        
        print(f"\nWarning Levels: {stats['warning_levels']}")
        print(f"Severity Levels: {stats['severity_levels']}")
    else:
        print(f"❌ Statistics calculation failed: {stats_result['error']}")
    
    # Example 5: Sensor Status
    print("\n" + "="*50)
    print("EXAMPLE 5: SENSOR STATUS")
    print("="*50)
    
    sensor_status_result = manager.get_sensor_status()
    
    if sensor_status_result['success']:
        status = sensor_status_result['sensor_status']
        print(f"📡 Total Sensors: {status['total_sensors']}")
        print(f"✅ Active Sensors: {status['active_sensors']}")
        print(f"❌ Inactive Sensors: {status['inactive_sensors']}")
        
        print("\nSensor Details:")
        for sensor_id, sensor_info in status['sensors'].items():
            status_icon = "✅" if sensor_info['is_active'] else "❌"
            print(f"  {status_icon} {sensor_id}: {sensor_info['location']} ({sensor_info['type']})")
    else:
        print(f"❌ Sensor status retrieval failed: {sensor_status_result['error']}")
    
    # Example 6: Model Information
    print("\n" + "="*50)
    print("EXAMPLE 6: MODEL INFORMATION")
    print("="*50)
    
    model_info_result = manager.get_model_info()
    
    if model_info_result['success']:
        models = model_info_result['model_info']
        print("🤖 Trained Models:")
        for model_name, model_info in models.items():
            print(f"  - {model_name}: {model_info['type']} ({model_info['features']} features)")
    else:
        print(f"❌ Model info retrieval failed: {model_info_result['error']}")
    
    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("🚀 SYSTEM READY FOR FRONTEND INTEGRATION")
    print("="*60)
    
    print("\n📋 INTEGRATION NOTES:")
    print("1. Use 'manager.predict_single(sensor_data)' for single predictions")
    print("2. Use 'manager.predict_batch(sensor_data_list)' for batch predictions")
    print("3. Use 'manager.calculate_gas_impact(gas_data)' for environmental calculations")
    print("4. Use 'manager.start_monitoring()' and 'manager.stop_monitoring()' for real-time monitoring")
    print("5. Use 'manager.get_latest_predictions()' to get monitoring results")
    print("6. All methods return JSON-serializable dictionaries for easy frontend integration")

if __name__ == "__main__":
    main()
