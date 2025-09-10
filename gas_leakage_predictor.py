"""
Gas Leakage Prediction System
AI-driven model for predicting gas leaks in mining and metallurgy operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

class GasLeakagePredictor:
    """
    ML model for predicting gas leakage in mining and metallurgy operations
    """
    
    def __init__(self, data_path='parsed_dataset.csv'):
        """
        Initialize the gas leakage prediction model
        
        Args:
            data_path (str): Path to the parsed dataset
        """
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            print(f"Loading dataset from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            
            # Clean and preprocess the data
            self.df = self.df.dropna()
            
            # Convert timestamp to datetime
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            print(f"Dataset loaded: {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def prepare_features(self):
        """Prepare features for ML models"""
        # Select features for gas leakage prediction
        feature_columns = [
            'CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm',
            'temperature_C', 'humidity_percent', 'pressure_kPa', 'wind_speed_ms',
            'vibration_level_mm_s', 'equipment_age_months', 'maintenance_days_ago',
            'production_rate_percent'
        ]
        
        # Categorical features
        categorical_features = [
            'location_id', 'location_name', 'process_area', 'equipment_id'
        ]
        
        # Prepare features
        X = self.df[feature_columns].copy()
        
        # Encode categorical features
        for col in categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(self.df[col].astype(str))
                X[col] = encoded_values
                self.label_encoders[col] = le
        
        # Target variables
        y_leakage = self.df['leakage_detected']
        y_severity = self.df['leakage_severity']
        y_warning = self.df['warning_level']
        y_location = self.df['predicted_leak_location']
        
        return X, y_leakage, y_severity, y_warning, y_location
    
    def train_models(self):
        """Train ML models for gas leakage prediction"""
        print("Training gas leakage prediction models...")
        
        X, y_leakage, y_severity, y_warning, y_location = self.prepare_features()
        
        # Split data
        X_train, X_test, y_leakage_train, y_leakage_test = train_test_split(
            X, y_leakage, test_size=0.2, random_state=42, stratify=y_leakage
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # 1. Leakage Detection Model
        print("Training leakage detection model...")
        rf_leakage = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_leakage.fit(X_train_scaled, y_leakage_train)
        self.models['leakage_detection'] = rf_leakage
        
        # 2. Severity Classification Model
        print("Training severity classification model...")
        severity_encoder = LabelEncoder()
        y_severity_encoded = severity_encoder.fit_transform(y_severity)
        y_severity_train_encoded = severity_encoder.transform(y_severity_train)
        y_severity_test_encoded = severity_encoder.transform(y_severity_test)
        self.label_encoders['severity'] = severity_encoder
        
        # Split severity data
        X_train_sev, X_test_sev, y_severity_train, y_severity_test = train_test_split(
            X, y_severity_encoded, test_size=0.2, random_state=42, stratify=y_severity_encoded
        )
        
        # Scale severity features
        X_train_sev_scaled = scaler.transform(X_train_sev)
        X_test_sev_scaled = scaler.transform(X_test_sev)
        
        rf_severity = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_severity.fit(X_train_sev_scaled, y_severity_train)
        self.models['severity_classification'] = rf_severity
        
        # 3. Warning Level Model
        print("Training warning level model...")
        warning_encoder = LabelEncoder()
        y_warning_encoded = warning_encoder.fit_transform(y_warning)
        y_warning_train_encoded = warning_encoder.transform(y_warning_train)
        y_warning_test_encoded = warning_encoder.transform(y_warning_test)
        self.label_encoders['warning'] = warning_encoder
        
        # Split warning data
        X_train_warn, X_test_warn, y_warning_train, y_warning_test = train_test_split(
            X, y_warning_encoded, test_size=0.2, random_state=42, stratify=y_warning_encoded
        )
        
        # Scale warning features
        X_train_warn_scaled = scaler.transform(X_train_warn)
        X_test_warn_scaled = scaler.transform(X_test_warn)
        
        rf_warning = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_warning.fit(X_train_warn_scaled, y_warning_train)
        self.models['warning_level'] = rf_warning
        
        # 4. Location Prediction Model
        print("Training location prediction model...")
        location_encoder = LabelEncoder()
        y_location_encoded = location_encoder.fit_transform(y_location)
        y_location_train_encoded = location_encoder.transform(y_location_train)
        y_location_test_encoded = location_encoder.transform(y_location_test)
        self.label_encoders['location'] = location_encoder
        
        # Split location data
        X_train_loc, X_test_loc, y_location_train, y_location_test = train_test_split(
            X, y_location_encoded, test_size=0.2, random_state=42, stratify=y_location_encoded
        )
        
        # Scale location features
        X_train_loc_scaled = scaler.transform(X_train_loc)
        X_test_loc_scaled = scaler.transform(X_test_loc)
        
        rf_location = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_location.fit(X_train_loc_scaled, y_location_train)
        self.models['location_prediction'] = rf_location
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_leakage_test, X_test_sev_scaled, y_severity_test_encoded, 
                            X_test_warn_scaled, y_warning_test_encoded, X_test_loc_scaled, y_location_test_encoded)
        
        print("All models trained successfully!")
    
    def _evaluate_models(self, X_test, y_leakage_test, X_test_sev, y_severity_test, 
                        X_test_warn, y_warning_test, X_test_loc, y_location_test):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Leakage Detection
        y_leakage_pred = self.models['leakage_detection'].predict(X_test)
        print("\nLeakage Detection Model:")
        print(f"Accuracy: {accuracy_score(y_leakage_test, y_leakage_pred):.4f}")
        
        # Severity Classification
        y_severity_pred = self.models['severity_classification'].predict(X_test_sev)
        print("\nSeverity Classification Model:")
        print(f"Accuracy: {accuracy_score(y_severity_test, y_severity_pred):.4f}")
        
        # Warning Level
        y_warning_pred = self.models['warning_level'].predict(X_test_warn)
        print("\nWarning Level Model:")
        print(f"Accuracy: {accuracy_score(y_warning_test, y_warning_pred):.4f}")
        
        # Location Prediction
        y_location_pred = self.models['location_prediction'].predict(X_test_loc)
        print("\nLocation Prediction Model:")
        print(f"Accuracy: {accuracy_score(y_location_test, y_location_pred):.4f}")
    
    def predict_leakage(self, sensor_data):
        """
        Predict gas leakage for given sensor data
        
        Args:
            sensor_data (dict): Sensor readings and environmental data
            
        Returns:
            dict: Leakage prediction results
        """
        # Convert to DataFrame
        df_new = pd.DataFrame([sensor_data])
        
        # Prepare features
        feature_columns = [
            'CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm',
            'temperature_C', 'humidity_percent', 'pressure_kPa', 'wind_speed_ms',
            'vibration_level_mm_s', 'equipment_age_months', 'maintenance_days_ago',
            'production_rate_percent'
        ]
        
        X_new = df_new[feature_columns].copy()
        
        # Encode categorical features
        for col in ['location_id', 'location_name', 'process_area', 'equipment_id']:
            if col in df_new.columns and col in self.label_encoders:
                try:
                    X_new[col] = self.label_encoders[col].transform(df_new[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels by using a default value
                    print(f"Warning: Unseen label in {col}, using default value")
                    X_new[col] = 0  # Default to first encoded value
        
        # Scale features
        X_new_scaled = self.scalers['main'].transform(X_new)
        
        # Make predictions
        leakage_pred = self.models['leakage_detection'].predict(X_new_scaled)[0]
        leakage_prob = self.models['leakage_detection'].predict_proba(X_new_scaled)[0]
        
        severity_pred = self.models['severity_classification'].predict(X_new_scaled)[0]
        severity_name = self.label_encoders['severity'].inverse_transform([severity_pred])[0]
        
        warning_pred = self.models['warning_level'].predict(X_new_scaled)[0]
        warning_name = self.label_encoders['warning'].inverse_transform([warning_pred])[0]
        
        location_pred = self.models['location_prediction'].predict(X_new_scaled)[0]
        location_name = self.label_encoders['location'].inverse_transform([location_pred])[0]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(leakage_pred, severity_name, warning_name, sensor_data)
        
        return {
            'leakage_detected': bool(leakage_pred),
            'leakage_probability': float(leakage_prob[1]) if len(leakage_prob) > 1 else 0.0,
            'severity_level': severity_name,
            'warning_level': warning_name,
            'predicted_location': location_name,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_recommendations(self, leakage_pred, severity, warning, sensor_data):
        """Generate recommendations based on predictions"""
        recommendations = []
        
        if leakage_pred:
            if severity == 'critical':
                recommendations.append("🚨 CRITICAL: Immediate evacuation required")
                recommendations.append("🚨 EMERGENCY: Contact emergency services immediately")
            elif severity == 'high':
                recommendations.append("⚠️ HIGH: Evacuate area and shut down operations")
                recommendations.append("⚠️ URGENT: Contact safety team immediately")
            elif severity == 'medium':
                recommendations.append("⚠️ MEDIUM: Reduce operations and increase monitoring")
                recommendations.append("⚠️ CAUTION: Check equipment and ventilation")
            else:
                recommendations.append("ℹ️ LOW: Monitor closely and check for leaks")
                recommendations.append("ℹ️ INFO: Normal operations with increased awareness")
        else:
            recommendations.append("✅ SAFE: No leakage detected")
            recommendations.append("✅ NORMAL: Continue regular operations")
        
        # Gas-specific recommendations
        if sensor_data.get('CO_ppm', 0) > 50:
            recommendations.append("⚠️ CO DETECTED: Carbon monoxide levels elevated")
        if sensor_data.get('H2S_ppm', 0) > 10:
            recommendations.append("⚠️ H2S DETECTED: Hydrogen sulfide levels elevated")
        if sensor_data.get('CH4_ppm', 0) > 1000:
            recommendations.append("⚠️ CH4 DETECTED: Methane levels elevated")
        
        return recommendations
    
    def save_models(self, model_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/{name}_scaler.pkl")
        
        # Save encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{model_dir}/{name}_encoder.pkl")
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load trained models"""
        import os
        
        # Load models
        for name in ['leakage_detection', 'severity_classification', 'warning_level', 'location_prediction']:
            model_path = f"{model_dir}/{name}_model.pkl"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load scalers
        scaler_path = f"{model_dir}/main_scaler.pkl"
        if os.path.exists(scaler_path):
            self.scalers['main'] = joblib.load(scaler_path)
        
        # Load encoders
        for name in ['severity', 'warning', 'location', 'location_id', 'location_name', 'process_area', 'equipment_id']:
            encoder_path = f"{model_dir}/{name}_encoder.pkl"
            if os.path.exists(encoder_path):
                self.label_encoders[name] = joblib.load(encoder_path)
        
        print(f"Models loaded from {model_dir}/")
    
    def create_visualizations(self):
        """Create visualizations for gas leakage analysis"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gas Leakage Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Gas concentrations over time
        if 'timestamp' in self.df.columns:
            axes[0, 0].plot(self.df['timestamp'], self.df['CO2_ppm'], label='CO2', alpha=0.7)
            axes[0, 0].plot(self.df['timestamp'], self.df['CO_ppm'], label='CO', alpha=0.7)
            axes[0, 0].plot(self.df['timestamp'], self.df['CH4_ppm'], label='CH4', alpha=0.7)
            axes[0, 0].set_title('Gas Concentrations Over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Concentration (ppm)')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Leakage detection distribution
        leakage_counts = self.df['leakage_detected'].value_counts()
        axes[0, 1].pie(leakage_counts.values, labels=['No Leak', 'Leak'], autopct='%1.1f%%', colors=['green', 'red'])
        axes[0, 1].set_title('Leakage Detection Distribution')
        
        # Severity levels
        severity_counts = self.df['leakage_severity'].value_counts()
        axes[0, 2].bar(severity_counts.index, severity_counts.values, color=['green', 'yellow', 'orange', 'red'])
        axes[0, 2].set_title('Leakage Severity Levels')
        axes[0, 2].set_ylabel('Count')
        
        # Environmental conditions
        axes[1, 0].scatter(self.df['temperature_C'], self.df['humidity_percent'], 
                          c=self.df['leakage_detected'], cmap='RdYlGn', alpha=0.6)
        axes[1, 0].set_title('Environmental Conditions vs Leakage')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Humidity (%)')
        
        # Equipment age vs leakage
        equipment_leakage = self.df.groupby('equipment_age_months')['leakage_detected'].mean()
        axes[1, 1].plot(equipment_leakage.index, equipment_leakage.values, marker='o')
        axes[1, 1].set_title('Equipment Age vs Leakage Rate')
        axes[1, 1].set_xlabel('Equipment Age (months)')
        axes[1, 1].set_ylabel('Leakage Rate')
        
        # Location-based leakage
        location_leakage = self.df.groupby('location_name')['leakage_detected'].sum()
        axes[1, 2].bar(range(len(location_leakage)), location_leakage.values)
        axes[1, 2].set_title('Leakage by Location')
        axes[1, 2].set_xticks(range(len(location_leakage)))
        axes[1, 2].set_xticklabels(location_leakage.index, rotation=45)
        axes[1, 2].set_ylabel('Leakage Count')
        
        plt.tight_layout()
        plt.savefig('gas_leakage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate the gas leakage prediction model"""
    print("="*60)
    print("GAS LEAKAGE PREDICTION SYSTEM")
    print("Mining and Metallurgy Operations")
    print("="*60)
    
    # Initialize model
    model = GasLeakagePredictor()
    
    # Load data
    print("Loading data...")
    df = model.load_data()
    
    if df is None:
        print("Error: Could not load data")
        return
    
    # Train models
    print("\nTraining models...")
    model.train_models()
    
    # Save models
    print("\nSaving models...")
    model.save_models()
    
    # Create visualizations
    print("\nCreating visualizations...")
    model.create_visualizations()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    sample_sensor_data = {
        'CO2_ppm': 450,
        'CO_ppm': 15,
        'SO2_ppm': 5,
        'CH4_ppm': 1200,
        'H2S_ppm': 8,
        'NOx_ppm': 25,
        'temperature_C': 28.5,
        'humidity_percent': 65,
        'pressure_kPa': 101.3,
        'wind_speed_ms': 3.2,
        'vibration_level_mm_s': 2.1,
        'equipment_age_months': 18,
        'maintenance_days_ago': 5,
        'production_rate_percent': 85,
        'location_id': 'LOC_001',
        'location_name': 'Extraction_Zone_A',
        'process_area': 'mining',
        'equipment_id': 'EQ_001'
    }
    
    prediction = model.predict_leakage(sample_sensor_data)
    
    print(f"Leakage Detected: {prediction['leakage_detected']}")
    print(f"Leakage Probability: {prediction['leakage_probability']:.3f}")
    print(f"Severity Level: {prediction['severity_level']}")
    print(f"Warning Level: {prediction['warning_level']}")
    print(f"Predicted Location: {prediction['predicted_location']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(prediction['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*60)
    print("GAS LEAKAGE PREDICTION SYSTEM READY")
    print("="*60)

if __name__ == "__main__":
    main()