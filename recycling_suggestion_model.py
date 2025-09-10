import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class RecyclingSuggestionModel:
    """
    ML model for suggesting recycling strategies based on leftover materials
    """
    
    def __init__(self, mining_dataset_path='ai_lca_mining_dataset_150.csv'):
        self.mining_dataset_path = mining_dataset_path
        self.df = None
        self.mining_df = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.recycling_database = self._initialize_recycling_database()
        
    def _initialize_recycling_database(self):
        """Initialize recycling possibilities database"""
        return {
            'steel_scrap': {
                'recyclable_products': ['new_steel', 'rebar', 'automotive_parts', 'construction_materials'],
                'recycling_efficiency': 0.95,
                'energy_savings': 0.74,
                'co2_reduction': 0.58,
                'market_value': 0.8
            },
            'aluminum_scrap': {
                'recyclable_products': ['new_aluminum', 'cans', 'foil', 'automotive_parts'],
                'recycling_efficiency': 0.92,
                'energy_savings': 0.95,
                'co2_reduction': 0.92,
                'market_value': 0.9
            },
            'copper_scrap': {
                'recyclable_products': ['electrical_wire', 'plumbing', 'electronics', 'jewelry'],
                'recycling_efficiency': 0.88,
                'energy_savings': 0.85,
                'co2_reduction': 0.65,
                'market_value': 0.95
            },
            'iron_ore_tailings': {
                'recyclable_products': ['cement_additive', 'concrete_aggregate', 'road_base'],
                'recycling_efficiency': 0.70,
                'energy_savings': 0.60,
                'co2_reduction': 0.40,
                'market_value': 0.3
            },
            'slag': {
                'recyclable_products': ['cement', 'concrete_aggregate', 'road_construction'],
                'recycling_efficiency': 0.75,
                'energy_savings': 0.50,
                'co2_reduction': 0.45,
                'market_value': 0.4
            },
            'dust_particles': {
                'recyclable_products': ['paint_pigments', 'ceramics', 'filters'],
                'recycling_efficiency': 0.60,
                'energy_savings': 0.30,
                'co2_reduction': 0.25,
                'market_value': 0.2
            }
        }
    
    def load_mining_dataset(self):
        """Load and preprocess the mining LCA dataset"""
        try:
            print(f"Loading mining dataset from {self.mining_dataset_path}...")
            self.mining_df = pd.read_csv(self.mining_dataset_path)
            
            # Clean and preprocess the data
            self.mining_df['Material_Type'] = self.mining_df['Material_Type'].str.lower().str.replace(' ', '_')
            self.mining_df['Source_Process'] = self.mining_df['Source_Process'].str.lower().str.replace(' ', '_')
            self.mining_df['Suggested_Reuse'] = self.mining_df['Suggested_Reuse'].str.lower()
            
            # Create additional features from your dataset
            self.mining_df['recycling_feasibility'] = np.where(
                self.mining_df['Recyclable'] == 'Yes', 
                np.random.uniform(0.7, 0.95, len(self.mining_df)),
                np.random.uniform(0.2, 0.6, len(self.mining_df))
            )
            
            # Map toxicity to priority levels
            toxicity_mapping = {1: 'low', 2: 'low', 3: 'medium', 4: 'medium', 5: 'high', 6: 'high', 7: 'critical', 8: 'critical', 9: 'critical', 10: 'critical'}
            self.mining_df['recycling_priority'] = self.mining_df['Toxicity_Score'].map(toxicity_mapping)
            
            # Calculate environmental benefit based on carbon footprint
            max_carbon = self.mining_df['Carbon_Footprint_kgCO2'].max()
            self.mining_df['environmental_benefit'] = 1 - (self.mining_df['Carbon_Footprint_kgCO2'] / max_carbon)
            
            # Calculate economic viability based on purity and energy content
            self.mining_df['economic_viability'] = (
                self.mining_df['Purity_%'] / 100 * 0.6 + 
                (self.mining_df['Energy_Content_MJ'] / self.mining_df['Energy_Content_MJ'].max()) * 0.4
            )
            
            # Processing complexity based on contaminants and toxicity
            self.mining_df['processing_complexity'] = np.where(
                self.mining_df['Toxicity_Score'] > 7, 5,
                np.where(self.mining_df['Toxicity_Score'] > 5, 4,
                        np.where(self.mining_df['Toxicity_Score'] > 3, 3,
                                np.where(self.mining_df['Toxicity_Score'] > 1, 2, 1)))
            )
            
            # Map suggested reuse to product categories
            reuse_mapping = {
                'reuse in battery production': 'batteries',
                'reprocess to recover lead': 'lead_products',
                're-smelt for rebar': 'construction_materials',
                'compress into briquettes for reuse': 'briquettes',
                'safe disposal required': 'landfill_disposal',
                'reuse in cement production': 'cement',
                'reuse in road construction': 'road_construction',
                'reuse in steel production': 'new_steel',
                'reuse in aluminum production': 'new_aluminum'
            }
            
            self.mining_df['suggested_product'] = self.mining_df['Suggested_Reuse'].map(reuse_mapping)
            self.mining_df['suggested_product'] = self.mining_df['suggested_product'].fillna('landfill_disposal')
            
            print(f"Mining dataset loaded: {len(self.mining_df)} records")
            print(f"Material types: {self.mining_df['Material_Type'].nunique()}")
            print(f"Recyclable materials: {self.mining_df['Recyclable'].value_counts()['Yes']}")
            
            return self.mining_df
            
        except Exception as e:
            print(f"Error loading mining dataset: {e}")
            return None
    
    def create_sample_datasets(self):
        """Create comprehensive sample datasets for training"""
        
        # Dataset 1: Leftover Materials Inventory
        leftover_data = {
            'material_id': [f'MAT_{i:03d}' for i in range(1, 1001)],
            'material_type': np.random.choice([
                'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
                'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap',
                'titanium_scrap', 'gold_scrap', 'silver_scrap', 'platinum_scrap'
            ], 1000),
            'quantity_kg': np.random.exponential(1000, 1000),
            'purity_percent': np.random.uniform(60, 99, 1000),
            'contamination_level': np.random.choice(['low', 'medium', 'high'], 1000, p=[0.6, 0.3, 0.1]),
            'particle_size_mm': np.random.uniform(0.1, 50, 1000),
            'oxidation_level': np.random.uniform(0, 30, 1000),
            'moisture_content': np.random.uniform(0, 15, 1000),
            'collection_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'source_process': np.random.choice([
                'mining', 'smelting', 'refining', 'casting', 'machining',
                'welding', 'cutting', 'grinding', 'polishing'
            ], 1000),
            'location': np.random.choice([
                'Extraction_Zone_A', 'Smelter_Unit_1', 'Refinery_Section_B',
                'Casting_Facility', 'Machining_Shop', 'Storage_Yard'
            ], 1000),
            'storage_conditions': np.random.choice([
                'indoor_dry', 'indoor_humid', 'outdoor_covered', 'outdoor_exposed'
            ], 1000),
            'hazardous_classification': np.random.choice([
                'non_hazardous', 'low_hazard', 'moderate_hazard', 'high_hazard'
            ], 1000, p=[0.7, 0.2, 0.08, 0.02])
        }
        
        # Dataset 2: Market Demand and Pricing
        market_data = {
            'product_type': [
                'new_steel', 'rebar', 'automotive_parts', 'construction_materials',
                'new_aluminum', 'cans', 'foil', 'electrical_wire', 'plumbing',
                'electronics', 'jewelry', 'cement_additive', 'concrete_aggregate',
                'road_base', 'cement', 'road_construction', 'paint_pigments',
                'ceramics', 'filters'
            ] * 50,
            'market_demand_trend': np.random.choice(['increasing', 'stable', 'decreasing'], 950, p=[0.4, 0.4, 0.2]),
            'price_per_kg': np.random.uniform(0.5, 50, 950),
            'demand_volume_kg': np.random.exponential(10000, 950),
            'seasonality': np.random.choice(['high', 'medium', 'low'], 950, p=[0.3, 0.4, 0.3]),
            'geographic_region': np.random.choice([
                'North_America', 'Europe', 'Asia_Pacific', 'South_America', 'Africa'
            ], 950),
            'quality_requirements': np.random.choice(['high', 'medium', 'low'], 950, p=[0.3, 0.5, 0.2]),
            'certification_required': np.random.choice([True, False], 950, p=[0.6, 0.4])
        }
        
        # Dataset 3: Environmental Impact Data
        environmental_data = {
            'material_type': np.random.choice([
                'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
                'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap'
            ], 500),
            'recycling_process': np.random.choice([
                'mechanical_recycling', 'pyrometallurgy', 'hydrometallurgy',
                'electrometallurgy', 'biometallurgy', 'direct_reuse'
            ], 500),
            'energy_consumption_kwh_per_kg': np.random.uniform(0.5, 15, 500),
            'co2_emissions_kg_per_kg': np.random.uniform(0.1, 8, 500),
            'water_consumption_l_per_kg': np.random.uniform(0.1, 50, 500),
            'waste_generation_kg_per_kg': np.random.uniform(0.01, 0.3, 500),
            'recycling_efficiency': np.random.uniform(0.6, 0.98, 500),
            'circularity_score': np.random.uniform(0.3, 0.95, 500)
        }
        
        # Dataset 4: Processing Capabilities
        processing_data = {
            'facility_id': [f'FAC_{i:03d}' for i in range(1, 51)],
            'facility_type': np.random.choice([
                'recycling_plant', 'smelting_facility', 'refining_plant',
                'processing_center', 'research_lab'
            ], 50),
            'location': np.random.choice([
                'North_America', 'Europe', 'Asia_Pacific', 'South_America', 'Africa'
            ], 50),
            'processing_capacity_kg_per_day': np.random.uniform(1000, 50000, 50),
            'material_types_handled': [
                np.random.choice([
                    'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
                    'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap'
                ], np.random.randint(1, 6)) for _ in range(50)
            ],
            'technology_level': np.random.choice(['basic', 'intermediate', 'advanced'], 50, p=[0.3, 0.5, 0.2]),
            'certification_status': np.random.choice(['certified', 'pending', 'not_certified'], 50, p=[0.7, 0.2, 0.1]),
            'cost_per_kg': np.random.uniform(0.1, 5, 50),
            'processing_time_days': np.random.uniform(1, 30, 50)
        }
        
        # Dataset 5: Historical Recycling Success
        historical_data = {
            'project_id': [f'PROJ_{i:04d}' for i in range(1, 201)],
            'input_material': np.random.choice([
                'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
                'slag', 'dust_particles'
            ], 200),
            'output_product': np.random.choice([
                'new_steel', 'rebar', 'automotive_parts', 'construction_materials',
                'new_aluminum', 'cans', 'electrical_wire', 'cement_additive'
            ], 200),
            'input_quantity_kg': np.random.exponential(5000, 200),
            'output_quantity_kg': np.random.exponential(4500, 200),
            'success_rate': np.random.uniform(0.7, 0.98, 200),
            'profit_margin': np.random.uniform(-0.1, 0.4, 200),
            'processing_time_days': np.random.uniform(1, 45, 200),
            'environmental_impact_score': np.random.uniform(0.6, 0.95, 200),
            'customer_satisfaction': np.random.uniform(0.5, 1.0, 200),
            'project_date': pd.date_range('2022-01-01', periods=200, freq='D')
        }
        
        # Create DataFrames
        self.leftover_df = pd.DataFrame(leftover_data)
        self.market_df = pd.DataFrame(market_data)
        self.environmental_df = pd.DataFrame(environmental_data)
        self.processing_df = pd.DataFrame(processing_data)
        self.historical_df = pd.DataFrame(historical_data)
        
        # Create target variable for recycling suggestions
        self._create_target_variables()
        
        print("Sample datasets created successfully!")
        print(f"Leftover materials: {len(self.leftover_df)} records")
        print(f"Market data: {len(self.market_df)} records")
        print(f"Environmental data: {len(self.environmental_df)} records")
        print(f"Processing facilities: {len(self.processing_df)} records")
        print(f"Historical projects: {len(self.historical_df)} records")
        
        return self.leftover_df, self.market_df, self.environmental_df, self.processing_df, self.historical_df
    
    def _create_target_variables(self):
        """Create target variables for ML models"""
        # Recycling feasibility score (0-1)
        self.leftover_df['recycling_feasibility'] = np.random.uniform(0.3, 0.95, len(self.leftover_df))
        
        # Best recycling product suggestion
        product_suggestions = []
        for material in self.leftover_df['material_type']:
            if material in self.recycling_database:
                products = self.recycling_database[material]['recyclable_products']
                product_suggestions.append(np.random.choice(products))
            else:
                product_suggestions.append('landfill_disposal')
        
        self.leftover_df['suggested_product'] = product_suggestions
        
        # Economic viability score
        self.leftover_df['economic_viability'] = np.random.uniform(0.2, 0.9, len(self.leftover_df))
        
        # Environmental benefit score
        self.leftover_df['environmental_benefit'] = np.random.uniform(0.4, 0.95, len(self.leftover_df))
        
        # Processing complexity (1-5 scale)
        self.leftover_df['processing_complexity'] = np.random.randint(1, 6, len(self.leftover_df))
        
        # Priority level for recycling
        self.leftover_df['recycling_priority'] = np.random.choice(['low', 'medium', 'high', 'critical'], 
                                                                 len(self.leftover_df), p=[0.2, 0.4, 0.3, 0.1])
    
    def prepare_features(self):
        """Prepare features for ML models using mining dataset"""
        # Use mining dataset if available, otherwise use generated data
        if self.mining_df is not None:
            print("Using mining dataset for training...")
            df = self.mining_df
            
            # Map column names to match expected features
            feature_mapping = {
                'Quantity_kg': 'quantity_kg',
                'Purity_%': 'purity_percent',
                'Energy_Content_MJ': 'energy_content',
                'Toxicity_Score': 'toxicity_score',
                'Carbon_Footprint_kgCO2': 'carbon_footprint',
                'Material_Type': 'material_type',
                'Source_Process': 'source_process'
            }
            
            # Rename columns
            df = df.rename(columns=feature_mapping)
            
            # Select features for recycling suggestion
            feature_columns = [
                'quantity_kg', 'purity_percent', 'energy_content', 'toxicity_score',
                'carbon_footprint', 'recycling_feasibility', 'economic_viability',
                'environmental_benefit', 'processing_complexity'
            ]
            
            # Categorical features
            categorical_features = ['material_type', 'source_process']
            
        else:
            print("Using generated dataset for training...")
            df = self.leftover_df
            
            # Select features for recycling suggestion
            feature_columns = [
                'quantity_kg', 'purity_percent', 'particle_size_mm', 'oxidation_level',
                'moisture_content', 'recycling_feasibility', 'economic_viability',
                'environmental_benefit', 'processing_complexity'
            ]
            
            # Categorical features
            categorical_features = [
                'material_type', 'contamination_level', 'source_process', 'location',
                'storage_conditions', 'hazardous_classification'
            ]
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Encode categorical features
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Target variables
        y_product = df['suggested_product']
        y_priority = df['recycling_priority']
        y_feasibility = df['recycling_feasibility']
        
        return X, y_product, y_priority, y_feasibility
    
    def train_models(self):
        """Train ML models for recycling suggestions"""
        print("Training recycling suggestion models...")
        
        X, y_product, y_priority, y_feasibility = self.prepare_features()
        
        # Split data
        X_train, X_test, y_product_train, y_product_test = train_test_split(
            X, y_product, test_size=0.2, random_state=42
        )
        
        # Split priority and feasibility data
        _, _, y_priority_train, y_priority_test = train_test_split(
            X, y_priority, test_size=0.2, random_state=42
        )
        
        _, _, y_feasibility_train, y_feasibility_test = train_test_split(
            X, y_feasibility, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # 1. Product Suggestion Model
        print("Training product suggestion model...")
        product_encoder = LabelEncoder()
        y_product_encoded = product_encoder.fit_transform(y_product)
        y_product_train_encoded = product_encoder.transform(y_product_train)
        y_product_test_encoded = product_encoder.transform(y_product_test)
        self.label_encoders['product'] = product_encoder
        
        rf_product = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_product.fit(X_train_scaled, y_product_train_encoded)
        self.models['product_suggestion'] = rf_product
        
        # 2. Priority Classification Model
        print("Training priority classification model...")
        priority_encoder = LabelEncoder()
        y_priority_encoded = priority_encoder.fit_transform(y_priority)
        y_priority_train_encoded = priority_encoder.transform(y_priority_train)
        y_priority_test_encoded = priority_encoder.transform(y_priority_test)
        self.label_encoders['priority'] = priority_encoder
        
        rf_priority = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_priority.fit(X_train_scaled, y_priority_train_encoded)
        self.models['priority_classification'] = rf_priority
        
        # 3. Feasibility Regression Model
        print("Training feasibility regression model...")
        rf_feasibility = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_feasibility.fit(X_train_scaled, y_feasibility_train)
        self.models['feasibility_regression'] = rf_feasibility
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_product_test_encoded, y_priority_test_encoded, y_feasibility_test)
        
        print("All models trained successfully!")
    
    def _evaluate_models(self, X_test, y_product_test, y_priority_test, y_feasibility_test):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Product Suggestion
        y_product_pred = self.models['product_suggestion'].predict(X_test)
        print("\nProduct Suggestion Model:")
        print(f"Accuracy: {accuracy_score(y_product_test, y_product_pred):.4f}")
        
        # Priority Classification
        y_priority_pred = self.models['priority_classification'].predict(X_test)
        print("\nPriority Classification Model:")
        print(f"Accuracy: {accuracy_score(y_priority_test, y_priority_pred):.4f}")
        
        # Feasibility Regression
        y_feasibility_pred = self.models['feasibility_regression'].predict(X_test)
        print("\nFeasibility Regression Model:")
        print(f"R² Score: {r2_score(y_feasibility_test, y_feasibility_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_feasibility_test, y_feasibility_pred)):.4f}")
    
    def suggest_recycling(self, material_data):
        """
        Suggest recycling strategy for given material data
        
        Args:
            material_data (dict): Material characteristics
            
        Returns:
            dict: Recycling suggestions and recommendations
        """
        # Convert to DataFrame
        df_new = pd.DataFrame([material_data])
        
        # Handle different input formats (mining dataset vs generated data)
        if 'Material_Type' in material_data:
            # Mining dataset format
            feature_mapping = {
                'Quantity_kg': 'quantity_kg',
                'Purity_%': 'purity_percent',
                'Energy_Content_MJ': 'energy_content',
                'Toxicity_Score': 'toxicity_score',
                'Carbon_Footprint_kgCO2': 'carbon_footprint',
                'Material_Type': 'material_type',
                'Source_Process': 'source_process'
            }
            df_new = df_new.rename(columns=feature_mapping)
            
            # Create derived features
            df_new['recycling_feasibility'] = np.where(
                material_data.get('Recyclable') == 'Yes', 
                np.random.uniform(0.7, 0.95),
                np.random.uniform(0.2, 0.6)
            )
            
            toxicity_score = material_data.get('Toxicity_Score', 5)
            toxicity_mapping = {1: 'low', 2: 'low', 3: 'medium', 4: 'medium', 5: 'high', 6: 'high', 7: 'critical', 8: 'critical', 9: 'critical', 10: 'critical'}
            df_new['recycling_priority'] = toxicity_mapping.get(toxicity_score, 'medium')
            
            max_carbon = 1000  # Estimated max carbon footprint
            df_new['environmental_benefit'] = 1 - (material_data.get('Carbon_Footprint_kgCO2', 500) / max_carbon)
            
            df_new['economic_viability'] = (
                material_data.get('Purity_%', 50) / 100 * 0.6 + 
                (material_data.get('Energy_Content_MJ', 100) / 1000) * 0.4
            )
            
            df_new['processing_complexity'] = np.where(
                toxicity_score > 7, 5,
                np.where(toxicity_score > 5, 4,
                        np.where(toxicity_score > 3, 3,
                                np.where(toxicity_score > 1, 2, 1)))
            )
            
            feature_columns = [
                'quantity_kg', 'purity_percent', 'energy_content', 'toxicity_score',
                'carbon_footprint', 'recycling_feasibility', 'economic_viability',
                'environmental_benefit', 'processing_complexity'
            ]
            categorical_features = ['material_type', 'source_process']
        else:
            # Generated data format
            feature_columns = [
                'quantity_kg', 'purity_percent', 'particle_size_mm', 'oxidation_level',
                'moisture_content', 'recycling_feasibility', 'economic_viability',
                'environmental_benefit', 'processing_complexity'
            ]
            categorical_features = ['material_type', 'contamination_level', 'source_process', 'location',
                                   'storage_conditions', 'hazardous_classification']
        
        X_new = df_new[feature_columns].copy()
        
        # Encode categorical features
        for col in categorical_features:
            if col in df_new.columns and col in self.label_encoders:
                try:
                    X_new[col] = self.label_encoders[col].transform(df_new[col].astype(str))
                except ValueError:
                    X_new[col] = 0  # Default value
        
        # Scale features
        X_new_scaled = self.scalers['main'].transform(X_new)
        
        # Make predictions
        product_pred = self.models['product_suggestion'].predict(X_new_scaled)[0]
        product_name = self.label_encoders['product'].inverse_transform([product_pred])[0]
        
        priority_pred = self.models['priority_classification'].predict(X_new_scaled)[0]
        priority = self.label_encoders['priority'].inverse_transform([priority_pred])[0]
        
        feasibility_pred = self.models['feasibility_regression'].predict(X_new_scaled)[0]
        
        # Get recycling information
        material_type = material_data.get('material_type', material_data.get('Material_Type', 'unknown'))
        if material_type:
            material_type = material_type.lower().replace(' ', '_')
        recycling_info = self.recycling_database.get(material_type, {})
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            material_data, product_name, priority, feasibility_pred, recycling_info
        )
        
        return {
            'suggested_product': product_name,
            'recycling_priority': priority,
            'feasibility_score': float(feasibility_pred),
            'recycling_efficiency': recycling_info.get('recycling_efficiency', 0.7),
            'energy_savings': recycling_info.get('energy_savings', 0.5),
            'co2_reduction': recycling_info.get('co2_reduction', 0.4),
            'market_value': recycling_info.get('market_value', 0.5),
            'recommendations': recommendations,
            'environmental_impact': {
                'co2_savings_kg': material_data.get('quantity_kg', material_data.get('Quantity_kg', 0)) * recycling_info.get('co2_reduction', 0.4),
                'energy_savings_kwh': material_data.get('quantity_kg', material_data.get('Quantity_kg', 0)) * recycling_info.get('energy_savings', 0.5) * 10,
                'circularity_score': feasibility_pred
            }
        }
    
    def _generate_recommendations(self, material_data, product, priority, feasibility, recycling_info):
        """Generate recycling recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        if priority == 'critical':
            recommendations.append("URGENT: Process immediately - high value material")
            recommendations.append("IMMEDIATE: Contact specialized recycling facilities")
        elif priority == 'high':
            recommendations.append("HIGH PRIORITY: Schedule processing within 1 week")
            recommendations.append("RECOMMENDED: Pre-process to improve quality")
        elif priority == 'medium':
            recommendations.append("MEDIUM PRIORITY: Process within 1 month")
            recommendations.append("CONSIDER: Batch with similar materials for efficiency")
        else:
            recommendations.append("LOW PRIORITY: Process when capacity available")
            recommendations.append("EVALUATE: Consider storage costs vs processing benefits")
        
        # Feasibility-based recommendations
        if feasibility > 0.8:
            recommendations.append("EXCELLENT: High recycling potential - proceed immediately")
        elif feasibility > 0.6:
            recommendations.append("GOOD: Viable recycling option - recommended")
        elif feasibility > 0.4:
            recommendations.append("MODERATE: Consider pre-treatment before recycling")
        else:
            recommendations.append("LOW: Evaluate alternative disposal methods")
        
        # Material-specific recommendations
        material_type = material_data.get('material_type', '')
        if 'scrap' in material_type:
            recommendations.append("PREPARE: Clean and sort material for maximum value")
        if material_data.get('contamination_level') == 'high':
            recommendations.append("CLEAN: Remove contaminants to improve recycling efficiency")
        if material_data.get('purity_percent', 0) < 80:
            recommendations.append("REFINE: Consider purification before recycling")
        
        # Environmental recommendations
        if recycling_info.get('co2_reduction', 0) > 0.7:
            recommendations.append("ENVIRONMENTAL: High CO2 reduction potential - prioritize")
        if recycling_info.get('energy_savings', 0) > 0.8:
            recommendations.append("ENERGY: Significant energy savings achievable")
        
        return recommendations
    
    def create_visualizations(self):
        """Create visualizations for recycling analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI-Driven LCA Recycling Analysis', fontsize=16, fontweight='bold')
        
        # Use mining dataset if available, otherwise use generated data
        if self.mining_df is not None:
            df = self.mining_df
            # Use original column names for mining dataset
            material_col = 'Material_Type'
            priority_col = 'recycling_priority'
            feasibility_col = 'recycling_feasibility'
            economic_col = 'economic_viability'
            environmental_col = 'environmental_benefit'
            complexity_col = 'processing_complexity'
            product_col = 'suggested_product'
        else:
            df = self.leftover_df
            # Use generated data column names
            material_col = 'material_type'
            priority_col = 'recycling_priority'
            feasibility_col = 'recycling_feasibility'
            economic_col = 'economic_viability'
            environmental_col = 'environmental_benefit'
            complexity_col = 'processing_complexity'
            product_col = 'suggested_product'
        
        # Material distribution
        material_counts = df[material_col].value_counts()
        axes[0, 0].pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Material Distribution')
        
        # Recycling feasibility distribution
        axes[0, 1].hist(df[feasibility_col], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Recycling Feasibility Distribution')
        axes[0, 1].set_xlabel('Feasibility Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Priority levels
        priority_counts = df[priority_col].value_counts()
        axes[0, 2].bar(priority_counts.index, priority_counts.values, color=['red', 'orange', 'yellow', 'green'])
        axes[0, 2].set_title('Recycling Priority Levels')
        axes[0, 2].set_ylabel('Count')
        
        # Economic vs Environmental benefit
        axes[1, 0].scatter(df[economic_col], 
                          df[environmental_col], 
                          alpha=0.6, c=df[feasibility_col], cmap='viridis')
        axes[1, 0].set_title('Economic vs Environmental Benefit')
        axes[1, 0].set_xlabel('Economic Viability')
        axes[1, 0].set_ylabel('Environmental Benefit')
        
        # Processing complexity by material type
        complexity_by_material = df.groupby(material_col)[complexity_col].mean()
        axes[1, 1].bar(range(len(complexity_by_material)), complexity_by_material.values)
        axes[1, 1].set_title('Processing Complexity by Material')
        axes[1, 1].set_xticks(range(len(complexity_by_material)))
        axes[1, 1].set_xticklabels(complexity_by_material.index, rotation=45)
        axes[1, 1].set_ylabel('Average Complexity')
        
        # Suggested products distribution
        product_counts = df[product_col].value_counts()
        axes[1, 2].bar(range(len(product_counts)), product_counts.values)
        axes[1, 2].set_title('Suggested Products Distribution')
        axes[1, 2].set_xticks(range(len(product_counts)))
        axes[1, 2].set_xticklabels(product_counts.index, rotation=45)
        axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('recycling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate the recycling suggestion model"""
    print("="*60)
    print("AI-DRIVEN LCA RECYCLING SUGGESTION MODEL")
    print("Metallurgy and Mining Industry")
    print("="*60)
    
    # Initialize model
    model = RecyclingSuggestionModel()
    
    # Load mining dataset first
    print("Loading mining dataset...")
    mining_df = model.load_mining_dataset()
    
    # Create additional sample datasets if needed
    if mining_df is None:
        print("Creating sample datasets...")
        model.create_sample_datasets()
    
    # Train models
    print("\nTraining ML models...")
    model.train_models()
    
    # Create visualizations
    print("\nCreating visualizations...")
    model.create_visualizations()
    
    # Example recycling suggestion using mining dataset format
    print("\n" + "="*50)
    print("EXAMPLE RECYCLING SUGGESTION")
    print("="*50)
    
    # Use a sample from your mining dataset
    if model.mining_df is not None:
        sample_row = model.mining_df.iloc[0]
        sample_material = {
            'Material_ID': sample_row['Material_ID'],
            'Material_Type': sample_row['Material_Type'],
            'Source_Process': sample_row['Source_Process'],
            'Quantity_kg': sample_row['Quantity_kg'],
            'Purity_%': sample_row['Purity_%'],
            'Contaminants': sample_row['Contaminants'],
            'Energy_Content_MJ': sample_row['Energy_Content_MJ'],
            'Toxicity_Score': sample_row['Toxicity_Score'],
            'Carbon_Footprint_kgCO2': sample_row['Carbon_Footprint_kgCO2'],
            'Recyclable': sample_row['Recyclable'],
            'Suggested_Reuse': sample_row['Suggested_Reuse']
        }
        print(f"Using sample from mining dataset: {sample_material['Material_ID']}")
    else:
        sample_material = {
            'material_type': 'steel_scrap',
            'quantity_kg': 5000,
            'purity_percent': 85,
            'contamination_level': 'medium',
            'particle_size_mm': 15.5,
            'oxidation_level': 8.2,
            'moisture_content': 3.1,
            'source_process': 'machining',
            'location': 'Machining_Shop',
            'storage_conditions': 'indoor_dry',
            'hazardous_classification': 'non_hazardous',
            'recycling_feasibility': 0.85,
            'economic_viability': 0.78,
            'environmental_benefit': 0.82,
            'processing_complexity': 3
        }
    
    suggestion = model.suggest_recycling(sample_material)
    
    print(f"Suggested Product: {suggestion['suggested_product']}")
    print(f"Priority Level: {suggestion['recycling_priority']}")
    print(f"Feasibility Score: {suggestion['feasibility_score']:.3f}")
    print(f"Recycling Efficiency: {suggestion['recycling_efficiency']:.1%}")
    print(f"Energy Savings: {suggestion['energy_savings']:.1%}")
    print(f"CO2 Reduction: {suggestion['co2_reduction']:.1%}")
    print(f"Market Value: {suggestion['market_value']:.1%}")
    
    print("\nEnvironmental Impact:")
    print(f"  CO2 Savings: {suggestion['environmental_impact']['co2_savings_kg']:.1f} kg")
    print(f"  Energy Savings: {suggestion['environmental_impact']['energy_savings_kwh']:.1f} kWh")
    print(f"  Circularity Score: {suggestion['environmental_impact']['circularity_score']:.3f}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(suggestion['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*60)
    print("RECYCLING SUGGESTION MODEL READY")
    print("="*60)

if __name__ == "__main__":
    main()
