"""
Generate Sample Datasets for AI-Driven LCA Recycling Model
Creates comprehensive datasets for metallurgy and mining recycling suggestions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_leftover_materials_dataset():
    """Generate comprehensive leftover materials dataset"""
    
    # Material types commonly found in metallurgy and mining
    material_types = [
        'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
        'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap',
        'titanium_scrap', 'gold_scrap', 'silver_scrap', 'platinum_scrap',
        'chrome_scrap', 'manganese_scrap', 'tungsten_scrap', 'molybdenum_scrap',
        'cobalt_scrap', 'vanadium_scrap', 'hafnium_scrap'
    ]
    
    # Process sources
    source_processes = [
        'mining', 'smelting', 'refining', 'casting', 'machining',
        'welding', 'cutting', 'grinding', 'polishing', 'forging',
        'rolling', 'extrusion', 'stamping', 'bending', 'drilling'
    ]
    
    # Locations
    locations = [
        'Extraction_Zone_A', 'Smelter_Unit_1', 'Refinery_Section_B',
        'Casting_Facility', 'Machining_Shop', 'Storage_Yard',
        'Processing_Plant', 'Quality_Control', 'Maintenance_Depot'
    ]
    
    # Storage conditions
    storage_conditions = [
        'indoor_dry', 'indoor_humid', 'outdoor_covered', 'outdoor_exposed',
        'climate_controlled', 'refrigerated', 'vacuum_sealed'
    ]
    
    # Hazardous classifications
    hazardous_levels = [
        'non_hazardous', 'low_hazard', 'moderate_hazard', 'high_hazard',
        'extremely_hazardous'
    ]
    
    # Generate 2000 records
    n_records = 2000
    
    data = {
        'material_id': [f'MAT_{i:04d}' for i in range(1, n_records + 1)],
        'material_type': np.random.choice(material_types, n_records),
        'quantity_kg': np.random.exponential(2000, n_records),
        'purity_percent': np.random.uniform(50, 99.5, n_records),
        'contamination_level': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.6, 0.3, 0.1]),
        'particle_size_mm': np.random.uniform(0.01, 100, n_records),
        'oxidation_level': np.random.uniform(0, 40, n_records),
        'moisture_content': np.random.uniform(0, 20, n_records),
        'collection_date': pd.date_range('2023-01-01', periods=n_records, freq='H'),
        'source_process': np.random.choice(source_processes, n_records),
        'location': np.random.choice(locations, n_records),
        'storage_conditions': np.random.choice(storage_conditions, n_records),
        'hazardous_classification': np.random.choice(hazardous_levels, n_records, p=[0.7, 0.2, 0.08, 0.015, 0.005]),
        'temperature_c': np.random.uniform(-10, 60, n_records),
        'ph_level': np.random.uniform(4, 10, n_records),
        'density_kg_m3': np.random.uniform(2000, 8000, n_records),
        'magnetic_properties': np.random.choice(['ferromagnetic', 'paramagnetic', 'diamagnetic'], n_records),
        'corrosion_resistance': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.3, 0.5, 0.2]),
        'thermal_conductivity': np.random.uniform(10, 400, n_records),
        'electrical_conductivity': np.random.uniform(0.1, 100, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['value_per_kg'] = df['purity_percent'] * np.random.uniform(0.5, 50, n_records) / 100
    df['processing_difficulty'] = np.where(df['contamination_level'] == 'high', 
                                         np.random.uniform(0.7, 1.0, n_records),
                                         np.random.uniform(0.2, 0.8, n_records))
    
    return df

def generate_market_demand_dataset():
    """Generate market demand and pricing dataset"""
    
    product_types = [
        'new_steel', 'rebar', 'automotive_parts', 'construction_materials',
        'new_aluminum', 'cans', 'foil', 'electrical_wire', 'plumbing',
        'electronics', 'jewelry', 'cement_additive', 'concrete_aggregate',
        'road_base', 'cement', 'road_construction', 'paint_pigments',
        'ceramics', 'filters', 'batteries', 'solar_panels', 'wind_turbines',
        'medical_devices', 'aerospace_components', 'marine_equipment'
    ]
    
    regions = [
        'North_America', 'Europe', 'Asia_Pacific', 'South_America', 'Africa',
        'Middle_East', 'Oceania'
    ]
    
    n_records = 1500
    
    data = {
        'product_id': [f'PROD_{i:04d}' for i in range(1, n_records + 1)],
        'product_type': np.random.choice(product_types, n_records),
        'market_demand_trend': np.random.choice(['increasing', 'stable', 'decreasing'], n_records, p=[0.4, 0.4, 0.2]),
        'price_per_kg': np.random.uniform(0.5, 100, n_records),
        'demand_volume_kg': np.random.exponential(50000, n_records),
        'seasonality': np.random.choice(['high', 'medium', 'low'], n_records, p=[0.3, 0.4, 0.3]),
        'geographic_region': np.random.choice(regions, n_records),
        'quality_requirements': np.random.choice(['high', 'medium', 'low'], n_records, p=[0.3, 0.5, 0.2]),
        'certification_required': np.random.choice([True, False], n_records, p=[0.6, 0.4]),
        'market_maturity': np.random.choice(['emerging', 'growing', 'mature', 'declining'], n_records, p=[0.2, 0.3, 0.4, 0.1]),
        'competition_level': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.2, 0.5, 0.3]),
        'price_volatility': np.random.uniform(0.05, 0.5, n_records),
        'growth_rate': np.random.uniform(-0.1, 0.3, n_records),
        'market_size_kg': np.random.exponential(1000000, n_records),
        'customer_segments': np.random.choice(['B2B', 'B2C', 'Government', 'Mixed'], n_records, p=[0.5, 0.2, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['market_opportunity'] = df['demand_volume_kg'] * df['price_per_kg'] * df['growth_rate']
    df['competitive_advantage'] = np.where(df['competition_level'] == 'low', 
                                         np.random.uniform(0.7, 1.0, n_records),
                                         np.random.uniform(0.3, 0.8, n_records))
    
    return df

def generate_environmental_impact_dataset():
    """Generate environmental impact and sustainability dataset"""
    
    material_types = [
        'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
        'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap'
    ]
    
    recycling_processes = [
        'mechanical_recycling', 'pyrometallurgy', 'hydrometallurgy',
        'electrometallurgy', 'biometallurgy', 'direct_reuse',
        'chemical_treatment', 'thermal_treatment', 'magnetic_separation'
    ]
    
    n_records = 800
    
    data = {
        'process_id': [f'ENV_{i:04d}' for i in range(1, n_records + 1)],
        'material_type': np.random.choice(material_types, n_records),
        'recycling_process': np.random.choice(recycling_processes, n_records),
        'energy_consumption_kwh_per_kg': np.random.uniform(0.1, 20, n_records),
        'co2_emissions_kg_per_kg': np.random.uniform(0.05, 10, n_records),
        'water_consumption_l_per_kg': np.random.uniform(0.01, 100, n_records),
        'waste_generation_kg_per_kg': np.random.uniform(0.001, 0.5, n_records),
        'recycling_efficiency': np.random.uniform(0.5, 0.99, n_records),
        'circularity_score': np.random.uniform(0.2, 0.95, n_records),
        'sustainability_rating': np.random.choice(['A+', 'A', 'B', 'C', 'D'], n_records, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'renewable_energy_usage': np.random.uniform(0, 1, n_records),
        'toxicity_level': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.6, 0.3, 0.1]),
        'biodegradability': np.random.uniform(0, 0.8, n_records),
        'resource_depletion_factor': np.random.uniform(0.1, 0.9, n_records),
        'ecosystem_impact': np.random.choice(['minimal', 'moderate', 'significant'], n_records, p=[0.5, 0.4, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['environmental_score'] = (df['recycling_efficiency'] * 0.3 + 
                               (1 - df['co2_emissions_kg_per_kg'] / 10) * 0.3 +
                               df['circularity_score'] * 0.2 +
                               df['renewable_energy_usage'] * 0.2)
    
    return df

def generate_processing_facilities_dataset():
    """Generate processing facilities and capabilities dataset"""
    
    facility_types = [
        'recycling_plant', 'smelting_facility', 'refining_plant',
        'processing_center', 'research_lab', 'pilot_plant',
        'waste_treatment', 'material_recovery', 'sorting_facility'
    ]
    
    regions = [
        'North_America', 'Europe', 'Asia_Pacific', 'South_America', 'Africa',
        'Middle_East', 'Oceania'
    ]
    
    technology_levels = ['basic', 'intermediate', 'advanced', 'cutting_edge']
    
    n_records = 100
    
    data = {
        'facility_id': [f'FAC_{i:03d}' for i in range(1, n_records + 1)],
        'facility_name': [f'Facility_{i}' for i in range(1, n_records + 1)],
        'facility_type': np.random.choice(facility_types, n_records),
        'location': np.random.choice(regions, n_records),
        'processing_capacity_kg_per_day': np.random.uniform(1000, 100000, n_records),
        'technology_level': np.random.choice(technology_levels, n_records, p=[0.2, 0.4, 0.3, 0.1]),
        'certification_status': np.random.choice(['certified', 'pending', 'not_certified'], n_records, p=[0.7, 0.2, 0.1]),
        'cost_per_kg': np.random.uniform(0.05, 10, n_records),
        'processing_time_days': np.random.uniform(0.5, 60, n_records),
        'quality_standards': np.random.choice(['ISO_9001', 'ISO_14001', 'OHSAS_18001', 'Multiple'], n_records, p=[0.3, 0.3, 0.2, 0.2]),
        'automation_level': np.random.uniform(0.2, 0.95, n_records),
        'energy_efficiency': np.random.uniform(0.5, 0.95, n_records),
        'waste_minimization': np.random.uniform(0.6, 0.98, n_records),
        'safety_rating': np.random.uniform(0.7, 1.0, n_records),
        'environmental_compliance': np.random.uniform(0.8, 1.0, n_records),
        'years_in_operation': np.random.randint(1, 50, n_records),
        'employee_count': np.random.randint(10, 500, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['efficiency_score'] = (df['automation_level'] * 0.3 + 
                            df['energy_efficiency'] * 0.3 +
                            df['waste_minimization'] * 0.2 +
                            df['safety_rating'] * 0.2)
    
    return df

def generate_historical_projects_dataset():
    """Generate historical recycling projects dataset"""
    
    input_materials = [
        'steel_scrap', 'aluminum_scrap', 'copper_scrap', 'iron_ore_tailings',
        'slag', 'dust_particles', 'zinc_scrap', 'lead_scrap', 'nickel_scrap'
    ]
    
    output_products = [
        'new_steel', 'rebar', 'automotive_parts', 'construction_materials',
        'new_aluminum', 'cans', 'electrical_wire', 'cement_additive',
        'concrete_aggregate', 'road_base', 'paint_pigments', 'ceramics'
    ]
    
    project_statuses = ['completed', 'in_progress', 'cancelled', 'on_hold']
    
    n_records = 500
    
    data = {
        'project_id': [f'PROJ_{i:04d}' for i in range(1, n_records + 1)],
        'input_material': np.random.choice(input_materials, n_records),
        'output_product': np.random.choice(output_products, n_records),
        'input_quantity_kg': np.random.exponential(10000, n_records),
        'output_quantity_kg': np.random.exponential(9000, n_records),
        'success_rate': np.random.uniform(0.6, 0.99, n_records),
        'profit_margin': np.random.uniform(-0.2, 0.5, n_records),
        'processing_time_days': np.random.uniform(1, 90, n_records),
        'environmental_impact_score': np.random.uniform(0.5, 0.98, n_records),
        'customer_satisfaction': np.random.uniform(0.4, 1.0, n_records),
        'project_status': np.random.choice(project_statuses, n_records, p=[0.7, 0.2, 0.05, 0.05]),
        'project_date': pd.date_range('2020-01-01', periods=n_records, freq='D'),
        'investment_required': np.random.exponential(100000, n_records),
        'roi_percentage': np.random.uniform(-10, 50, n_records),
        'risk_level': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.5, 0.3, 0.2]),
        'innovation_level': np.random.choice(['conventional', 'improved', 'innovative', 'breakthrough'], n_records, p=[0.4, 0.3, 0.2, 0.1]),
        'market_acceptance': np.random.uniform(0.3, 1.0, n_records),
        'scalability': np.random.choice(['low', 'medium', 'high'], n_records, p=[0.2, 0.5, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['efficiency_ratio'] = df['output_quantity_kg'] / df['input_quantity_kg']
    df['sustainability_score'] = (df['environmental_impact_score'] * 0.4 + 
                                df['success_rate'] * 0.3 +
                                df['customer_satisfaction'] * 0.3)
    
    return df

def generate_recycling_database():
    """Generate comprehensive recycling possibilities database"""
    
    recycling_database = {
        'steel_scrap': {
            'recyclable_products': ['new_steel', 'rebar', 'automotive_parts', 'construction_materials', 'appliances'],
            'recycling_efficiency': 0.95,
            'energy_savings': 0.74,
            'co2_reduction': 0.58,
            'market_value': 0.8,
            'processing_complexity': 2,
            'investment_required': 'medium',
            'market_demand': 'high'
        },
        'aluminum_scrap': {
            'recyclable_products': ['new_aluminum', 'cans', 'foil', 'automotive_parts', 'packaging'],
            'recycling_efficiency': 0.92,
            'energy_savings': 0.95,
            'co2_reduction': 0.92,
            'market_value': 0.9,
            'processing_complexity': 2,
            'investment_required': 'medium',
            'market_demand': 'very_high'
        },
        'copper_scrap': {
            'recyclable_products': ['electrical_wire', 'plumbing', 'electronics', 'jewelry', 'coins'],
            'recycling_efficiency': 0.88,
            'energy_savings': 0.85,
            'co2_reduction': 0.65,
            'market_value': 0.95,
            'processing_complexity': 3,
            'investment_required': 'high',
            'market_demand': 'high'
        },
        'iron_ore_tailings': {
            'recyclable_products': ['cement_additive', 'concrete_aggregate', 'road_base', 'brick_making'],
            'recycling_efficiency': 0.70,
            'energy_savings': 0.60,
            'co2_reduction': 0.40,
            'market_value': 0.3,
            'processing_complexity': 1,
            'investment_required': 'low',
            'market_demand': 'medium'
        },
        'slag': {
            'recyclable_products': ['cement', 'concrete_aggregate', 'road_construction', 'fertilizer'],
            'recycling_efficiency': 0.75,
            'energy_savings': 0.50,
            'co2_reduction': 0.45,
            'market_value': 0.4,
            'processing_complexity': 2,
            'investment_required': 'medium',
            'market_demand': 'medium'
        },
        'dust_particles': {
            'recyclable_products': ['paint_pigments', 'ceramics', 'filters', 'abrasives'],
            'recycling_efficiency': 0.60,
            'energy_savings': 0.30,
            'co2_reduction': 0.25,
            'market_value': 0.2,
            'processing_complexity': 4,
            'investment_required': 'high',
            'market_demand': 'low'
        },
        'zinc_scrap': {
            'recyclable_products': ['galvanized_steel', 'batteries', 'alloys', 'coatings'],
            'recycling_efficiency': 0.85,
            'energy_savings': 0.70,
            'co2_reduction': 0.60,
            'market_value': 0.7,
            'processing_complexity': 3,
            'investment_required': 'high',
            'market_demand': 'medium'
        },
        'lead_scrap': {
            'recyclable_products': ['batteries', 'radiation_shielding', 'alloys', 'pipes'],
            'recycling_efficiency': 0.90,
            'energy_savings': 0.80,
            'co2_reduction': 0.70,
            'market_value': 0.6,
            'processing_complexity': 4,
            'investment_required': 'very_high',
            'market_demand': 'medium'
        },
        'nickel_scrap': {
            'recyclable_products': ['stainless_steel', 'batteries', 'alloys', 'catalysts'],
            'recycling_efficiency': 0.88,
            'energy_savings': 0.75,
            'co2_reduction': 0.65,
            'market_value': 0.85,
            'processing_complexity': 4,
            'investment_required': 'very_high',
            'market_demand': 'high'
        }
    }
    
    return recycling_database

def main():
    """Generate all sample datasets"""
    print("="*60)
    print("GENERATING SAMPLE DATASETS FOR AI-DRIVEN LCA RECYCLING MODEL")
    print("="*60)
    
    # Generate datasets
    print("Generating leftover materials dataset...")
    leftover_df = generate_leftover_materials_dataset()
    leftover_df.to_csv('leftover_materials_dataset.csv', index=False)
    
    print("Generating market demand dataset...")
    market_df = generate_market_demand_dataset()
    market_df.to_csv('market_demand_dataset.csv', index=False)
    
    print("Generating environmental impact dataset...")
    environmental_df = generate_environmental_impact_dataset()
    environmental_df.to_csv('environmental_impact_dataset.csv', index=False)
    
    print("Generating processing facilities dataset...")
    facilities_df = generate_processing_facilities_dataset()
    facilities_df.to_csv('processing_facilities_dataset.csv', index=False)
    
    print("Generating historical projects dataset...")
    historical_df = generate_historical_projects_dataset()
    historical_df.to_csv('historical_projects_dataset.csv', index=False)
    
    print("Generating recycling database...")
    recycling_db = generate_recycling_database()
    with open('recycling_database.json', 'w') as f:
        json.dump(recycling_db, f, indent=2)
    
    # Create summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Leftover Materials: {len(leftover_df)} records")
    print(f"Market Demand: {len(market_df)} records")
    print(f"Environmental Impact: {len(environmental_df)} records")
    print(f"Processing Facilities: {len(facilities_df)} records")
    print(f"Historical Projects: {len(historical_df)} records")
    print(f"Recycling Database: {len(recycling_db)} material types")
    
    print("\nFiles created:")
    print("- leftover_materials_dataset.csv")
    print("- market_demand_dataset.csv")
    print("- environmental_impact_dataset.csv")
    print("- processing_facilities_dataset.csv")
    print("- historical_projects_dataset.csv")
    print("- recycling_database.json")
    
    print("\n" + "="*60)
    print("SAMPLE DATASETS GENERATED SUCCESSFULLY!")
    print("Ready for ML model training and frontend integration")
    print("="*60)

if __name__ == "__main__":
    main()
