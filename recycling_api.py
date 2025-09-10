"""
API Server for AI-Driven LCA Recycling Suggestion System
Provides REST API endpoints for recycling recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
from recycling_suggestion_model import RecyclingSuggestionModel

app = Flask(__name__)
CORS(app)

# Global model instance
recycling_model = None

def initialize_model():
    """Initialize the recycling suggestion model"""
    global recycling_model
    
    print("Initializing recycling suggestion model...")
    recycling_model = RecyclingSuggestionModel()
    
    # Create and train the model
    recycling_model.create_sample_datasets()
    recycling_model.train_models()
    
    print("Recycling model initialized successfully!")

@app.route('/api/recycling/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_initialized': recycling_model is not None
    })

@app.route('/api/recycling/suggest', methods=['POST'])
def suggest_recycling():
    """
    Suggest recycling strategy for given material data
    
    Expected JSON payload:
    {
        "material_type": "steel_scrap",
        "quantity_kg": 5000,
        "purity_percent": 85,
        "contamination_level": "medium",
        "particle_size_mm": 15.5,
        "oxidation_level": 8.2,
        "moisture_content": 3.1,
        "source_process": "machining",
        "location": "Machining_Shop",
        "storage_conditions": "indoor_dry",
        "hazardous_classification": "non_hazardous"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make recycling suggestion
        suggestion = recycling_model.suggest_recycling(data)
        
        return jsonify({
            'success': True,
            'suggestion': suggestion,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/suggest/batch', methods=['POST'])
def suggest_recycling_batch():
    """
    Suggest recycling strategies for multiple materials
    
    Expected JSON payload:
    {
        "materials": [
            {material_data_1},
            {material_data_2},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'materials' not in data:
            return jsonify({'error': 'No materials provided'}), 400
        
        suggestions = []
        for material in data['materials']:
            suggestion = recycling_model.suggest_recycling(material)
            suggestions.append({
                'material_data': material,
                'suggestion': suggestion
            })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'count': len(suggestions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/materials', methods=['GET'])
def get_material_types():
    """Get available material types"""
    try:
        material_types = list(recycling_model.recycling_database.keys())
        
        return jsonify({
            'success': True,
            'material_types': material_types,
            'count': len(material_types),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/products', methods=['GET'])
def get_recyclable_products():
    """Get all recyclable products"""
    try:
        all_products = set()
        for material_info in recycling_model.recycling_database.values():
            all_products.update(material_info['recyclable_products'])
        
        return jsonify({
            'success': True,
            'recyclable_products': list(all_products),
            'count': len(all_products),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/material/<material_type>', methods=['GET'])
def get_material_info(material_type):
    """Get detailed information about a specific material type"""
    try:
        if material_type not in recycling_model.recycling_database:
            return jsonify({
                'success': False,
                'error': f'Material type {material_type} not found'
            }), 404
        
        material_info = recycling_model.recycling_database[material_type]
        
        return jsonify({
            'success': True,
            'material_type': material_type,
            'info': material_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/statistics', methods=['GET'])
def get_recycling_statistics():
    """Get recycling statistics from the dataset"""
    try:
        stats = {
            'total_materials': len(recycling_model.leftover_df),
            'material_distribution': recycling_model.leftover_df['material_type'].value_counts().to_dict(),
            'priority_distribution': recycling_model.leftover_df['recycling_priority'].value_counts().to_dict(),
            'average_feasibility': float(recycling_model.leftover_df['recycling_feasibility'].mean()),
            'average_economic_viability': float(recycling_model.leftover_df['economic_viability'].mean()),
            'average_environmental_benefit': float(recycling_model.leftover_df['environmental_benefit'].mean()),
            'processing_complexity_distribution': recycling_model.leftover_df['processing_complexity'].value_counts().to_dict(),
            'suggested_products_distribution': recycling_model.leftover_df['suggested_product'].value_counts().to_dict()
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

@app.route('/api/recycling/optimize', methods=['POST'])
def optimize_recycling_strategy():
    """
    Optimize recycling strategy for multiple materials
    
    Expected JSON payload:
    {
        "materials": [material_data_list],
        "constraints": {
            "max_investment": 1000000,
            "max_processing_time": 30,
            "min_environmental_benefit": 0.7,
            "priority_materials": ["steel_scrap", "aluminum_scrap"]
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'materials' not in data:
            return jsonify({'error': 'No materials provided'}), 400
        
        materials = data['materials']
        constraints = data.get('constraints', {})
        
        # Analyze each material
        suggestions = []
        total_investment = 0
        total_environmental_benefit = 0
        high_priority_count = 0
        
        for material in materials:
            suggestion = recycling_model.suggest_recycling(material)
            suggestions.append({
                'material_data': material,
                'suggestion': suggestion
            })
            
            # Calculate totals
            total_investment += material.get('quantity_kg', 0) * 0.5  # Estimated cost
            total_environmental_benefit += suggestion['environmental_impact']['co2_savings_kg']
            
            if suggestion['recycling_priority'] in ['high', 'critical']:
                high_priority_count += 1
        
        # Optimization recommendations
        optimization = {
            'total_materials': len(materials),
            'total_investment_required': total_investment,
            'total_environmental_benefit': total_environmental_benefit,
            'high_priority_materials': high_priority_count,
            'recommendations': []
        }
        
        # Add optimization recommendations
        if total_investment > constraints.get('max_investment', float('inf')):
            optimization['recommendations'].append("Consider phased implementation to manage investment")
        
        if high_priority_count > len(materials) * 0.5:
            optimization['recommendations'].append("High priority materials detected - consider immediate processing")
        
        if total_environmental_benefit > 10000:
            optimization['recommendations'].append("Significant environmental benefits achievable - prioritize implementation")
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'optimization': optimization,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recycling/lca/calculate', methods=['POST'])
def calculate_lca_impact():
    """
    Calculate LCA impact for recycling strategy
    
    Expected JSON payload:
    {
        "materials": [material_data_list],
        "recycling_strategy": "suggested_products"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'materials' not in data:
            return jsonify({'error': 'No materials provided'}), 400
        
        materials = data['materials']
        strategy = data.get('recycling_strategy', 'suggested_products')
        
        total_lca_impact = {
            'total_co2_savings': 0,
            'total_energy_savings': 0,
            'total_water_savings': 0,
            'total_waste_reduction': 0,
            'circularity_score': 0,
            'sustainability_rating': 'A',
            'materials_processed': len(materials)
        }
        
        for material in materials:
            suggestion = recycling_model.suggest_recycling(material)
            impact = suggestion['environmental_impact']
            
            total_lca_impact['total_co2_savings'] += impact['co2_savings_kg']
            total_lca_impact['total_energy_savings'] += impact['energy_savings_kwh']
            total_lca_impact['circularity_score'] += impact['circularity_score']
        
        # Calculate averages
        total_lca_impact['circularity_score'] /= len(materials)
        total_lca_impact['total_water_savings'] = total_lca_impact['total_co2_savings'] * 0.1  # Estimated
        total_lca_impact['total_waste_reduction'] = total_lca_impact['total_co2_savings'] * 0.8  # Estimated
        
        # Determine sustainability rating
        if total_lca_impact['circularity_score'] > 0.8:
            total_lca_impact['sustainability_rating'] = 'A+'
        elif total_lca_impact['circularity_score'] > 0.6:
            total_lca_impact['sustainability_rating'] = 'A'
        elif total_lca_impact['circularity_score'] > 0.4:
            total_lca_impact['sustainability_rating'] = 'B'
        else:
            total_lca_impact['sustainability_rating'] = 'C'
        
        return jsonify({
            'success': True,
            'lca_impact': total_lca_impact,
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

def run_server(host='0.0.0.0', port=5001, debug=False):
    """Run the recycling API server"""
    print("="*60)
    print("AI-DRIVEN LCA RECYCLING SUGGESTION API SERVER")
    print("="*60)
    
    # Initialize model
    initialize_model()
    
    print(f"Starting API server on {host}:{port}")
    print("Available endpoints:")
    print("  GET  /api/recycling/health - Health check")
    print("  POST /api/recycling/suggest - Single recycling suggestion")
    print("  POST /api/recycling/suggest/batch - Batch recycling suggestions")
    print("  GET  /api/recycling/materials - Available material types")
    print("  GET  /api/recycling/products - Recyclable products")
    print("  GET  /api/recycling/material/<type> - Material information")
    print("  GET  /api/recycling/statistics - Recycling statistics")
    print("  POST /api/recycling/optimize - Optimize recycling strategy")
    print("  POST /api/recycling/lca/calculate - Calculate LCA impact")
    print("="*60)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server(debug=True)
