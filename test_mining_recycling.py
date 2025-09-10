"""
Test script for AI-Driven LCA Recycling Model with Mining Dataset
Demonstrates recycling suggestions for mining materials
"""

from recycling_suggestion_model import RecyclingSuggestionModel
import pandas as pd

def test_mining_recycling():
    """Test recycling suggestions with mining dataset"""
    print("="*60)
    print("TESTING AI-DRIVEN LCA RECYCLING MODEL")
    print("WITH MINING DATASET")
    print("="*60)
    
    # Initialize model
    model = RecyclingSuggestionModel()
    
    # Load mining dataset
    print("Loading mining dataset...")
    mining_df = model.load_mining_dataset()
    
    if mining_df is None:
        print("Error: Could not load mining dataset")
        return
    
    # Train models
    print("\nTraining models...")
    model.train_models()
    
    print(f"\nDataset loaded: {len(mining_df)} materials")
    print(f"Material types: {mining_df['Material_Type'].unique()}")
    print(f"Recyclable materials: {mining_df['Recyclable'].value_counts()['Yes']}")
    
    # Test with different materials from your dataset
    test_materials = [
        mining_df.iloc[0],   # Bauxite Residue
        mining_df.iloc[1],   # Aluminium Scrap
        mining_df.iloc[2],   # Chromite Waste
        mining_df.iloc[5],   # Iron Scrap
        mining_df.iloc[6],   # Steel Slag
    ]
    
    print("\n" + "="*60)
    print("RECYCLING SUGGESTIONS FOR MINING MATERIALS")
    print("="*60)
    
    for i, material in enumerate(test_materials, 1):
        print(f"\n{i}. MATERIAL: {material['Material_ID']} - {material['Material_Type']}")
        print(f"   Source: {material['Source_Process']}")
        print(f"   Quantity: {material['Quantity_kg']} kg")
        print(f"   Purity: {material['Purity_%']}%")
        print(f"   Toxicity: {material['Toxicity_Score']}/10")
        print(f"   Recyclable: {material['Recyclable']}")
        print(f"   Original Suggestion: {material['Suggested_Reuse']}")
        
        # Convert to dict for prediction
        material_dict = material.to_dict()
        
        # Get AI recycling suggestion
        suggestion = model.suggest_recycling(material_dict)
        
        print(f"\n   🤖 AI RECYCLING SUGGESTION:")
        print(f"   • Suggested Product: {suggestion['suggested_product']}")
        print(f"   • Priority Level: {suggestion['recycling_priority']}")
        print(f"   • Feasibility Score: {suggestion['feasibility_score']:.3f}")
        print(f"   • Recycling Efficiency: {suggestion['recycling_efficiency']:.1%}")
        print(f"   • Energy Savings: {suggestion['energy_savings']:.1%}")
        print(f"   • CO2 Reduction: {suggestion['co2_reduction']:.1%}")
        print(f"   • Market Value: {suggestion['market_value']:.1%}")
        
        print(f"\n   🌍 ENVIRONMENTAL IMPACT:")
        print(f"   • CO2 Savings: {suggestion['environmental_impact']['co2_savings_kg']:.1f} kg")
        print(f"   • Energy Savings: {suggestion['environmental_impact']['energy_savings_kwh']:.1f} kWh")
        print(f"   • Circularity Score: {suggestion['environmental_impact']['circularity_score']:.3f}")
        
        print(f"\n   📋 RECOMMENDATIONS:")
        for j, rec in enumerate(suggestion['recommendations'][:3], 1):
            print(f"   {j}. {rec}")
        
        print("-" * 60)
    
    # Summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total Materials: {len(mining_df)}")
    print(f"Recyclable Materials: {mining_df['Recyclable'].value_counts()['Yes']}")
    print(f"Non-Recyclable Materials: {mining_df['Recyclable'].value_counts()['No']}")
    
    print(f"\nMaterial Type Distribution:")
    material_counts = mining_df['Material_Type'].value_counts()
    for material, count in material_counts.items():
        print(f"  {material}: {count} materials")
    
    print(f"\nToxicity Distribution:")
    toxicity_counts = mining_df['Toxicity_Score'].value_counts().sort_index()
    for score, count in toxicity_counts.items():
        print(f"  Score {score}: {count} materials")
    
    print(f"\nSource Process Distribution:")
    process_counts = mining_df['Source_Process'].value_counts()
    for process, count in process_counts.items():
        print(f"  {process}: {count} materials")
    
    print("\n" + "="*60)
    print("AI-DRIVEN LCA RECYCLING MODEL READY!")
    print("Ready for frontend integration and API deployment")
    print("="*60)

if __name__ == "__main__":
    test_mining_recycling()
