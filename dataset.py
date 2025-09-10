import pandas as pd
import random

# Number of rows
num_entries = 150

# Sample values
material_types = [
    "Copper Slag", "Iron Scrap", "Bauxite Residue", "Zinc Ash", "Tailings", 
    "Steel Slag", "Aluminium Scrap", "Nickel Dust", "Lead Slag", "Chromite Waste"
]
source_processes = [
    "Smelting", "Steel Manufacturing", "Alumina Refining", "Galvanizing", 
    "Gold Mining", "Furnace Operation", "Casting", "Electroplating", 
    "Battery Recycling", "Ore Processing"
]
contaminants_list = [
    "Arsenic", "Lead", "Oil Residue", "Sodium Hydroxide", "Chlorides", 
    "Cyanide", "Silicates", "Paint", "Plastic", "Sulfates", "Cadmium", 
    "Mercury", "Chromium", "Fluoride", "Ammonia"
]
reuse_suggestions = [
    "Extract copper for cables", "Re-smelt for rebar", "Safe landfill with encapsulation",
    "Reprocess to recover zinc", "Detoxification for backfill use", "Use in road construction",
    "Remelt to make sheet metal", "Compress into briquettes for reuse", "Use in cement production",
    "Solidify for construction fill", "Reuse in battery production", "Reprocess to recover lead"
]

# Generate data
data = []
for i in range(num_entries):
    material_type = random.choice(material_types)
    source_process = random.choice(source_processes)
    quantity = random.randint(100, 3000)
    purity = round(random.uniform(10, 99), 2)
    contaminants = ", ".join(random.sample(contaminants_list, random.randint(1, 3)))
    energy = round(random.uniform(40, 600), 1)
    toxicity = round(random.uniform(3.0, 9.5), 1)
    carbon = round(quantity * random.uniform(0.1, 0.4), 1)
    recyclable = "Yes" if purity > 50 and "Cyanide" not in contaminants else "No"
    reuse = random.choice(reuse_suggestions) if recyclable == "Yes" else "Safe disposal required"
    
    data.append({
        "Material_ID": f"M{i+1:03}",
        "Material_Type": material_type,
        "Source_Process": source_process,
        "Quantity_kg": quantity,
        "Purity_%": purity,
        "Contaminants": contaminants,
        "Energy_Content_MJ": energy,
        "Toxicity_Score": toxicity,
        "Carbon_Footprint_kgCO2": carbon,
        "Recyclable": recyclable,
        "Suggested_Reuse": reuse
    })

# Save as CSV
df = pd.DataFrame(data)
df.to_csv("ai_lca_mining_dataset_150.csv", index=False)