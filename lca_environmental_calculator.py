

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class LCAEnvironmentalCalculator:
    """
    LCA Environmental Impact Calculator for mining and metallurgy processes
    """
    
    def __init__(self):
        """
        Initialize the LCA calculator with industry-specific parameters
        """
        # Global Warming Potential (GWP) values (CO2 equivalent)
        self.gwp_factors = {
            'CO2': 1.0,      # CO2 equivalent
            'CH4': 25.0,     # 25x more potent than CO2
            'N2O': 298.0,    # 298x more potent than CO2
            'SO2': 0.0,      # Not a direct greenhouse gas
            'CO': 0.0,       # Not a direct greenhouse gas
            'H2S': 0.0,      # Not a direct greenhouse gas
            'NOx': 0.0       # Not a direct greenhouse gas (but contributes to smog)
        }
        
        # Environmental impact factors (per kg of gas)
        self.impact_factors = {
            'CO2': {
                'global_warming': 1.0,
                'acidification': 0.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            },
            'CH4': {
                'global_warming': 25.0,
                'acidification': 0.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            },
            'SO2': {
                'global_warming': 0.0,
                'acidification': 1.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            },
            'NOx': {
                'global_warming': 0.0,
                'acidification': 0.7,
                'eutrophication': 0.13,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            },
            'CO': {
                'global_warming': 0.0,
                'acidification': 0.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.03
            },
            'H2S': {
                'global_warming': 0.0,
                'acidification': 0.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            }
        }
        
        # Process-specific emission factors (kg CO2e per unit production)
        self.process_emission_factors = {
            'Open_Pit': 0.15,        # kg CO2e per kg ore
            'Processing': 0.08,      # kg CO2e per kg processed material
            'Smelting': 0.45,        # kg CO2e per kg metal
            'Refining': 0.12,        # kg CO2e per kg refined metal
            'Waste_Management': 0.05  # kg CO2e per kg waste
        }
        
        # Energy consumption factors (kWh per unit)
        self.energy_factors = {
            'Open_Pit': 0.05,        # kWh per kg ore
            'Processing': 0.12,      # kWh per kg processed material
            'Smelting': 0.8,         # kWh per kg metal
            'Refining': 0.15,        # kWh per kg refined metal
            'Waste_Management': 0.02  # kWh per kg waste
        }
        
        # Water consumption factors (L per unit)
        self.water_factors = {
            'Open_Pit': 0.1,         # L per kg ore
            'Processing': 0.5,       # L per kg processed material
            'Smelting': 2.0,         # L per kg metal
            'Refining': 0.8,         # L per kg refined metal
            'Waste_Management': 0.2   # L per kg waste
        }
    
    def calculate_gas_emissions_impact(self, gas_data: Dict) -> Dict:
        """
        Calculate environmental impact of gas emissions
        
        Args:
            gas_data (Dict): Dictionary containing gas concentrations in ppm
            
        Returns:
            Dict: Environmental impact calculations
        """
        # Convert ppm to kg (assuming 1 m³ of air = 1.225 kg at standard conditions)
        volume_factor = 1.225e-6  # kg per ppm per m³
        
        impacts = {
            'total_co2e': 0.0,
            'individual_impacts': {},
            'impact_categories': {
                'global_warming': 0.0,
                'acidification': 0.0,
                'eutrophication': 0.0,
                'ozone_depletion': 0.0,
                'smog_formation': 0.0
            }
        }
        
        # Calculate impacts for each gas
        for gas, concentration in gas_data.items():
            if gas in self.gwp_factors and gas in self.impact_factors:
                # Convert to kg
                mass_kg = concentration * volume_factor
                
                # Calculate CO2 equivalent
                co2e = mass_kg * self.gwp_factors[gas]
                impacts['total_co2e'] += co2e
                
                # Calculate individual impacts
                impacts['individual_impacts'][gas] = {
                    'mass_kg': mass_kg,
                    'co2e': co2e,
                    'concentration_ppm': concentration
                }
                
                # Calculate impact categories
                for category, factor in self.impact_factors[gas].items():
                    impacts['impact_categories'][category] += mass_kg * factor
        
        return impacts
    
    def calculate_process_environmental_impact(self, process_data: Dict) -> Dict:
        """
        Calculate environmental impact of mining/metallurgy processes
        
        Args:
            process_data (Dict): Process data including production rates, process areas
            
        Returns:
            Dict: Process environmental impact calculations
        """
        impacts = {
            'carbon_footprint': 0.0,
            'energy_consumption': 0.0,
            'water_consumption': 0.0,
            'process_impacts': {}
        }
        
        # Calculate impacts for each process area
        for process_area, production_rate in process_data.items():
            if process_area in self.process_emission_factors:
                # Carbon footprint
                carbon_impact = production_rate * self.process_emission_factors[process_area]
                impacts['carbon_footprint'] += carbon_impact
                
                # Energy consumption
                energy_impact = production_rate * self.energy_factors[process_area]
                impacts['energy_consumption'] += energy_impact
                
                # Water consumption
                water_impact = production_rate * self.water_factors[process_area]
                impacts['water_consumption'] += water_impact
                
                impacts['process_impacts'][process_area] = {
                    'carbon_footprint': carbon_impact,
                    'energy_consumption': energy_impact,
                    'water_consumption': water_impact,
                    'production_rate': production_rate
                }
        
        return impacts
    
    def calculate_leakage_environmental_cost(self, leakage_data: Dict) -> Dict:
        """
        Calculate environmental cost of gas leakage incidents
        
        Args:
            leakage_data (Dict): Leakage prediction data
            
        Returns:
            Dict: Environmental cost calculations
        """
        costs = {
            'immediate_impact': 0.0,
            'long_term_impact': 0.0,
            'remediation_cost': 0.0,
            'regulatory_fines': 0.0,
            'total_cost': 0.0,
            'cost_breakdown': {}
        }
        
        if leakage_data['leakage_detected']:
            # Calculate immediate environmental impact
            gas_impacts = self.calculate_gas_emissions_impact(leakage_data.get('gas_concentrations', {}))
            costs['immediate_impact'] = gas_impacts['total_co2e'] * 50  # $50 per kg CO2e
            
            # Calculate long-term impact based on severity
            severity_multipliers = {
                'low': 1.0,
                'medium': 2.5,
                'high': 5.0,
                'critical': 10.0
            }
            
            severity = leakage_data.get('severity', 'low')
            multiplier = severity_multipliers.get(severity, 1.0)
            costs['long_term_impact'] = costs['immediate_impact'] * multiplier
            
            # Calculate remediation costs
            remediation_base = 10000  # Base remediation cost
            costs['remediation_cost'] = remediation_base * multiplier
            
            # Calculate regulatory fines
            fine_base = 5000  # Base fine amount
            costs['regulatory_fines'] = fine_base * multiplier
            
            # Total cost
            costs['total_cost'] = (costs['immediate_impact'] + 
                                 costs['long_term_impact'] + 
                                 costs['remediation_cost'] + 
                                 costs['regulatory_fines'])
            
            # Cost breakdown
            costs['cost_breakdown'] = {
                'immediate_impact_percent': (costs['immediate_impact'] / costs['total_cost']) * 100,
                'long_term_impact_percent': (costs['long_term_impact'] / costs['total_cost']) * 100,
                'remediation_percent': (costs['remediation_cost'] / costs['total_cost']) * 100,
                'fines_percent': (costs['regulatory_fines'] / costs['total_cost']) * 100
            }
        
        return costs
    
    def calculate_sustainability_metrics(self, historical_data: pd.DataFrame) -> Dict:
        """
        Calculate sustainability metrics from historical data
        
        Args:
            historical_data (pd.DataFrame): Historical process and leakage data
            
        Returns:
            Dict: Sustainability metrics
        """
        metrics = {
            'leakage_frequency': 0.0,
            'environmental_efficiency': 0.0,
            'carbon_intensity': 0.0,
            'energy_efficiency': 0.0,
            'water_efficiency': 0.0,
            'sustainability_score': 0.0,
            'improvement_recommendations': []
        }
        
        # Calculate leakage frequency (leakages per day)
        total_days = (historical_data['timestamp'].max() - historical_data['timestamp'].min()).days
        if total_days > 0:
            total_leakages = historical_data['leakage_detected'].sum()
            metrics['leakage_frequency'] = total_leakages / total_days
        
        # Calculate environmental efficiency
        total_production = historical_data['production_rate_percent'].mean()
        if total_production > 0:
            metrics['environmental_efficiency'] = (1 - metrics['leakage_frequency']) * (total_production / 100)
        
        # Calculate carbon intensity
        avg_gas_concentrations = {
            'CO2_ppm': historical_data['CO2_ppm'].mean(),
            'CH4_ppm': historical_data['CH4_ppm'].mean(),
            'SO2_ppm': historical_data['SO2_ppm'].mean(),
            'NOx_ppm': historical_data['NOx_ppm'].mean()
        }
        
        gas_impacts = self.calculate_gas_emissions_impact(avg_gas_concentrations)
        metrics['carbon_intensity'] = gas_impacts['total_co2e'] / total_production if total_production > 0 else 0
        
        # Calculate energy efficiency
        avg_equipment_age = historical_data['equipment_age_months'].mean()
        avg_maintenance_days = historical_data['maintenance_days_ago'].mean()
        metrics['energy_efficiency'] = max(0, 1 - (avg_equipment_age / 100) - (avg_maintenance_days / 30))
        
        # Calculate water efficiency
        avg_humidity = historical_data['humidity_percent'].mean()
        metrics['water_efficiency'] = avg_humidity / 100
        
        # Calculate overall sustainability score (0-100)
        metrics['sustainability_score'] = (
            (1 - metrics['leakage_frequency']) * 30 +
            metrics['environmental_efficiency'] * 25 +
            (1 - min(metrics['carbon_intensity'], 1)) * 20 +
            metrics['energy_efficiency'] * 15 +
            metrics['water_efficiency'] * 10
        ) * 100
        
        # Generate improvement recommendations
        if metrics['leakage_frequency'] > 0.1:
            metrics['improvement_recommendations'].append("High leakage frequency detected - implement preventive maintenance")
        
        if metrics['carbon_intensity'] > 0.5:
            metrics['improvement_recommendations'].append("High carbon intensity - optimize gas handling processes")
        
        if metrics['energy_efficiency'] < 0.7:
            metrics['improvement_recommendations'].append("Low energy efficiency - upgrade equipment and maintenance")
        
        if metrics['sustainability_score'] < 70:
            metrics['improvement_recommendations'].append("Overall sustainability score below target - comprehensive review needed")
        
        return metrics
    
    def create_environmental_report(self, prediction_data: Dict, process_data: Dict, 
                                  historical_data: pd.DataFrame) -> Dict:
        """
        Create comprehensive environmental impact report
        
        Args:
            prediction_data (Dict): Current leakage prediction data
            process_data (Dict): Current process data
            historical_data (pd.DataFrame): Historical data for trend analysis
            
        Returns:
            Dict: Comprehensive environmental report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': {},
            'current_impacts': {},
            'historical_trends': {},
            'recommendations': [],
            'compliance_status': {},
            'sustainability_metrics': {}
        }
        
        # Calculate current impacts
        if 'gas_concentrations' in prediction_data:
            gas_impacts = self.calculate_gas_emissions_impact(prediction_data['gas_concentrations'])
            report['current_impacts']['gas_emissions'] = gas_impacts
        
        process_impacts = self.calculate_process_environmental_impact(process_data)
        report['current_impacts']['process_impacts'] = process_impacts
        
        leakage_costs = self.calculate_leakage_environmental_cost(prediction_data)
        report['current_impacts']['leakage_costs'] = leakage_costs
        
        # Calculate historical trends
        sustainability_metrics = self.calculate_sustainability_metrics(historical_data)
        report['historical_trends'] = sustainability_metrics
        
        # Generate executive summary
        report['executive_summary'] = {
            'leakage_detected': prediction_data.get('leakage_detected', False),
            'leakage_probability': prediction_data.get('leakage_probability', 0.0),
            'total_co2e': gas_impacts.get('total_co2e', 0.0) if 'gas_impacts' in locals() else 0.0,
            'environmental_cost': leakage_costs.get('total_cost', 0.0),
            'sustainability_score': sustainability_metrics.get('sustainability_score', 0.0),
            'risk_level': self._assess_risk_level(prediction_data, sustainability_metrics)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_environmental_recommendations(
            prediction_data, process_impacts, sustainability_metrics
        )
        
        # Assess compliance status
        report['compliance_status'] = self._assess_compliance_status(
            prediction_data, gas_impacts if 'gas_impacts' in locals() else {}
        )
        
        return report
    
    def _assess_risk_level(self, prediction_data: Dict, sustainability_metrics: Dict) -> str:
        """Assess overall environmental risk level"""
        risk_factors = []
        
        if prediction_data.get('leakage_detected', False):
            risk_factors.append(3)
        
        if prediction_data.get('leakage_probability', 0) > 0.7:
            risk_factors.append(2)
        
        if sustainability_metrics.get('leakage_frequency', 0) > 0.1:
            risk_factors.append(2)
        
        if sustainability_metrics.get('sustainability_score', 0) < 70:
            risk_factors.append(1)
        
        total_risk = sum(risk_factors)
        
        if total_risk >= 6:
            return 'HIGH'
        elif total_risk >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_environmental_recommendations(self, prediction_data: Dict, 
                                              process_impacts: Dict, 
                                              sustainability_metrics: Dict) -> List[str]:
        """Generate environmental improvement recommendations"""
        recommendations = []
        
        # Leakage prevention recommendations
        if prediction_data.get('leakage_detected', False):
            recommendations.append("IMMEDIATE: Implement emergency containment measures")
            recommendations.append("IMMEDIATE: Deploy additional monitoring sensors")
        
        if prediction_data.get('leakage_probability', 0) > 0.5:
            recommendations.append("HIGH PRIORITY: Schedule preventive maintenance")
            recommendations.append("HIGH PRIORITY: Review and update safety protocols")
        
        # Process optimization recommendations
        if process_impacts.get('carbon_footprint', 0) > 1000:  # kg CO2e
            recommendations.append("MEDIUM PRIORITY: Optimize process efficiency to reduce carbon footprint")
        
        if process_impacts.get('energy_consumption', 0) > 500:  # kWh
            recommendations.append("MEDIUM PRIORITY: Implement energy efficiency measures")
        
        # Sustainability improvement recommendations
        if sustainability_metrics.get('sustainability_score', 0) < 80:
            recommendations.append("LONG TERM: Develop comprehensive sustainability improvement plan")
        
        if sustainability_metrics.get('leakage_frequency', 0) > 0.05:
            recommendations.append("LONG TERM: Invest in advanced monitoring and control systems")
        
        return recommendations
    
    def _assess_compliance_status(self, prediction_data: Dict, gas_impacts: Dict) -> Dict:
        """Assess environmental compliance status"""
        compliance = {
            'air_quality': 'COMPLIANT',
            'emission_limits': 'COMPLIANT',
            'safety_standards': 'COMPLIANT',
            'overall_status': 'COMPLIANT',
            'violations': []
        }
        
        # Check air quality compliance
        gas_thresholds = {
            'CO2_ppm': 1000,
            'CO_ppm': 50,
            'SO2_ppm': 5,
            'CH4_ppm': 1000,
            'H2S_ppm': 10,
            'NOx_ppm': 25
        }
        
        for gas, threshold in gas_thresholds.items():
            if gas in prediction_data.get('gas_concentrations', {}):
                if prediction_data['gas_concentrations'][gas] > threshold:
                    compliance['air_quality'] = 'NON-COMPLIANT'
                    compliance['violations'].append(f"{gas} exceeds limit: {prediction_data['gas_concentrations'][gas]} > {threshold}")
        
        # Check emission limits
        if gas_impacts.get('total_co2e', 0) > 100:  # kg CO2e per day
            compliance['emission_limits'] = 'NON-COMPLIANT'
            compliance['violations'].append(f"Daily CO2e emissions exceed limit: {gas_impacts['total_co2e']:.2f} kg")
        
        # Check safety standards
        if prediction_data.get('leakage_detected', False):
            compliance['safety_standards'] = 'NON-COMPLIANT'
            compliance['violations'].append("Active gas leakage detected")
        
        # Overall compliance status
        if any(status == 'NON-COMPLIANT' for status in [compliance['air_quality'], 
                                                        compliance['emission_limits'], 
                                                        compliance['safety_standards']]):
            compliance['overall_status'] = 'NON-COMPLIANT'
        
        return compliance
    
    def create_environmental_visualizations(self, report: Dict, save_path: str = "environmental_report.png"):
        """Create visualizations for environmental impact report"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Environmental Impact Assessment Report', fontsize=16, fontweight='bold')
        
        # 1. CO2 equivalent emissions
        if 'gas_emissions' in report['current_impacts']:
            gas_data = report['current_impacts']['gas_emissions']['individual_impacts']
            gases = list(gas_data.keys())
            co2e_values = [gas_data[gas]['co2e'] for gas in gases]
            
            axes[0, 0].bar(gases, co2e_values, color='red', alpha=0.7)
            axes[0, 0].set_title('CO2 Equivalent Emissions by Gas')
            axes[0, 0].set_ylabel('CO2e (kg)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Process carbon footprint
        if 'process_impacts' in report['current_impacts']:
            process_data = report['current_impacts']['process_impacts']['process_impacts']
            processes = list(process_data.keys())
            carbon_values = [process_data[proc]['carbon_footprint'] for proc in processes]
            
            axes[0, 1].pie(carbon_values, labels=processes, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Carbon Footprint by Process Area')
        
        # 3. Environmental cost breakdown
        if 'leakage_costs' in report['current_impacts']:
            cost_data = report['current_impacts']['leakage_costs']['cost_breakdown']
            cost_categories = ['immediate_impact', 'long_term_impact', 'remediation', 'fines']
            cost_values = [cost_data.get(f'{cat}_percent', 0) for cat in cost_categories]
            
            axes[0, 2].bar(cost_categories, cost_values, color='orange', alpha=0.7)
            axes[0, 2].set_title('Environmental Cost Breakdown')
            axes[0, 2].set_ylabel('Percentage (%)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Sustainability metrics
        if 'sustainability_metrics' in report['historical_trends']:
            metrics = report['historical_trends']['sustainability_metrics']
            metric_names = ['leakage_frequency', 'environmental_efficiency', 
                          'carbon_intensity', 'energy_efficiency', 'water_efficiency']
            metric_values = [metrics.get(name, 0) for name in metric_names]
            
            axes[1, 0].bar(metric_names, metric_values, color='green', alpha=0.7)
            axes[1, 0].set_title('Sustainability Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Compliance status
        compliance = report['compliance_status']
        compliance_categories = list(compliance.keys())[:-1]  # Exclude 'overall_status'
        compliance_values = [1 if compliance[cat] == 'COMPLIANT' else 0 for cat in compliance_categories]
        
        colors = ['green' if val == 1 else 'red' for val in compliance_values]
        axes[1, 1].bar(compliance_categories, compliance_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Compliance Status')
        axes[1, 1].set_ylabel('Compliant (1) / Non-Compliant (0)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Risk assessment
        risk_level = report['executive_summary']['risk_level']
        risk_colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'red'}
        risk_values = [1]
        risk_labels = [risk_level]
        
        axes[1, 2].bar(risk_labels, risk_values, color=risk_colors.get(risk_level, 'gray'), alpha=0.7)
        axes[1, 2].set_title('Environmental Risk Level')
        axes[1, 2].set_ylabel('Risk Level')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Environmental report visualizations saved as '{save_path}'")

def main():
    """Main function to demonstrate LCA environmental calculator"""
    print("="*60)
    print("LCA ENVIRONMENTAL IMPACT CALCULATOR")
    print("Mining and Metallurgy Industry")
    print("="*60)
    
    # Initialize calculator
    calculator = LCAEnvironmentalCalculator()
    
    # Sample data for demonstration
    sample_gas_data = {
        'CO2_ppm': 1200,
        'CO_ppm': 15,
        'SO2_ppm': 2.5,
        'CH4_ppm': 800,
        'H2S_ppm': 5,
        'NOx_ppm': 30
    }
    
    sample_process_data = {
        'Open_Pit': 1000,      # kg ore
        'Processing': 800,     # kg processed material
        'Smelting': 200,       # kg metal
        'Refining': 150,       # kg refined metal
        'Waste_Management': 500 # kg waste
    }
    
    sample_prediction_data = {
        'leakage_detected': True,
        'leakage_probability': 0.85,
        'severity': 'high',
        'gas_concentrations': sample_gas_data
    }
    
    # Load historical data
    try:
        historical_data = pd.read_csv('parsed_dataset.csv')
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    except FileNotFoundError:
        print("Historical data not found. Using sample data.")
        historical_data = pd.DataFrame()
    
    # Calculate environmental impacts
    print("Calculating gas emissions impact...")
    gas_impacts = calculator.calculate_gas_emissions_impact(sample_gas_data)
    print(f"Total CO2 equivalent: {gas_impacts['total_co2e']:.4f} kg")
    
    print("\nCalculating process environmental impact...")
    process_impacts = calculator.calculate_process_environmental_impact(sample_process_data)
    print(f"Total carbon footprint: {process_impacts['carbon_footprint']:.2f} kg CO2e")
    print(f"Total energy consumption: {process_impacts['energy_consumption']:.2f} kWh")
    print(f"Total water consumption: {process_impacts['water_consumption']:.2f} L")
    
    print("\nCalculating leakage environmental cost...")
    leakage_costs = calculator.calculate_leakage_environmental_cost(sample_prediction_data)
    print(f"Total environmental cost: ${leakage_costs['total_cost']:.2f}")
    
    if not historical_data.empty:
        print("\nCalculating sustainability metrics...")
        sustainability_metrics = calculator.calculate_sustainability_metrics(historical_data)
        print(f"Sustainability score: {sustainability_metrics['sustainability_score']:.1f}/100")
        print(f"Leakage frequency: {sustainability_metrics['leakage_frequency']:.4f} per day")
    
    # Create comprehensive environmental report
    print("\nCreating environmental impact report...")
    report = calculator.create_environmental_report(
        sample_prediction_data, 
        sample_process_data, 
        historical_data
    )
    
    # Print executive summary
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    summary = report['executive_summary']
    print(f"Leakage Detected: {summary['leakage_detected']}")
    print(f"Leakage Probability: {summary['leakage_probability']:.4f}")
    print(f"Total CO2e: {summary['total_co2e']:.4f} kg")
    print(f"Environmental Cost: ${summary['environmental_cost']:.2f}")
    print(f"Sustainability Score: {summary['sustainability_score']:.1f}/100")
    print(f"Risk Level: {summary['risk_level']}")
    
    # Print compliance status
    print("\n" + "="*50)
    print("COMPLIANCE STATUS")
    print("="*50)
    compliance = report['compliance_status']
    print(f"Overall Status: {compliance['overall_status']}")
    print(f"Air Quality: {compliance['air_quality']}")
    print(f"Emission Limits: {compliance['emission_limits']}")
    print(f"Safety Standards: {compliance['safety_standards']}")
    
    if compliance['violations']:
        print("\nViolations:")
        for violation in compliance['violations']:
            print(f"  - {violation}")
    
    # Print recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Create visualizations
    print("\nCreating environmental visualizations...")
    calculator.create_environmental_visualizations(report)
    
    # Save report
    with open('environmental_impact_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nEnvironmental impact report saved as 'environmental_impact_report.json'")
    print("\n" + "="*60)
    print("LCA ENVIRONMENTAL CALCULATOR COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
