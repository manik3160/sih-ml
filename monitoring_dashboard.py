

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.animation import FuncAnimation
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import dash
# from dash import dcc, html, Input, Output, callback
# import json
# from datetime import datetime, timedelta
# import threading
# import time
# from gas_leakage_predictor import GasLeakagePredictor
# from iot_sensor_system import IoTSensorNetwork, AlertHandler

# class MonitoringDashboard:
#     """
#     Real-time monitoring dashboard for gas leakage detection
#     """
    
#     def __init__(self, predictor: GasLeakagePredictor, sensor_network: IoTSensorNetwork):
#         self.predictor = predictor
#         self.sensor_network = sensor_network
#         self.dashboard_data = {
#             'timestamps': [],
#             'locations': [],
#             'leakage_probabilities': [],
#             'gas_concentrations': [],
#             'warnings': [],
#             'alerts': []
#         }
#         self.update_interval = 5  # seconds
        
#     def create_static_visualizations(self):
#         """Create static visualizations for the dataset"""
#         print("Creating static visualizations...")
        
#         # Set style
#         plt.style.use('seaborn-v0_8')
#         fig, axes = plt.subplots(2, 3, figsize=(20, 12))
#         fig.suptitle('Gas Leakage Detection System - Comprehensive Analysis', 
#                      fontsize=16, fontweight='bold')
        
#         # 1. Gas concentrations over time
#         gas_columns = ['CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm']
#         time_data = self.predictor.df.set_index('timestamp')
        
#         for i, gas in enumerate(gas_columns):
#             if i < 3:
#                 axes[0, i].plot(time_data.index, time_data[gas], label=gas, linewidth=2)
#                 axes[0, i].set_title(f'{gas} Concentration Over Time')
#                 axes[0, i].set_ylabel('Concentration (ppm)')
#                 axes[0, i].legend()
#                 axes[0, i].grid(True, alpha=0.3)
#             else:
#                 axes[1, i-3].plot(time_data.index, time_data[gas], label=gas, linewidth=2)
#                 axes[1, i-3].set_title(f'{gas} Concentration Over Time')
#                 axes[1, i-3].set_ylabel('Concentration (ppm)')
#                 axes[1, i-3].legend()
#                 axes[1, i-3].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('gas_concentrations_timeline.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # 2. Leakage detection heatmap
#         self._create_leakage_heatmap()
        
#         # 3. Environmental factors correlation
#         self._create_correlation_heatmap()
        
#         # 4. Equipment condition analysis
#         self._create_equipment_analysis()
        
#         # 5. Process area risk assessment
#         self._create_risk_assessment()
        
#         # 6. Warning levels distribution
#         self._create_warning_distribution()
    
#     def _create_leakage_heatmap(self):
#         """Create leakage detection heatmap by location and time"""
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         # Create pivot table for heatmap
#         df_pivot = self.predictor.df.pivot_table(
#             values='leakage_detected',
#             index='location_name',
#             columns=self.predictor.df['timestamp'].dt.hour,
#             aggfunc='mean'
#         )
        
#         sns.heatmap(df_pivot, annot=True, cmap='Reds', ax=ax, cbar_kws={'label': 'Leakage Rate'})
#         ax.set_title('Leakage Detection Heatmap by Location and Hour')
#         ax.set_xlabel('Hour of Day')
#         ax.set_ylabel('Location')
        
#         plt.tight_layout()
#         plt.savefig('leakage_heatmap.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def _create_correlation_heatmap(self):
#         """Create correlation heatmap for environmental factors"""
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         # Select numeric columns for correlation
#         numeric_cols = [
#             'CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm',
#             'temperature_C', 'humidity_percent', 'pressure_kPa', 'wind_speed_ms',
#             'vibration_level_mm_s', 'equipment_age_months', 'maintenance_days_ago',
#             'production_rate_percent', 'leakage_detected'
#         ]
        
#         corr_matrix = self.predictor.df[numeric_cols].corr()
        
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
#                    square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
#         ax.set_title('Environmental Factors Correlation Matrix')
        
#         plt.tight_layout()
#         plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def _create_equipment_analysis(self):
#         """Create equipment condition analysis"""
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Equipment age vs leakage
#         axes[0].scatter(self.predictor.df['equipment_age_months'], 
#                        self.predictor.df['leakage_detected'], 
#                        alpha=0.6, s=50)
#         axes[0].set_xlabel('Equipment Age (months)')
#         axes[0].set_ylabel('Leakage Detected')
#         axes[0].set_title('Equipment Age vs Leakage Detection')
#         axes[0].grid(True, alpha=0.3)
        
#         # Maintenance days vs leakage
#         axes[1].scatter(self.predictor.df['maintenance_days_ago'], 
#                        self.predictor.df['leakage_detected'], 
#                        alpha=0.6, s=50, color='orange')
#         axes[1].set_xlabel('Days Since Last Maintenance')
#         axes[1].set_ylabel('Leakage Detected')
#         axes[1].set_title('Maintenance Schedule vs Leakage Detection')
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('equipment_analysis.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def _create_risk_assessment(self):
#         """Create process area risk assessment"""
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         # Calculate risk score for each process area
#         risk_data = self.predictor.df.groupby('process_area').agg({
#             'leakage_detected': 'sum',
#             'CO2_ppm': 'mean',
#             'CO_ppm': 'mean',
#             'SO2_ppm': 'mean',
#             'CH4_ppm': 'mean',
#             'H2S_ppm': 'mean',
#             'NOx_ppm': 'mean'
#         }).reset_index()
        
#         # Calculate risk score (weighted combination)
#         risk_data['risk_score'] = (
#             risk_data['leakage_detected'] * 0.4 +
#             (risk_data['CO2_ppm'] / 1000) * 0.1 +
#             (risk_data['CO_ppm'] / 50) * 0.1 +
#             (risk_data['SO2_ppm'] / 5) * 0.1 +
#             (risk_data['CH4_ppm'] / 1000) * 0.1 +
#             (risk_data['H2S_ppm'] / 10) * 0.1 +
#             (risk_data['NOx_ppm'] / 25) * 0.1
#         )
        
#         # Sort by risk score
#         risk_data = risk_data.sort_values('risk_score', ascending=True)
        
#         # Create horizontal bar chart
#         bars = ax.barh(risk_data['process_area'], risk_data['risk_score'], 
#                       color=plt.cm.Reds(np.linspace(0.3, 1, len(risk_data))))
        
#         ax.set_xlabel('Risk Score')
#         ax.set_title('Process Area Risk Assessment')
#         ax.grid(True, alpha=0.3, axis='x')
        
#         # Add value labels on bars
#         for i, bar in enumerate(bars):
#             width = bar.get_width()
#             ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
#                    f'{width:.2f}', ha='left', va='center')
        
#         plt.tight_layout()
#         plt.savefig('risk_assessment.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def _create_warning_distribution(self):
#         """Create warning levels distribution"""
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Warning levels pie chart
#         warning_counts = self.predictor.df['warning_level'].value_counts()
#         axes[0].pie(warning_counts.values, labels=warning_counts.index, 
#                    autopct='%1.1f%%', startangle=90)
#         axes[0].set_title('Warning Levels Distribution')
        
#         # Severity levels bar chart
#         severity_counts = self.predictor.df['leakage_severity'].value_counts()
#         bars = axes[1].bar(severity_counts.index, severity_counts.values, 
#                           color=plt.cm.Set3(np.linspace(0, 1, len(severity_counts))))
#         axes[1].set_title('Leakage Severity Distribution')
#         axes[1].set_xlabel('Severity Level')
#         axes[1].set_ylabel('Count')
#         axes[1].tick_params(axis='x', rotation=45)
        
#         # Add value labels on bars
#         for bar in bars:
#             height = bar.get_height()
#             axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                         f'{int(height)}', ha='center', va='bottom')
        
#         plt.tight_layout()
#         plt.savefig('warning_distribution.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def create_interactive_dashboard(self):
#         """Create interactive Plotly dashboard"""
#         print("Creating interactive dashboard...")
        
#         # Create subplots
#         fig = make_subplots(
#             rows=3, cols=2,
#             subplot_titles=('Gas Concentrations Over Time', 'Leakage Detection by Location',
#                           'Environmental Factors', 'Equipment Condition',
#                           'Warning Levels', 'Risk Assessment'),
#             specs=[[{"secondary_y": True}, {"type": "bar"}],
#                    [{"type": "scatter"}, {"type": "scatter"}],
#                    [{"type": "pie"}, {"type": "bar"}]]
#         )
        
#         # 1. Gas concentrations over time
#         gas_columns = ['CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm']
#         time_data = self.predictor.df.set_index('timestamp')
        
#         for gas in gas_columns:
#             fig.add_trace(
#                 go.Scatter(x=time_data.index, y=time_data[gas], 
#                           name=gas, mode='lines+markers'),
#                 row=1, col=1
#             )
        
#         # 2. Leakage detection by location
#         location_leakage = self.predictor.df.groupby('location_name')['leakage_detected'].sum()
#         fig.add_trace(
#             go.Bar(x=location_leakage.index, y=location_leakage.values, 
#                   name='Leakage Incidents'),
#             row=1, col=2
#         )
        
#         # 3. Environmental factors scatter
#         fig.add_trace(
#             go.Scatter(x=self.predictor.df['temperature_C'], 
#                       y=self.predictor.df['humidity_percent'],
#                       mode='markers',
#                       marker=dict(size=self.predictor.df['leakage_detected'] * 10 + 5,
#                                 color=self.predictor.df['leakage_detected'],
#                                 colorscale='Reds'),
#                       name='Environmental Factors'),
#             row=2, col=1
#         )
        
#         # 4. Equipment condition
#         fig.add_trace(
#             go.Scatter(x=self.predictor.df['equipment_age_months'], 
#                       y=self.predictor.df['maintenance_days_ago'],
#                       mode='markers',
#                       marker=dict(size=self.predictor.df['leakage_detected'] * 15 + 5,
#                                 color=self.predictor.df['leakage_detected'],
#                                 colorscale='Blues'),
#                       name='Equipment Condition'),
#             row=2, col=2
#         )
        
#         # 5. Warning levels pie chart
#         warning_counts = self.predictor.df['warning_level'].value_counts()
#         fig.add_trace(
#             go.Pie(labels=warning_counts.index, values=warning_counts.values,
#                   name="Warning Levels"),
#             row=3, col=1
#         )
        
#         # 6. Risk assessment
#         risk_data = self.predictor.df.groupby('process_area')['leakage_detected'].sum()
#         fig.add_trace(
#             go.Bar(x=risk_data.index, y=risk_data.values, 
#                   name='Risk by Process Area'),
#             row=3, col=2
#         )
        
#         # Update layout
#         fig.update_layout(
#             title_text="Gas Leakage Detection System - Interactive Dashboard",
#             showlegend=True,
#             height=1200
#         )
        
#         # Save interactive dashboard
#         fig.write_html("interactive_dashboard.html")
#         print("Interactive dashboard saved as 'interactive_dashboard.html'")
        
#         return fig
    
#     def start_real_time_monitoring(self):
#         """Start real-time monitoring with live updates"""
#         print("Starting real-time monitoring dashboard...")
        
#         # Initialize alert handler
#         alert_handler = AlertHandler()
#         self.sensor_network.add_alert_callback(alert_handler.handle_alert)
        
#         # Start sensor monitoring
#         self.sensor_network.start_monitoring(interval=2.0)
        
#         try:
#             # Monitor for a specified duration
#             print("Real-time monitoring active. Press Ctrl+C to stop.")
#             while True:
#                 # Get latest predictions
#                 predictions = self.sensor_network.get_latest_predictions(limit=10)
                
#                 if predictions:
#                     print(f"\n--- Latest Predictions ({datetime.now().strftime('%H:%M:%S')}) ---")
#                     for pred in predictions:
#                         print(f"Location: {pred['location']}")
#                         print(f"Leakage: {pred['prediction']['leakage_detected']}")
#                         print(f"Probability: {pred['prediction']['leakage_probability']:.4f}")
#                         print(f"Warning Level: {pred['prediction']['warning_level']}")
#                         print("-" * 40)
                
#                 time.sleep(5)  # Update every 5 seconds
                
#         except KeyboardInterrupt:
#             print("\nStopping real-time monitoring...")
#         finally:
#             self.sensor_network.stop_monitoring()
#             print("Real-time monitoring stopped.")

# def create_dash_app(predictor: GasLeakagePredictor, sensor_network: IoTSensorNetwork):
#     """Create a Dash web application for the dashboard"""
    
#     app = dash.Dash(__name__)
    
#     app.layout = html.Div([
#         html.H1("Gas Leakage Detection System - Real-time Dashboard", 
#                 style={'textAlign': 'center', 'color': '#2c3e50'}),
        
#         html.Div([
#             html.Div([
#                 html.H3("System Status"),
#                 html.Div(id="system-status"),
#             ], className="six columns"),
            
#             html.Div([
#                 html.H3("Latest Alerts"),
#                 html.Div(id="latest-alerts"),
#             ], className="six columns"),
#         ], className="row"),
        
#         html.Div([
#             dcc.Graph(id="gas-concentrations"),
#         ], className="row"),
        
#         html.Div([
#             dcc.Graph(id="leakage-predictions"),
#         ], className="row"),
        
#         dcc.Interval(
#             id='interval-component',
#             interval=5*1000,  # Update every 5 seconds
#             n_intervals=0
#         )
#     ])
    
#     @app.callback(
#         [Output('system-status', 'children'),
#          Output('latest-alerts', 'children'),
#          Output('gas-concentrations', 'figure'),
#          Output('leakage-predictions', 'figure')],
#         [Input('interval-component', 'n_intervals')]
#     )
#     def update_dashboard(n):
#         # Get sensor status
#         status = sensor_network.get_sensor_status()
        
#         # Get latest predictions
#         predictions = sensor_network.get_latest_predictions(limit=5)
        
#         # Create system status
#         system_status = html.Div([
#             html.P(f"Total Sensors: {status['total_sensors']}"),
#             html.P(f"Active Sensors: {status['active_sensors']}"),
#             html.P(f"Inactive Sensors: {status['inactive_sensors']}"),
#         ])
        
#         # Create latest alerts
#         alerts_html = []
#         for pred in predictions:
#             if pred['prediction']['leakage_detected']:
#                 alerts_html.append(html.Div([
#                     html.P(f"🚨 {pred['location']}: Leakage detected!"),
#                     html.P(f"Probability: {pred['prediction']['leakage_probability']:.4f}"),
#                 ], style={'color': 'red', 'margin': '5px'}))
        
#         if not alerts_html:
#             alerts_html = [html.P("No active alerts", style={'color': 'green'})]
        
#         # Create gas concentrations plot
#         gas_fig = go.Figure()
#         gas_columns = ['CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm']
        
#         for gas in gas_columns:
#             gas_fig.add_trace(go.Scatter(
#                 x=predictor.df['timestamp'],
#                 y=predictor.df[gas],
#                 mode='lines+markers',
#                 name=gas
#             ))
        
#         gas_fig.update_layout(
#             title="Gas Concentrations Over Time",
#             xaxis_title="Time",
#             yaxis_title="Concentration (ppm)"
#         )
        
#         # Create leakage predictions plot
#         leakage_fig = go.Figure()
        
#         if predictions:
#             locations = [p['location'] for p in predictions]
#             probabilities = [p['prediction']['leakage_probability'] for p in predictions]
            
#             leakage_fig.add_trace(go.Bar(
#                 x=locations,
#                 y=probabilities,
#                 name='Leakage Probability'
#             ))
        
#         leakage_fig.update_layout(
#             title="Latest Leakage Predictions",
#             xaxis_title="Location",
#             yaxis_title="Leakage Probability"
#         )
        
#         return system_status, alerts_html, gas_fig, leakage_fig
    
#     return app

# def main():
#     """Main function to run the monitoring dashboard"""
#     print("="*60)
#     print("MONITORING DASHBOARD")
#     print("Gas Leakage Detection System")
#     print("="*60)
    
#     # Initialize predictor
#     print("Loading ML models...")
#     predictor = GasLeakagePredictor()
#     predictor.load_data()
#     predictor.train_models()
    
#     # Initialize sensor networks
#     print("Initializing sensor network...")
#     sensor_network = IoTSensorNetwork(predictor)
    
#     # Initialize dashboard
#     dashboard = MonitoringDashboard(predictor, sensor_network)
    
#     # Create static visualizations
#     print("Creating static visualizations...")
#     dashboard.create_static_visualizations()
    
#     # Create interactive dashboard
#     print("Creating interactive dashboard...")
#     dashboard.create_interactive_dashboard()
    
#     # Start real-time monitoring
#     print("Starting real-time monitoring...")
#     dashboard.start_real_time_monitoring()

# if __name__ == "__main__":
#     main()
