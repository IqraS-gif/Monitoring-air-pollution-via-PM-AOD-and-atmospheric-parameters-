!pip install pandas numpy scikit-learn xgboost matplotlib
!pip install xgboost joblib

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import folium
from branca.colormap import LinearColormap
from ipywidgets import interact, widgets
from datetime import datetime
from IPython.display import display, clear_output
import ipywidgets as wgt
from google.colab import output

# Enable larger widget output
output.enable_custom_widget_manager()

# City coordinates and data ranges with zoom levels
CITY_DATA = {
    'Ramanathapuram': {
        'coords': [9.37, 78.83],
        'zoom': 11
    },
    'Delhi': {
        'coords': [28.65, 77.22],  # Precise coordinates for Delhi
        'zoom': 11  # Lower zoom for larger area
    },
    'Mumbai': {
        'coords': [19.06, 72.86],
        'zoom': 9
    }
}

# --------------------------
# 1. Enhanced Data Loading & Feature Engineering
# --------------------------
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Detect city automatically
    if 'Delhi' in filepath:
        city = 'Delhi'
    elif 'Mumbai' in filepath:
        city = 'Mumbai'
    else:
        city = 'Ramanathapuram'
    df['city'] = city

    # Feature Engineering
    df['WIND_SPEED'] = np.sqrt(df['U10M']**2 + df['V10M']**2)
    df['AOD_PBLH_INTERACTION'] = df['TOTEXTTAU'] / (df['PBLH'].replace(0, 1e-6))
    df['AOD_WIND_INTERACTION'] = df['TOTEXTTAU'] * df['WIND_SPEED']

    # Temporal Features
    df['HOUR'] = df['time'].dt.hour
    df['IS_DAYTIME'] = df['HOUR'].between(6, 18).astype(int)
    df['DAY_OF_WEEK'] = df['time'].dt.dayofweek
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

    # Lag Features for both PM types
    for pm_type in ['PM2.5', 'PM10']:
        df[f'{pm_type}_24H_MEAN'] = df[pm_type].rolling(24, min_periods=1).mean()
        df[f'{pm_type}_DIURNAL_ADJUSTED'] = df[pm_type].shift(1) * (1 + 0.1*np.sin(2*np.pi*df['HOUR']/24))

    # Meteorology Composite
    for col in ['WIND_SPEED', 'PBLH', 'T2M']:
        df[f'{col}_NORM'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df['METEOROLOGY_SCORE'] = 0.3*df['WIND_SPEED_NORM'] + 0.3*df['PBLH_NORM'] + 0.2*df['T2M_NORM']

    return df.dropna()

# --------------------------
# 2. Model Training and Visualization Class
# --------------------------
class PMCityVisualizer:
    def __init__(self, data):
        self.data = data
        self.current_city = 'Ramanathapuram'
        self.current_pm_type = 'PM2.5'

        # Custom CSS for widgets
        style = """
        <style>
            .widget-label { font-weight: bold !important; color: #1a5276 !important; }
            .widget-dropdown { background-color: #f8f9fa !important; border-radius: 5px !important; }
            .widget-slider { background-color: #f8f9fa !important; }
            .map-container { margin-top: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .info-container {
                margin-top: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                font-family: monospace;
            }
            .pm-display {
                padding: 8px;
                border-radius: 4px;
                margin: 5px 0;
                color: white;
                font-weight: bold;
            }
        </style>
        """
        display(widgets.HTML(style))

        # Create widgets
        self.city_picker = widgets.Dropdown(
            options=list(CITY_DATA.keys()),
            description='City:',
            style={'description_width': 'initial'},
            layout={'width': '250px'}
        )

        self.pm_type_picker = widgets.Dropdown(
            options=['PM2.5', 'PM10'],
            description='PM Type:',
            style={'description_width': 'initial'},
            layout={'width': '200px'}
        )

        self.date_picker = widgets.Dropdown(
            options=sorted(data['time'].dt.date.unique()),
            description='Date:',
            style={'description_width': 'initial'},
            layout={'width': '250px'}
        )

        self.time_picker = widgets.SelectionSlider(
            options=[(f"{h:02d}:00", h) for h in range(24)],
            value=12,
            description='Time:',
            style={'description_width': 'initial'},
            continuous_update=False,
            layout={'width': '350px'},
            readout=True
        )

        # Output areas
        self.map_output = widgets.Output(layout={'border': '1px solid #ddd', 'padding': '10px'})
        self.info_output = widgets.Output(layout={'border': '1px solid #ddd', 'padding': '10px'})

        # Set up observers
        self.city_picker.observe(self.update_city, names='value')
        self.pm_type_picker.observe(self.update_pm_type, names='value')
        self.date_picker.observe(self.update_display, names='value')
        self.time_picker.observe(self.update_display, names='value')

        # Initial display
        self.display_controls()
        self.update_display()

    def display_controls(self):
        header = widgets.HTML(
            "<h2 style='color: #1a5276; text-align: center;'>🌍 Air Quality Monitoring Dashboard</h2>"
            "<p style='text-align: center; color: #566573;'>Explore PM2.5 and PM10 concentrations across cities</p>"
        )

        controls = widgets.VBox([
            widgets.HBox([self.city_picker, self.pm_type_picker, self.date_picker, self.time_picker],
                        layout={'justify_content': 'space-between', 'flex-wrap': 'wrap'}),
            widgets.HTML("<hr style='margin: 10px 0;'>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<h3 style='color: #1a5276;'>Map View</h3>"),
                    self.map_output
                ], layout={'width': '60%'}),
                widgets.VBox([
                    widgets.HTML("<h3 style='color: #1a5276;'>Air Quality Info</h3>"),
                    self.info_output
                ], layout={'width': '38%'})
            ])
        ], layout={'padding': '10px'})

        display(widgets.VBox([header, controls]))

    def update_city(self, change):
        self.current_city = change.new
        city_data = self.data[self.data['city'] == self.current_city]
        self.date_picker.options = sorted(city_data['time'].dt.date.unique())
        self.update_display()

    def update_pm_type(self, change):
        self.current_pm_type = change.new
        self.update_display()

    def get_dynamic_colormap(self, pm_values):
        """Create dynamic colormap based on data distribution"""
        q95 = pm_values.quantile(0.95)
        return LinearColormap(
            ['green', 'yellow', 'orange', 'red', 'darkred'],
            vmin=0,
            vmax=q95,
            caption=f'{self.current_pm_type} Concentration (µg/m³)'
        )

    def update_display(self, change=None):
        city_data = self.data[self.data['city'] == self.current_city]
        selected_date = self.date_picker.value
        selected_hour = int(self.time_picker.value)
        selected_dt = datetime.combine(selected_date, datetime.strptime(f"{selected_hour:02d}:00", "%H:%M").time())

        # Find nearest timestamp
        nearest_idx = (city_data['time'] - selected_dt).abs().idxmin()
        row = city_data.iloc[nearest_idx]

        # Get dynamic colormap for current PM type
        pm_values = city_data[f'{self.current_pm_type}_PREDICTED']
        colormap = self.get_dynamic_colormap(pm_values)

        with self.map_output:
            clear_output(wait=True)
            m = folium.Map(
                location=CITY_DATA[self.current_city]['coords'],
                zoom_start=CITY_DATA[self.current_city]['zoom'],
                tiles='CartoDB positron',
                width='100%',
                height='400px'
            )

            # Add marker for city center
            folium.Marker(
                location=CITY_DATA[self.current_city]['coords'],
                popup=f"Center of {self.current_city}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

            # Add PM marker
            folium.CircleMarker(
                location=CITY_DATA[self.current_city]['coords'],
                radius=15,
                color=colormap(row[f'{self.current_pm_type}_PREDICTED']),
                fill=True,
                fill_opacity=0.8,
                popup=f"""
                <b>City:</b> {self.current_city}<br>
                <b>PM Type:</b> {self.current_pm_type}<br>
                <b>Time:</b> {row['time'].strftime('%Y-%m-%d %H:%M')}<br>
                <b>{self.current_pm_type}:</b> {row[f'{self.current_pm_type}_PREDICTED']:.1f} µg/m³<br>
                <b>Wind:</b> {row['WIND_SPEED']:.1f} m/s<br>
                <b>PBL Height:</b> {row['PBLH']:.0f} m<br>
                <b>AOD:</b> {row['TOTEXTTAU']:.2f}
                """
            ).add_to(m)

            # Add color scale
            colormap.add_to(m)

            display(m)

        with self.info_output:
            clear_output(wait=True)
            pm25_color = colormap(row['PM2.5_PREDICTED']) if self.current_pm_type == 'PM2.5' else '#f8f9fa'
            pm10_color = colormap(row['PM10_PREDICTED']) if self.current_pm_type == 'PM10' else '#f8f9fa'

            info_html = f"""
            <div style='font-family: Arial, sans-serif;'>
                <h3 style='color: #1a5276;'>{self.current_city} Air Quality</h3>
                <p><strong>Time:</strong> {row['time'].strftime('%Y-%m-%d %H:%M')}</p>

                <div style='display: flex; justify-content: space-between;'>
                    <div style='background-color: {pm25_color};' class='pm-display'>
                        <strong>PM2.5:</strong> {row['PM2.5_PREDICTED']:.1f} µg/m³
                    </div>
                    <div style='background-color: {pm10_color};' class='pm-display'>
                        <strong>PM10:</strong> {row['PM10_PREDICTED']:.1f} µg/m³
                    </div>
                </div>

                <p><strong>PM2.5/PM10 Ratio:</strong> {row['PM2.5_PREDICTED']/row['PM10_PREDICTED']:.2f}</p>
                <p><strong>Wind Speed:</strong> {row['WIND_SPEED']:.1f} m/s</p>
                <p><strong>Boundary Layer Height:</strong> {row['PBLH']:.0f} m</p>
                <p><strong>AOD:</strong> {row['TOTEXTTAU']:.2f}</p>
                <hr>
                <p style='font-size: 0.9em; color: #566573;'>
                    Showing: {self.current_pm_type} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </p>
            </div>
            """
            display(widgets.HTML(info_html))

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load all city data
    city_files = {
        'Ramanathapuram': '/content/RamanathpuramMerge_Jan2023.csv',
        'Delhi': '/content/DelhiMerge_Jan2023.csv',
        'Mumbai': '/content/MumbaiMerge_Jan2023.csv'
    }

    all_data = pd.concat([load_and_preprocess(f) for f in city_files.values()])

    # Features and targets
    base_features = [
        'PM2.5_DIURNAL_ADJUSTED',
        'PM2.5_24H_MEAN',
        'PM10_DIURNAL_ADJUSTED',
        'PM10_24H_MEAN',
        'AOD_PBLH_INTERACTION',
        'AOD_WIND_INTERACTION',
        'METEOROLOGY_SCORE',
        'HOUR',
        'IS_WEEKEND'
    ]

    # Train models for each city and PM type
    models = {}
    for city in CITY_DATA.keys():
        city_data = all_data[all_data['city'] == city]
        test_size = int(0.2 * len(city_data))

        for pm_type in ['PM2.5', 'PM10']:
            X_train, X_test = city_data[base_features][:-test_size], city_data[base_features][-test_size:]
            y_train, y_test = city_data[pm_type][:-test_size], city_data[pm_type][-test_size:]

            model = XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.02,
                reg_alpha=1.0,
                reg_lambda=1.0,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='reg:squarederror',
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
            models[(city, pm_type)] = model
            all_data.loc[all_data['city'] == city, f'{pm_type}_PREDICTED'] = model.predict(city_data[base_features])

    # Initialize visualizer
    visualizer = PMCityVisualizer(all_data)

    # Enhanced evaluation function
    def evaluate_models():
        for (city, pm_type), model in models.items():
            city_data = all_data[all_data['city'] == city]
            test_size = int(0.2 * len(city_data))
            X_test = city_data[base_features][-test_size:]
            y_test = city_data[pm_type][-test_size:]
            y_pred = model.predict(X_test)

            print(f"\n{city} - {pm_type} Model Performance:")
            print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} µg/m³")
            print(f"R²: {r2_score(y_test, y_pred):.2f}")

            # Feature Importance
            importance = pd.DataFrame({
                'Feature': base_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            plt.figure(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
            plt.title(f'{city} - {pm_type} Feature Importance')
            plt.show()

            # Time Series Plot
            plt.figure(figsize=(14, 6))
            plt.plot(city_data['time'], city_data[pm_type], label='Actual', alpha=0.7)
            plt.plot(city_data['time'], city_data[f'{pm_type}_PREDICTED'], label='Predicted', linestyle='--')
            plt.title(f'{pm_type} Concentration in {city}')
            plt.xlabel('Date')
            plt.ylabel(f'{pm_type} (µg/m³)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    evaluate_models()

    # Save models
    for (city, pm_type), model in models.items():
        joblib.dump(model, f'{city.lower()}_{pm_type.lower()}_model.pkl')
    print("All models saved successfully!")
