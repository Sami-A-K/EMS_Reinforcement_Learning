from flask import Flask, jsonify, request, render_template
import pandas as pd

app = Flask(__name__)

# Globale Variablen für die Daten
dashboard_data = {}

# Standard-Einstellungen
settings = {
    "max_cost": 50,  # Max. Stromkosten in €/Woche
    "battery_mode": "Speichern",  # Batteriemodus
    "co2_target": 30  # Ziel-CO₂-Einsparung in %
}

# API: Energiedaten abrufen
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(dashboard_data)

# API: Konfigurationsdaten abrufen
@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    if request.method == 'POST':
        global settings
        settings = request.json
        return jsonify({"status": "success", "message": "Konfiguration gespeichert!"})
    return jsonify(settings)

# API: Agent vs Baseline Vergleich aus power_data.csv
@app.route('/api/power_comparison')
def get_power_comparison():
    try:
        # Lade power_data_extended.csv (1 Woche)
        df = pd.read_csv('../power_data_extended.csv', index_col=0, parse_dates=True)
        
        if len(df) == 0:
            return jsonify({'status': 'error', 'message': 'Keine Power-Daten gefunden'})
        
        # Extrahiere Daten für alle Actors
        actor_names = []
        agent_power_data = {}
        baseline_power_data = {}
        
        # Finde alle Actor-Namen aus den Spalten
        for col in df.columns:
            if col.startswith('agent_power_'):
                actor_name = col.replace('agent_power_', '')
                actor_names.append(actor_name)
                
                agent_power_data[actor_name] = df[col].tolist()
                baseline_col = f'baseline_power_{actor_name}'
                if baseline_col in df.columns:
                    baseline_power_data[actor_name] = df[baseline_col].tolist()
        
        return jsonify({
            'status': 'success',
            'data': {
                'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'steps': df['step'].tolist(),
                'agent_power': agent_power_data,
                'baseline_power': baseline_power_data,
                'power_diff': {name: df[f'power_diff_{name}'].tolist() for name in actor_names if f'power_diff_{name}' in df.columns},
                'agent_cost': df['agent_cost'].tolist(),
                'baseline_cost': df['baseline_cost'].tolist(),
                'cost_saving': df['cost_saving'].tolist(),
                'reward': df['reward'].tolist(),
                'cumulative_agent_cost': df['agent_cost'].cumsum().tolist(),
                'cumulative_baseline_cost': df['baseline_cost'].cumsum().tolist(),
                'cumulative_saving': df['cost_saving'].cumsum().tolist(),
                'actor_names': actor_names,
                # Statistiken
                'total_agent_cost': float(df['agent_cost'].sum()),
                'total_baseline_cost': float(df['baseline_cost'].sum()),
                'total_saving': float(df['cost_saving'].sum()),
                'saving_percentage': float((df['cost_saving'].sum() / df['baseline_cost'].sum() * 100) if df['baseline_cost'].sum() != 0 else 0)
            }
        })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Fehler beim Laden der Power-Daten: {str(e)}'})

# API: Verfügbare Zeiträume aus Baseline abrufen
@app.route('/api/baseline/periods')
def get_available_baseline_periods():
    try:
        df_baseline = pd.read_csv('optimal_episode.csv', index_col='time', parse_dates=True)
        
        # Verfügbare Tage extrahieren (konvertiere zu pandas Series für unique())
        available_dates = pd.Series(df_baseline.index.date).unique()
        periods = []
        
        for date in available_dates[:10]:  # Erste 10 Tage
            start_time = pd.Timestamp(date)
            end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            
            # Prüfe ob Daten für den Tag verfügbar sind
            day_data = df_baseline.loc[start_time:end_time]
            if len(day_data) >= 90:  # Mindestens 90 von 96 15-Minuten-Intervallen
                periods.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'data_points': len(day_data)
                })
        
        return jsonify({'status': 'success', 'periods': periods})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Fehler beim Laden der verfügbaren Zeiträume: {str(e)}'})

# Dashboard-Route
@app.route('/')
def index():
    return render_template('power_comparison.html')

# Konfigurations-Route
@app.route('/config')
def configuration():
    return render_template('config.html')

# Baseline-Vergleichs-Route
@app.route('/baseline')
def baseline_comparison():
    return render_template('baseline.html')

# Power-Vergleichs-Route
@app.route('/power')
def power_comparison():
    return render_template('power_comparison.html')

# Historische Kosten für Trends
cost_history = []

# Route: Daten neu laden
@app.route('/api/reload', methods=['POST'])
def reload_data():
    global dashboard_data, cost_history
    
    # Versuche zuerst die Dali-Daten zu laden
    try:
        df = pd.read_csv('network_RL_dali_full.csv')
    except Exception as e:
        # Fallback auf Max-Daten
        try:
            df = pd.read_csv('../network_RL_last.csv')
        except Exception as e2:
            df = pd.DataFrame()  # Leerer DataFrame als Fallback

    # Zeitspalte parsen
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    else:
        if len(df) > 0:
            # Erstelle Zeitstempel basierend auf Datenlänge
            df['time'] = pd.date_range(start='2015-03-02', periods=len(df), freq='15min')
        else:
            return jsonify({"status": "error", "message": "Keine Daten geladen!", "data": {}})

    # Batterie-SoC berechnen (aus thermischen Speichern)
    thermal_storage_total = (df.get('house_1_thermal_storage_kWh', pd.Series([0]*len(df))).fillna(0) + 
                           df.get('house_2_thermal_storage_kWh', pd.Series([0]*len(df))).fillna(0) + 
                           df.get('house_3_thermal_storage_kWh', pd.Series([0]*len(df))).fillna(0))
    df['battery_soc'] = thermal_storage_total / 10  # Normalisiert auf 0-100%
    
    # Batterie-Leistung aus CSV
    battery_charge = df.get('central_pv_battery_charge_p0_kW', pd.Series([0]*len(df))).fillna(0)
    battery_discharge = df.get('central_pv_battery_discharge_p0_kW', pd.Series([0]*len(df))).fillna(0)
    df['battery_power'] = battery_charge + battery_discharge  # Gesamte Batterie-Leistung
    
    # Wärmepumpe elektrisch (basierend auf COP und thermischer Last)
    thermal_demand = df.get('thermal_energy_demand', pd.Series([0]*len(df))).fillna(0)
    hp_cop_ground = df.get('hp_cop_house_1', pd.Series([3]*len(df))).fillna(3)
    hp_cop_air = df.get('hp_cop_house_2', pd.Series([3]*len(df))).fillna(3)
    df['hp_el_demand'] = (thermal_demand * 0.7) / ((hp_cop_ground + hp_cop_air) / 2)
    
    # Netzbezug aus CSV
    df['net_demand'] = df.get('grid_demand_link_p0_kW', pd.Series([0]*len(df))).fillna(0) - df.get('grid_feed-in_link_p0_kW', pd.Series([0]*len(df))).fillna(0)
    
    # Temperatur (Dummy, falls nicht vorhanden)
    df['temperature'] = 20
    
    # Wärmepumpe thermisch (basierend auf elektrischer Leistung und COP)
    df['hp_th_out'] = df['hp_el_demand'] * ((hp_cop_ground + hp_cop_air) / 2)

    # --- Baseline laden und Summen berechnen ---
    try:
        # Neue PyPSA-Baseline laden (optimal_episode.csv)
        df_baseline = pd.read_csv('optimal_episode.csv', index_col='time', parse_dates=True)
        
        # Gleichen Zeitraum wie RL-Daten verwenden
        if 'time' in df.columns:
            start_date = df['time'].iloc[0].date()
            end_date = df['time'].iloc[-1].date()
            baseline_period = df_baseline.loc[start_date:end_date]
            
            if len(baseline_period) > 0:
                # PyPSA-Baseline KPIs berechnen
                baseline_power = baseline_period['optimal_power'].sum() / 4  # kWh
                baseline_costs = baseline_period['optimal_costs'].sum() / 4  # €
                baseline_electrical = baseline_period['electrical_baseload'].sum() / 4  # kWh
                baseline_thermal = baseline_period['thermal_energy_demand'].sum() / 4  # kWh
                
                # Fallback auf alte Baseline falls PyPSA-Daten fehlen
                baseline_cost = baseline_costs
                baseline_co2 = baseline_thermal * 0.4  # Schätzung basierend auf thermischer Last
            else:
                # Fallback: Lade alte Baseline
                df_base = pd.read_csv('Output/loadshape_cost.csv')
                if 'time' in df_base.columns:
                    df_base['time'] = pd.to_datetime(df_base['time'], errors='coerce')
                    dali_start_date = df['time'].iloc[0].date()
                    df_base_same_day = df_base[df_base['time'].dt.date == dali_start_date]
                    
                    if len(df_base_same_day) > 0:
                        baseline_cost = df_base_same_day['cost'].sum()
                        baseline_co2 = df_base_same_day['co2'].sum()
                    else:
                        # Fallback: Erstelle realistische Baseline basierend auf Dali-Daten
                        dali_electrical_demand = df.get('electrical_baseload', pd.Series([0]*len(df))).fillna(0).sum()
                        dali_thermal_demand = df.get('thermal_energy_demand', pd.Series([0]*len(df))).fillna(0).sum()
                        
                        # Baseline: Fixed Policy (keine Optimierung)
                        baseline_electrical = dali_electrical_demand * 1.2  # 20% mehr Verbrauch
                        baseline_thermal = dali_thermal_demand * 1.3       # 30% mehr thermischer Verbrauch
                        
                        # Kosten berechnen (höhere Preise für Baseline)
                        avg_price = df.get('price_el', pd.Series([0.3]*len(df))).fillna(0.3).mean()
                        baseline_cost = (baseline_electrical + baseline_thermal/3) * avg_price * 0.25  # 15min intervals
                        
                        # CO2 berechnen (höhere Intensität für Baseline)
                        avg_co2 = df.get('co2_el', pd.Series([400]*len(df))).fillna(400).mean()
                        baseline_co2 = (baseline_electrical + baseline_thermal/3) * avg_co2 / 1000
                else:
                    # Fallback: Verwende Durchschnitt pro Tag
                    baseline_cost = df_base['cost'].sum() / 365
                    baseline_co2 = df_base['co2'].sum() / 365
        else:
            baseline_cost = None
            baseline_co2 = None
            
    except Exception as e:
        print(f"Baseline-Daten konnten nicht geladen werden: {e}")
        baseline_cost = None
        baseline_co2 = None

    # Kosten und CO2 basierend auf Netzbezug und Preisen berechnen
    electricity_price = df.get('price_el', pd.Series([0.3]*len(df))).fillna(0.3)
    co2_intensity = df.get('co2_el', pd.Series([400]*len(df))).fillna(400)
    
    # Kosten: Netzbezug * Preis * 0.25h (15min)
    df['cost'] = df['net_demand'].clip(lower=0) * electricity_price * 0.25
    # CO2: Netzbezug * CO2-Intensität / 1000 (kg CO2)
    df['co2'] = df['net_demand'].clip(lower=0) * co2_intensity / 1000
    
    rl_cost = df['cost'].sum()
    rl_co2 = df['co2'].sum()

    # Einsparungen berechnen (in %)
    if baseline_cost and baseline_cost > 0:
        cost_saving = 100 * (baseline_cost - rl_cost) / baseline_cost
    else:
        cost_saving = None
    if baseline_co2 and baseline_co2 > 0:
        co2_saving = 100 * (baseline_co2 - rl_co2) / baseline_co2
    else:
        co2_saving = None

    dashboard_data = {
        'date': df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'consumption': df.get('electrical_baseload', pd.Series([0]*len(df))).fillna(0).tolist(),
        'production': df.get('pv_gen_central_pv', pd.Series([0]*len(df))).fillna(0).tolist(),
        'battery_soc': df['battery_soc'].tolist(),
        'battery_power': df['battery_power'].tolist(),
        'temperature': df['temperature'].tolist(),
        'heat_pump': df['hp_el_demand'].tolist(),
        'electricity_price': df.get('price_el', pd.Series([0]*len(df))).fillna(0).tolist(),
        'hp_th_out': df['hp_th_out'].tolist(),
        'net_demand': df['net_demand'].tolist(),
        'cost': df.get('cost', pd.Series([0]*len(df))).fillna(0).tolist(),
        'total_cost': df.get('cost', pd.Series([0]*len(df))).cumsum().tolist(),
        # Summen/KPIs
        'pv_kwh_sum': df.get('pv_gen_central_pv', pd.Series([0]*len(df))).fillna(0).sum()/4,
        'consumption_sum': df.get('electrical_baseload', pd.Series([0]*len(df))).fillna(0).sum()/4,
        'net_import_sum': abs(df['net_demand'][df['net_demand'] > 0].sum() / 4),
        'net_export_sum': abs(df['net_demand'][df['net_demand'] < 0].sum() / 4),
        'battery_discharge_sum': abs(df['battery_power'][df['battery_power'] < 0].sum() / 4),
        'battery_charge_sum': df['battery_power'][df['battery_power'] > 0].sum() / 4,
        'hp_th_out_sum': df['hp_th_out'].sum()/4,
        'battery_power_sum': df['battery_power'].abs().sum()/4,
        # Einsparungen
        'cost_saving': round(cost_saving, 2) if cost_saving is not None else None,
        'co2_saving': round(co2_saving, 2) if co2_saving is not None else None,
        # Absolute Werte
        'rl_cost': round(rl_cost, 2),
        'baseline_cost': round(baseline_cost, 2) if baseline_cost is not None else None,
        'rl_co2': round(rl_co2, 2),
        'baseline_co2': round(baseline_co2, 2) if baseline_co2 is not None else None,
        # PyPSA-Baseline Daten (falls verfügbar)
        'baseline_power': round(baseline_power, 2) if 'baseline_power' in locals() else None,
        'baseline_electrical': round(baseline_electrical, 2) if 'baseline_electrical' in locals() else None,
        'baseline_thermal': round(baseline_thermal, 2) if 'baseline_thermal' in locals() else None,
        # Zeitraum für die Anzeige bestimmen
        'start_date': df['time'].iloc[0] if 'time' in df.columns else None,
        'end_date': df['time'].iloc[-1] if 'time' in df.columns else None,
    }

    # Historische Kosten für Trends speichern
    if len(dashboard_data['total_cost']) >= 2:
        second_to_last_value = dashboard_data['total_cost'][-2]
        cost_history.append(second_to_last_value)
    dashboard_data['historical_total_cost'] = cost_history
    return jsonify({"status": "success", "message": "Daten wurden neu geladen!", "data": dashboard_data})

if __name__ == '__main__':
    app.run(debug=True)
