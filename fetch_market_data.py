import pandas as pd
import requests
from datetime import datetime

start = "2023-01-01T00:000+01:00"
end = "2024-01-01T00:000+01:00"

#2022-11-25T13:54:00.000+00:00



# **API 1: Spotmarktpreise abrufen**
url_spot = "https://api.energy-charts.info/price"
params_spot = {"bzn": "DE-LU", "start": start, "end": end}

response_spot = requests.get(url_spot, params_spot)

if response_spot.status_code == 200:
    data_spot = response_spot.json()
    timestamps_spot = data_spot["unix_seconds"]
    prices_eur_mwh = data_spot["price"]

    # Preise von €/MWh in ct/kWh umwandeln
    prices_ct_kwh = [price * 0.1 for price in prices_eur_mwh]  # 1 €/MWh = 0.1 ct/kWh

    # Timestamps umwandeln
    timestamps_spot = pd.to_datetime([datetime.fromtimestamp(ts) for ts in timestamps_spot])

    # DataFrame erstellen
    df_market_data = pd.DataFrame({"price_el": prices_ct_kwh}, index=timestamps_spot)

    # Doppelte Zeitstempel entfernen
    df_market_data = df_market_data[~df_market_data.index.duplicated(keep="first")]

    # Resample auf 15-Minuten-Intervalle
    df_market_data = df_market_data.resample("15min").ffill()

    # Letzten Eintrag entfernen (nur für Resampling genutzt)
    df_market_data = df_market_data.drop(df_market_data.index[-1])
    df_market_data["price_el"] = df_market_data["price_el"].round(2)

else:
    print(f"Fehler beim Abruf der Spotpreise: {response_spot.status_code} - {response_spot.text}")

print(df_market_data)

""" API-Call CO2-Emissionen """
API_KEY = "qNMn08b16qgDhLZmGxaNGGAWEdsEqTUdceYKWEWwCgr69m3v"  # Hier API-Schlüssel einfügen
url_co2 = f"https://eco2grid.com/green-grid-compass/co2intensity/co2/summary/hourly?limit=9000&zone_code=DE_LU&start=2023-01-01T00%3A00%3A00.000%2B00%3A00&end=2024-01-01T00%3A00%3A00.000%2B00%3A00&emission_scope=operational&apikey={API_KEY}"

response_co2 = requests.get(url_co2, headers={"Accept": "application/json"})

if response_co2.status_code == 200:
    data_co2 = response_co2.json()
    
    if "datetimes" in data_co2 and "zone_data" in data_co2:
        timestamps_co2 = data_co2["datetimes"]  # Liste der Zeitstempel
        co2_values = [entry["consumption_co2_intensity"] for entry in data_co2["zone_data"]]  # CO₂-Werte

        # DataFrame erstellen
        df_co2 = pd.DataFrame({"co2_el": co2_values}, index=pd.to_datetime(timestamps_co2))
        # Stündliche Werte auf 15-Minuten-Intervalle interpolieren
        df_co2 = df_co2.resample("15min").ffill()

        # CO₂-Daten in den Haupt-DataFrame integrieren
        df_market_data["co2_el"] = df_co2["co2_el"].round(2)
    
    else:
        print("Fehler: Die API enthält keine 'datetimes' oder 'zone_data'-Daten.")
        df_market_data["co2_el"] = 400  # Falls API fehlschlägt, Standardwert setzen

else:
    print(f"Fehler beim Abruf der CO₂-Daten: {response_co2.status_code} - {response_co2.text}")
    df_market_data["co2_el"] = 400  # Falls API fehlschlägt, Standardwert setzen

# Index als "time" umbenennen
df_market_data = df_market_data.reset_index().rename(columns={"index": "time"})
df_market_data['time'] = df_market_data['time'].apply(lambda x: x.replace(year=2015))
# CSV speichern
df_market_data.to_csv("market_data.csv", index=False)
print(df_market_data)
print("CSV-Datei erfolgreich erstellt:")

