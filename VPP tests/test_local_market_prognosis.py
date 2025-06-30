import requests
import json
from datetime import datetime
import csv

def fetch_market_data(zip_code, token, output_file='berlin_market_data.csv'):
    """
    Fetch electricity market data for a given zip code in Berlin and save it to a CSV file.

    Parameters:
    - zip_code: str, the zip code (Postleitzahl) of the city.
    - token: str, the access token for the API.
    - output_file: str, the name of the output CSV file to save the data.
    """
    url = 'https://api.corrently.io/v2.0/marketdata'
    headers = {'Accept': 'application/json'}

    params = {
        'zip': zip_code,
        'token': token
    }

    try:
        # Sending the GET request to the API
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an error for bad responses

        # Parse the JSON response
        market_data = response.json()

        # Extracting the relevant data
        data_rows = []
        for item in market_data.get('data', []):
            # Convert timestamps to datetime
            start_time = datetime.utcfromtimestamp(item['start_timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            end_time = datetime.utcfromtimestamp(item['end_timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            row = {
                'start_time': start_time,
                'end_time': end_time,
                'marketprice': item['marketprice'],
                'localprice': item['localprice'],
                'unit': item['unit'],
                'localcell': item['localcell'],
            }
            data_rows.append(row)

        # Saving the data to a CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['start_time', 'end_time', 'marketprice', 'localprice', 'unit', 'localcell']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data_rows)

        print(f"Market data for zip code {zip_code} saved successfully in {output_file}.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # e.g. 404 Not Found
    except Exception as err:
        print(f"An error occurred: {err}")

# Beispielaufruf der Funktion
fetch_market_data('10117', '02ce55aeb9msh17a433fb00abfe7p1d4bbfjsn60cac2d359e4')
