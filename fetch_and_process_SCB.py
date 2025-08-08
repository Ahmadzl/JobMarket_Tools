"""
Script Overview:
This script retrieves labor market data from Statistics Sweden (SCB) for the years 2001–2024.
It focuses on unemployment and labor force statistics by gender (male and female),
calculates unemployment rates, and saves the results into a CSV file named 'unemployment_by_gender.csv'.
Useful for gender-based labor market analysis over time.
"""
import requests
import csv

# Fetch data from SCB API
def fetch_unemployment_data(years):
    """
    Fetch unemployment and labor force data from the SCB API for the given years.
    """
    url = "https://api.scb.se/OV0104/v1/doris/en/ssd/AM/AM0401/AM0401A/AKURLBefAr"

    payload = {
        "query": [
            {
                "code": "Arbetskraftstillh",
                "selection": {
                    "filter": "item",
                    "values": ["ALÖS", "TOTB"]
                }
            },
            {
                "code": "TypData",
                "selection": {
                    "filter": "item",
                    "values": ["O_DATA"]
                }
            },
            {
                "code": "Kon",  # Gender: 1 = Male, 2 = Female
                "selection": {
                    "filter": "item",
                    "values": ["1", "2"]
                }
            },
            {
                "code": "Alder",
                "selection": {
                    "filter": "item",
                    "values": ["tot15-74"]
                }
            },
            {
                "code": "ContentsCode",
                "selection": {
                    "filter": "item",
                    "values": ["000007V1"]
                }
            },
            {
                "code": "Tid",
                "selection": {
                    "filter": "item",
                    "values": years
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        print("Error fetching data:", e)
        return None
# Process raw SCB data
# Calculates unemployment rate (unemployed / labor force) by year and gender
def process_data(raw_data):
    """
    Process the SCB data to calculate unemployment rates for males and females separately.
    """
    temp = {}

    for item in raw_data.get('data', []):
        keys = item['key']
        category = keys[0]         
        gender_code = keys[2]      # 1 = Male, 2 = Female
        year = keys[-1]            # Year

        gender = "Male" if gender_code == "1" else "Female"

        try:
            value = float(item['values'][0])
        except:
            value = None

        key = (year, gender)
        if key not in temp:
            temp[key] = {
                'year': year,
                'gender': gender,
                'unemployed': None,
                'labor_force': None
            }

        if category == "ALÖS":
            temp[key]['unemployed'] = value
        elif category == "TOTB":
            temp[key]['labor_force'] = value

    # Calculate unemployment rates
    processed = []
    for (year, gender), data in temp.items():
        u = data['unemployed']
        l = data['labor_force']
        if u is not None and l is not None and l > 0:
            data['unemployment_rate'] = round((u / l) * 100, 2)
        else:
            data['unemployment_rate'] = None
        processed.append(data)

    return processed
# Save data to CSV file
def save_to_csv(data, filename):
    """
    Save the processed unemployment data to a CSV file.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['year', 'gender', 'unemployed', 'labor_force', 'unemployment_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Controls the data fetching, processing, and saving steps
if __name__ == "__main__":
    years = [str(y) for y in range(2001, 2025)]
    raw_data = fetch_unemployment_data(years)

    if raw_data:
        processed_data = process_data(raw_data)
        save_to_csv(processed_data, 'unemployment_by_gender.csv')
        print("Data saved to 'unemployment_by_gender.csv'")
    else:
        print("Failed to fetch data.")
