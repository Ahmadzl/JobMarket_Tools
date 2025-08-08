"""
Script Overview:
This script scrapes educational program details from the Swedish Yrkeshögskolan website.
It loops through a range of program IDs, extracts key admission requirements (basic and specific),
organizer info, and program title, and saves the results into a CSV file called 'utbildningar_requirements.csv'.
Useful for analyzing admission requirements across multiple programs.
"""
import requests
from bs4 import BeautifulSoup
import csv

# Base URL and ID range for scraping
BASE = "https://www.yrkeshogskolan.se"
program_ids = list(range(8500, 10500))  # Wide range to cover many programs

rows = []

# Loop through program IDs and scrape details
for pid in program_ids:
    url = f"{BASE}/hitta-utbildning/sok/utbildning/?id={pid}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Extract program title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Okänd"

        # Extract basic requirements
        req = soup.find(string="Grundläggande behörighet")
        grund = req.find_next("ul").get_text(separator="; ") if req and req.find_next("ul") else "Okänd"

        # Extract specific requirements or other conditions
        spec_text = "–"
        for text_option in ["Särskilda förkunskaper", "Andra villkor"]:
            tag = soup.find(string=lambda t: t and text_option in t)
            if tag:
                ul = tag.find_next("ul")
                if ul:
                    spec_text = ul.get_text(separator="; ")
                    break

        # Extract organizer info
        org_tag = soup.find(string="Utbildningsanordnare")
        org = org_tag.find_next("dd").get_text(strip=True) if org_tag and org_tag.find_next("dd") else "Okänd"

        rows.append([title, "YH", grund, spec_text, "", f"{org} – {url}"])

        print(f"[+] Sparad: {title}")

    except Exception as e:
        print(f"[-] Fel vid ID {pid}: {e}")
        continue

# Export data to CSV
with open("utbildningar_requirements.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Utbildning", "Nivå", "Grundläggande behörighet", "Särskild behörighet", "Andra villkor", "Organisatör/Länk"])
    writer.writerows(rows)

print("\nData saved to utbildningar_requirements.csv")
