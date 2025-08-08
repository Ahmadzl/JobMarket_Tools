"""
Script Overview:
This script fetches job listings from the JobTech Development API in Sweden.
It retrieves jobs in batches, extracts key details (title, employer, location, etc.),
and saves the collected data into a CSV file named 'jobtech_jobs.csv'.
The script can be run once or repeatedly at set intervals (loop section provided but commented out).
Useful for labor market analysis or job monitoring applications.
"""
import requests
import pandas as pd
import time
from datetime import datetime

# JobTech API base URL
base_url = "https://jobsearch.api.jobtechdev.se/search"

# HTTP headers with user agent
headers = {
    "User-Agent": "JobMarketAnalysisBot/1.0"
}
# Fetch and save job data
def fetch_and_save_jobs():
    all_jobs = []
    limit = 100
    offset = 0

    print(f"\nStarting data fetch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")

    while True: # Build API request
        params = {
            "q": "*",       
            "limit": limit,
            "offset": offset,
            "lang": "sv"
        }

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f" Request failed at offset {offset} with status code: {response.status_code}")
            break

        data = response.json()
        hits = data.get("hits", [])

        if not hits: # Stop if no more jobs
            print("No more jobs to fetch.")
            break
        
        # Extract relevant job info
        for job in hits:
            all_jobs.append({
                "job_title": job.get("headline"),
                "occupation_group": job.get("occupation_group", {}).get("label"),
                "location": job.get("workplace_address", {}).get("municipality"),
                "posting_date": job.get("publication_date"),
                "employer": job.get("employer", {}).get("name"),
                "description": job.get("description", {}).get("text"),
                "source_url": job.get("webpage_url")
            })

        print(f"Fetched {len(hits)} jobs (offset {offset})")
        offset += limit
        time.sleep(1)  # Politeness delay

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"jobtech_jobs_{timestamp}.csv"
    filename = f"jobtech_jobs.csv"
    df = pd.DataFrame(all_jobs)
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(df)} jobs to '{filename}'.")
    
fetch_and_save_jobs()

# ----------------------------
# Main loop for automatic updates
# Time interval between each full fetch (in seconds)
# interval_seconds = 3600  # = 1 hour 

# while True:
#     fetch_and_save_jobs()
#     print(f"\nWaiting for {interval_seconds / 60} minutes before next fetch.\n")
#     time.sleep(interval_seconds)
