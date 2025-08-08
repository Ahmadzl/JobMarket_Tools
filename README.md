# Swedish AI-Powered Labor Market Tools
# Swedish Job Market & Education Analysis

## Project Overview
This project provides tools and scripts for analyzing the Swedish labor market using real-time data. It includes web scraping, data processing, AI-based job matching and a web-based interface for insights and exploration.

## Contents
- `jobtech_scraper.py`  Fetches job listings from the [JobTech API](https://jobtechdev.se/) and saves them to a CSV file.
- `unemployment_data.py`  Downloads and processes unemployment data by gender and year from the [SCB API](https://scb.se/) and calculates unemployment rates.
- `education_scraper.py`  Scrapes requirements and organizers from the Swedish Yrkeshögskolan website for various program IDs.
- `main.py`  Interactive Streamlit web app with job matching, labor market stats, unemployment dashboard and training suggestions.

## Key Features
- **Job Listings Scraper**  
  Collects job titles, locations, employers, and descriptions via the JobTech API and saves them locally.

- **Unemployment Data Processor**  
  Fetches unemployment and labor force data (2001–2024) by gender and year, calculates unemployment rates and exports to CSV.

- **YH Education Requirements Scraper**  
  Extracts admission requirements and organizer information for vocational education programs from the official website.

- **AI-Powered Job Match App**
  - Matches CVs with job listings using SentenceTransformer + cosine similarity.
  - Text summarization with multilingual BART models (Swedish & English):
    - `facebook/bart-large-cnn`: English summarization model fine-tuned on the CNN/Daily Mail dataset. It generates clear, concise summaries from long news-style content.
    - `Gabriel/bart-base-cnn-swe`: Swedish summarization model fine-tuned on Swedish news articles. It creates accurate and readable summaries for Swedish-language texts.
  - Interactive visualizations by city and job title. 
  - Unemployment trends visualization.
  - Personalized training suggestions.

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

streamlit
pandas
requests
plotly
matplotlib
beautifulsoup4
PyMuPDF
spacy
transformers
sentence-transformers


### 2. Run scripts 
python jobtech_scraper.py
python unemployment_data.py
python education_scraper.py

### 3. Launch the web app
streamlit run main.py
python -m streamlit run main.py

## Output Files
- `jobtech_jobs.csv` Jobs collected from JobTech API.
- `unemployment_by_gender.csv` Unemployment rates by gender and year.
- `utbildningar_requirements.csv` YH program requirements.

## Notes
- All data is sourced from **official Swedish public APIs or websites**.
- The app supports both **Swedish and English** for summarization and interface.
- A stable internet connection is required to fetch live data during execution.

## Developed by
**Ahmad Zalkat**  
Project Lead & NLP/Data Science Enthusiast focused on labor market insights using AI.


