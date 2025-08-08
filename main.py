"""
Script Overview:
This Streamlit application, 'Swedish JobbStat', helps users explore job listings,
analyze unemployment statistics, match their experience or CV with job descriptions,
and receive training suggestions. It integrates job market data (JobTech),
unemployment data (SCB), and education requirements (Yrkeshögskolan).

Main Features:
- AI-powered job matching with CV summarization and semantic comparison
- Interactive job statistics by location and title
- Unemployment trend visualization (2001–2024)
- Personalized training tips based on user input
- Multi-language support (Swedish & English)
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load SpaCy and BART models for Swedish & English text summarization
nlp = spacy.load("en_core_web_sm")

# Load job data
# Load job listings, unemployment statistics, and extract text from PDFs
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Load unemployment data
def load_unemployment_data():
    return pd.read_csv("unemployment_by_gender.csv")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# AI Summarization and Embedding Caching
# Load transformer-based summarizers and embedding model with caching
# BART - Två språk (SV & EN)
# @st.cache_resource
@st.cache_resource(show_spinner=False)
def load_summarizers():
    summarizer_sv = pipeline("summarization", model="Gabriel/bart-base-cnn-swe")
    summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer_sv, summarizer_en

#   SentenceTransformer  
# @st.cache_resource
@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# @st.cache_data
@st.cache_data(show_spinner=False)
def get_job_embeddings(job_descriptions):
    model = load_sentence_model()
    return model.encode(job_descriptions, convert_to_tensor=True)

# Page: Job Matching
# Match user's experience or CV to job listings using semantic similarity
def job_matching_page(df):
    st.title("Job Match AI - Smart CV Analyzer")

    # summarizer = load_summarizer()
    summarizer_sv, summarizer_en = load_summarizers()
    
    sentence_model = load_sentence_model()

    job_titles = ["Alla"] + sorted(df['job_title'].dropna().unique().tolist())
    cities = ["Alla"] + sorted(df['location'].dropna().unique().tolist())

    selected_job = st.selectbox("Välj en jobbtitel (valfritt):", job_titles)
    selected_city = st.multiselect("Välj städer för att söka jobb:", options=cities, default=["Alla"])

    language = st.selectbox("Välj språk / Select language:", ["Svenska", "Engelska"])
    st.markdown("Skriv din erfarenhet eller ladda upp ett CV (PDF):")
    experience_text = st.text_area("Skriv din erfarenhet här:")

    uploaded_cv = st.file_uploader("Ladda upp ditt CV", type=["pdf"])

    # Knapp för att starta sökningen
    start_search = st.button("Starta sökning")

    if start_search:
        full_text = experience_text.strip()

        if uploaded_cv:
            st.info("Bearbetar CV...")
            cv_text = extract_text_from_pdf(uploaded_cv)
            full_text += "\n" + cv_text  # Combine both inputs

            with st.expander("Visa CV-text"):
                st.write(cv_text[:1500] + "..." if len(cv_text) > 1500 else cv_text)

        if full_text.strip():
            # if len(full_text) > 1000:
            #     st.info("Sammanfattar din text...")
            #     full_text = summarizer(full_text[:1024])[0]['summary_text']

            if len(full_text) > 1000:
                st.info("Sammanfattar din text...")

                if language == "Svenska":
                    full_text = summarizer_sv(full_text[:1024], max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                else:
                    full_text = summarizer_en(full_text[:1024], max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        
            doc = nlp(full_text)

            # Filtrera jobb enligt valda filter
            filtered_df = df.copy()

            if selected_job != "Alla":
                filtered_df = filtered_df[filtered_df['job_title'] == selected_job]

            if "Alla" not in selected_city:
                filtered_df = filtered_df[filtered_df['location'].isin(selected_city)]

            if filtered_df.empty:
                st.warning("Inga jobb hittades med de valda filtren.")
                return

            st.info("Matchar din erfarenhet med jobbannonser...")
            job_descriptions = filtered_df['description'].fillna("").tolist()
            # job_embeddings = get_job_embeddings(job_descriptions)
            with st.spinner("Bearbetar jobb embeddings..."):
                job_embeddings = get_job_embeddings(job_descriptions)
            user_embedding = sentence_model.encode(full_text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(user_embedding, job_embeddings)[0].cpu().numpy()

            filtered_df = filtered_df.copy()
            filtered_df['similarity'] = cosine_scores
            filtered_df = filtered_df.sort_values(by="similarity", ascending=False)

            matched_jobs = filtered_df[filtered_df['similarity'] > 0.4].head(10)

            if matched_jobs.empty:
                st.warning("Inga starka matchningar hittades. Här är några förslag:")
                matched_jobs = filtered_df.head(5)

            st.success(f"Visar {len(matched_jobs)} jobb med högst matchning.")

            for _, row in matched_jobs.iterrows():
                st.markdown(f"---\n### {row['job_title']} - {row['employer']}")
                st.markdown(f"**Plats:** {row['location']}")
                st.markdown(f"**Matchningspoäng:** {row['similarity'] * 100:.2f}%")
                if pd.notna(row.get("source_url")) and row["source_url"].strip():
                    st.markdown(f"[Ansök här]({row['source_url']})", unsafe_allow_html=True)
                else:
                    st.markdown("*Ingen ansökningslänk tillgänglig*")
                with st.expander("Visa jobbannons"):
                    st.write(row["description"] if pd.notna(row["description"]) else "Ingen beskrivning tillgänglig.")
        else:
            st.warning("Fyll i din erfarenhet eller ladda upp ett CV för att hitta matchande jobb.")

# Page: Job Statistics
# Visualize job availability by cities and job titles using bar charts
def statistics_page(df):
    st.title("Job Data Statistics ")

    st.subheader("Antal jobb per Städer")

    all_cities = sorted(df['location'].dropna().unique().tolist())
    selected_cities = st.multiselect(
        "Välj städer:",
        options=all_cities,
        default=[]
    )

    if selected_cities:
        filtered_df = df[df['location'].isin(selected_cities)]
    else:
        filtered_df = df.copy()

    location_counts = filtered_df['location'].value_counts().reset_index()
    location_counts.columns = ['Location', 'Antal jobb']

    fig = px.bar(location_counts, y='Location', x='Antal jobb',
                 title="Antal jobb per Städer",
                 labels={'Antal jobb': 'Antal jobb', 'Location': 'Location'},
                 orientation='h',
                 height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Antal jobb per jobbtitel")

    job_title_counts = df['job_title'].value_counts().reset_index()
    job_title_counts.columns = ['Jobbtitel', 'Antal jobb']

    fig2 = px.bar(job_title_counts.head(20), y='Jobbtitel', x='Antal jobb',
                  title="Top 20 Job Titles by Antal jobb",
                  labels={'Antal jobb': 'Antal jobb', 'Jobbtitel': 'Jobbtitel'},
                  orientation='h',
                  height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Topp 10 städer efter antal jobberbjudanden")

    city_job_counts = df['location'].value_counts().reset_index()
    city_job_counts.columns = ['Stad', 'Antal jobb']
    city_job_counts = city_job_counts.head(10).reset_index(drop=True)

    city_job_counts['Rank'] = city_job_counts.index + 1
    city_job_counts = city_job_counts[['Rank', 'Stad', 'Antal jobb']]

    st.dataframe(city_job_counts)

    fig_city = px.bar(
        city_job_counts,
        y='Stad',
        x='Antal jobb',
        color='Antal jobb',
        orientation='h',
        height=450,
        title="Antal jobb i topp 10 städer"
    )
    fig_city.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig_city, use_container_width=True)

# Page: Unemployment Dashboard
# Display unemployment trends by gender (2001–2024) with filtering and plots
def unemployment_dashboard_page():
    st.title("Arbetslöshets Statistik")
    st.write("Denna instrumentpanel visar årliga arbetslöshetsdata från SCB.")
    st.markdown("Dataperiod: Arbetslöshetsdata täcker åren 2001 till 2024.")

    data = load_unemployment_data()
    years = sorted(data['year'].unique())
    start_year, end_year = st.select_slider("Välj år från - till:", options=years, value=(years[0], years[-1]))
    genders = ["Male", "Female"]
    selected_genders = st.multiselect("Välj kön:", options=genders, default=genders)

    filtered_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]
    if 'gender' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['gender'].isin(selected_genders)]

    st.subheader("Filtrerad tabell över arbetslöshetsdata")
    data_display = filtered_data.copy()
    data_display['year'] = data_display['year'].astype(str)
    st.dataframe(data_display)

    mean_rate = filtered_data['unemployment_rate'].mean()
    max_rate = filtered_data['unemployment_rate'].max()
    min_rate = filtered_data['unemployment_rate'].min()

    st.markdown(f"- **Medel arbetslöshetsgrad:** {mean_rate:.2f}%")
    st.markdown(f"- **Högsta arbetslöshetsgrad:** {max_rate:.2f}%")
    st.markdown(f"- **Lägsta arbetslöshetsgrad:** {min_rate:.2f}%")

    st.subheader("Arbetslöshetsgrad över tid efter kön")
    fig, ax = plt.subplots(figsize=(12, 6))

    if 'gender' in filtered_data.columns:
        for gender in selected_genders:
            gender_data = filtered_data[filtered_data['gender'] == gender]
            line, = ax.plot(gender_data['year'], gender_data['unemployment_rate'],
                            marker='o', linestyle='-', label=gender)
            mean_gender = gender_data['unemployment_rate'].mean()
            ax.axhline(mean_gender, linestyle='--', linewidth=1,
                       color=line.get_color(), label=f'{gender} Medel: {mean_gender:.2f}%')
        ax.legend()
    else:
        ax.plot(filtered_data['year'], filtered_data['unemployment_rate'], marker='o', color='blue')

    ax.set_xlabel("År")
    ax.set_ylabel("Arbetslöshetsgrad (%)")
    ax.set_title(f"Arbetslöshetsutveckling ({start_year}-{end_year})")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Page: Tips and Training
# Suggest relevant training programs based on user's input or uploaded CV
def tips_and_training_page(df):
    st.title("Tips och Träning")
    st.markdown("""
        Den här sidan ger dig personliga tips och rekommendationer om träning 
        baserat på din erfarenhet eller CV.
    """)
    
    language = st.selectbox("Välj språk / Select language:", ["Svenska", "Engelska"])
    experience_text = st.text_area("Beskriv din erfarenhet eller kompetens:")
    uploaded_cv = st.file_uploader("Ladda upp ditt CV (PDF)", type=["pdf"])
    
    summarizer_sv, summarizer_en = load_summarizers()  # تحميل النموذجين

    if uploaded_cv:
        experience_text += "\n" + extract_text_from_pdf(uploaded_cv)

    if experience_text.strip():
        if st.button("Starta analys"):
            if len(experience_text.split()) > 250:
                try:
                    if language == "Svenska":
                        summary = summarizer_sv(experience_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                    else:
                        summary = summarizer_en(experience_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                    st.markdown("### Sammanfattning av din text:")
                    st.success(summary)
                    experience_text = summary
                except Exception as e:
                    st.warning(f"Sammanfattning misslyckades: {e}")

            # Extrahera nyckelord
            doc = nlp(experience_text)
            keywords = set(token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "VERB") and not token.is_stop)

            # Ladda utbildningar
            utbildningar_df = pd.read_csv("utbildningar_requirements.csv")

            # Beräkna semantisk likhet
            st.markdown("### Passande utbildningar baserat på din beskrivning:")
            user_doc = nlp(experience_text)

            similarities = []
            for _, row in utbildningar_df.iterrows():
                req_text = str(row["Grundläggande behörighet"]) + " " + str(row["Särskild behörighet"])
                req_doc = nlp(req_text)
                similarity = user_doc.similarity(req_doc)
                if similarity > 0.6:  # Tröskel kan justeras
                    similarities.append((row["Utbildning"], similarity, row["Organisatör/Länk"]))

            if similarities:
                similarities.sort(key=lambda x: x[1], reverse=True)
                for utbildning, score, link in similarities[:5]:
                    st.markdown(f"**Matchning:** {score * 100:.1f}%")
                    st.markdown(f"**Utbildning:** {utbildning}")
                    st.markdown(f"[Mer info här]({link})")
                    st.markdown("---")
            else:
                st.info("Inga passande utbildningar hittades.")
    else:
        st.warning("Skriv din erfarenhet eller ladda upp ditt CV.")

# Page: About
# Project description, technologies
def about_page():
    st.title("Swedish JobbStat")
    st.markdown("""
    ## Beskrivning
    Job Match AI är ett intelligent jobbrekommendationssystem som använder NLP och statistik från svenska källor.
    ## Funktioner
    - Arbetsmarknadsstatistik
    - Jobbfiltrering
    - CV-matchning med träningstips
    ## Bibliotek
    Python, Streamlit, Spacy, Pandas, Plotly
    ## Utvecklad av
    Ahmad Zalkat
    """)

# Main Function - App Entry Point
# Sidebar navigation and page routing based on user selection
def main():
    csv_file_path = "jobtech_jobs.csv"
    try:
        df = load_data(csv_file_path)
    except Exception as e:
        st.error(f"Fel vid inläsning av jobbdata: {e}")
        return

    if df.empty:
        st.warning("Inga jobbdata tillgängliga.")
        return

    page = st.sidebar.selectbox("Välj sida", [
        "Jobbmatchning", "Tips och Träning",
        "Jobb Data Statistik", "Arbetslöshets Statistik", "Om"
    ])

    if page == "Jobbmatchning":
        job_matching_page(df)
    elif page == "Tips och Träning":
        tips_and_training_page(df)
    elif page == "Jobb Data Statistik":
        statistics_page(df)
    elif page == "Arbetslöshets Statistik":
        unemployment_dashboard_page()
    elif page == "Om":
        about_page()

# Start the Streamlit App
if __name__ == "__main__":
    main()
