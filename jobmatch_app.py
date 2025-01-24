import streamlit as st
from playwright.sync_api import sync_playwright
import re
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.cli import download

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Resume Parsing
def parse_resume(file_path):
    """Parse the resume and dynamically extract skills and experience."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Clean and preprocess resume text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()

    # Use NLP to extract skills dynamically
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:
            skills.add(ent.text.lower())

    # Extract experience
    experience_match = re.search(r'(\d+)\s+years of experience', text, re.IGNORECASE)
    experience = int(experience_match.group(1)) if experience_match else 0

    return {'skills': list(skills), 'experience': experience, 'text': text}


# Job Scraping
def scrape_jobs(position, location, preference):
    """Scrape jobs from LinkedIn using Playwright."""

    def extract_experience_level(job_description):
        """Extract experience level from the job description."""
        try:
            match = re.search(
                r'\b(\d+\s*-\s*\d+\s*years?|\d+\s*years?|entry-level|mid-level|senior|internship)\b',
                job_description.lower(),
            )
            return match.group(0) if match else "Not specified"
        except Exception as e:
            print(f"Error extracting experience level: {e}")
            return "Not specified"

    job_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Build LinkedIn search URL
        url = f"https://www.linkedin.com/jobs/search?keywords={position}&location={location}&f_WT={preference}"
        print(f"Scraping jobs from URL: {url}")
        page.goto(url)
        page.wait_for_selector(".base-card")

        # Extract job listings
        jobs = page.query_selector_all(".base-card")
        for job in jobs:
            try:
                title = job.query_selector(".base-search-card__title").inner_text().strip() or "N/A"
                company = job.query_selector(".base-search-card__subtitle").inner_text().strip() or "N/A"
                location = job.query_selector(".job-search-card__location").inner_text().strip() or "N/A"
                link = job.query_selector("a").get_attribute("href") or "N/A"
                posted_day = job.query_selector(".job-search-card__listdate").inner_text().strip() or "N/A"

                # Create a job description for experience extraction
                job_description = f"{title} at {company} in {location}"
                experience = extract_experience_level(job_description)

                if title != "N/A" and company != "N/A":
                    job_data.append({
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Link": link,
                        "Posted": posted_day,
                        "Experience": experience,
                    })
            except Exception as e:
                print(f"Error scraping job: {e}")

        browser.close()

    return job_data


# Job Matching
def match_jobs_with_resume(resume_data, job_data):
    """Match jobs with the resume based on dynamically extracted skills."""
    results = []
    resume_skills_text = " ".join(resume_data['skills']).lower()

    for job in job_data:
        job_text = f"{job['Title']} {job['Company']} {job['Location']} {job['Posted']}".lower()
        job_text = re.sub(r"[^a-zA-Z0-9\s]", "", job_text.strip())

        vectorizer = CountVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_skills_text, job_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        job['Similarity'] = similarity
        results.append({'Job': job, 'Similarity': similarity})

    return sorted(results, key=lambda x: x['Similarity'], reverse=True)


# Streamlit App
def main():
    st.title("LinkedIn Job Scraper & Matcher")
    st.markdown("""
    This app helps you find and match jobs on LinkedIn based on your resume and preferences.
    Upload your resume, specify job criteria, and get a list of top-matched jobs.
    """)

    # User Inputs
    position = st.text_input("Job Title (e.g., Data Scientist, Nurse, Teacher):")
    location = st.text_input("Location (e.g., United States, New York):")
    preference = st.multiselect("Job Type:", ["Remote", "Onsite", "Hybrid"])

    if not preference:
        st.warning("Please select at least one job type.")
        return

    uploaded_file = st.file_uploader("Upload Your Resume (PDF):", type=["pdf"])

    if st.button("Find Jobs"):
        if uploaded_file and position and location:
            with open("uploaded_resume.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            resume_data = parse_resume("uploaded_resume.pdf")

            job_data = []
            for pref in preference:
                preference_map = {"Remote": "2", "Onsite": "1", "Hybrid": "3"}
                pref_code = preference_map[pref]
                job_data += scrape_jobs(position, location, pref_code)

            matched_jobs = match_jobs_with_resume(resume_data, job_data)

            results_df = pd.DataFrame([
                {
                    "Title": match["Job"]["Title"],
                    "Company": match["Job"]["Company"],
                    "Location": match["Job"]["Location"],
                    "Link": match["Job"]["Link"],
                    "Posted": match["Job"]["Posted"],
                    "Experience": match["Job"]["Experience"],
                    "Similarity": f"{match['Similarity']:.2f}"
                }
                for match in matched_jobs[:50]
            ])

            st.subheader("Top Job Matches")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="matched_jobs.csv",
                mime="text/csv",
            )
        else:
            st.error("Please fill all fields and upload a valid resume.")


if __name__ == "__main__":
    main()
