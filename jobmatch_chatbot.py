import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import csv
import time
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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
        if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:  # Dynamically extract skills or relevant terms
            skills.add(ent.text.lower())

    # Extract experience
    experience_match = re.search(r'(\d+)\s+years of experience', text, re.IGNORECASE)
    experience = int(experience_match.group(1)) if experience_match else 0

    return {'skills': list(skills), 'experience': experience, 'text': text}

def scrape_jobs(position, location, preference):
    """Scrape jobs from LinkedIn based on user input."""
    def extract_experience_level(job):
        """Extract experience level from the job description."""
        try:
            experience_text = job.text.lower()
            match = re.search(r'\b(\d+\s*-\s*\d+\s*years?|\d+\s*years?|entry-level|mid-level|senior|internship)\b', experience_text)
            return match.group(0) if match else "Not specified"
        except Exception as e:
            print(f"Error extracting experience level: {e}")
            return "Not specified"

    # Set up Selenium WebDriver with headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    url = f"https://www.linkedin.com/jobs/search?keywords={position}&location={location}&f_WT={preference}"

    driver.get(url)
    time.sleep(5)

    jobs = driver.find_elements(By.CLASS_NAME, 'base-card')

    job_data = []
    for job in jobs:
        try:
            title = job.find_element(By.CLASS_NAME, 'base-search-card__title').text.strip() or "N/A"
            company = job.find_element(By.CLASS_NAME, 'base-search-card__subtitle').text.strip() or "N/A"
            location = job.find_element(By.CLASS_NAME, 'job-search-card__location').text.strip() or "N/A"
            link = job.find_element(By.TAG_NAME, 'a').get_attribute('href') or "N/A"
            posted_day = job.find_element(By.CLASS_NAME, 'job-search-card__listdate').text.strip() or "N/A"
            experience = extract_experience_level(job)

            # Add only valid job entries
            if title != "N/A" and company != "N/A":
                job_data.append({'Title': title, 'Company': company, 'Location': location, 'Link': link, 'Posted': posted_day, 'Experience': experience})
        except Exception as e:
            print(f"Error scraping job: {e}")

    driver.quit()
    return job_data


def match_jobs_with_resume(resume_data, job_data):
    """Match jobs with the resume based on dynamically extracted skills."""
    results = []
    resume_skills_text = " ".join(resume_data['skills']).lower()

    if not resume_skills_text:
        print("Error: Resume skills are empty.")
        return results

    for job in job_data:
        # Combine job fields to form job description text
        job_text = f"{job['Title']} {job['Company']} {job['Location']} {job['Posted']}".lower()
        job_text = re.sub(r"[^a-zA-Z0-9\s]", "", job_text.strip())

        if not job_text:
            print("Warning: Job text is empty for a listing.")
            continue

        try:
            # Match dynamically extracted resume skills with the job description
            vectorizer = CountVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform([resume_skills_text, job_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            similarity = 0.0

        job['Similarity'] = similarity
        results.append({'Job': job, 'Similarity': similarity})

    return sorted(results, key=lambda x: x['Similarity'], reverse=True)

def main():
    st.markdown("""
    ## Welcome to the Universal Job Scraper
    This application helps you find and match jobs from LinkedIn across all industries based on your resume and preferences. 
    Upload your resume and provide the job title, location, and preferences (e.g., remote, onsite, or hybrid). 
    The system dynamically extracts your skills and matches them with job descriptions.
    """)

    st.title("Job Scraper with Resume Matching")

    # User inputs
    position = st.text_input("Enter the job position (e.g., Nurse, Teacher, Engineer, etc.): *")
    location = st.text_input("Enter the location (e.g., United States, New York, etc.): *")
    preference = st.multiselect("Select your preference (multiple allowed): *", ["Remote", "Onsite", "Hybrid"])

    if not preference:
        st.warning("Please select at least one preference.")
        return

    preference_map = {"Remote": "2", "Onsite": "1", "Hybrid": "3"}
    preference_codes = [preference_map[pref] for pref in preference]

    uploaded_file = st.file_uploader("Upload your resume (PDF only): *", type=["pdf"])

    if st.button("Find Jobs"):
        if uploaded_file and position and location:
            # Save uploaded file temporarily
            with open("uploaded_resume.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Parse the resume dynamically without predefined skills
            resume_data = parse_resume("uploaded_resume.pdf")

            # Scrape jobs for all preferences
            job_data = []
            for pref_code in preference_codes:
                st.write(f"Scraping jobs with preference: {pref_code}. This may take a few moments...")
                job_data += scrape_jobs(position, location, pref_code)

            # Match jobs with resume
            matched_jobs = match_jobs_with_resume(resume_data, job_data)

            # Display results
            st.subheader("Top Job Matches")
            results_df = pd.DataFrame([
                {
                    "Title": match["Job"].get("Title", "N/A"),
                    "Company": match["Job"].get("Company", "N/A"),
                    "Location": match["Job"].get("Location", "N/A"),
                    "Link": match["Job"].get("Link", "N/A"),
                    "Posted": match["Job"].get("Posted", "N/A"),
                    "Experience": match["Job"].get("Experience", "Not specified"),
                    "Similarity": f"{match['Similarity']:.2f}"
                }
                for match in matched_jobs[:50]
            ])

            st.dataframe(results_df)

            # Option to download results
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="matched_jobs.csv",
                mime="text/csv",
            )
        else:
            st.error("Please fill all inputs and upload a valid resume!")

if __name__ == "__main__":
    main()

