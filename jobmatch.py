from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to parse the resume
def parse_resume(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Extract skills
    skills_keywords = ['Python', 'Machine Learning', 'Data Science', 'SQL', 'Statistics', 'Deep Learning', 'TensorFlow', 'NLP']
    skills = [skill for skill in skills_keywords if re.search(rf'\b{skill}\b', text, re.IGNORECASE)]

    # Extract experience
    experience_match = re.search(r'(\d+)\s+years of experience', text, re.IGNORECASE)
    experience = int(experience_match.group(1)) if experience_match else 0

    return {
        'skills': skills,
        'experience': experience,
        'text': text
    }

# Function to match jobs with the resume
def match_jobs_with_resume(resume_data, job_data):
    results = []
    resume_text = " ".join(resume_data['skills'])

    for job in job_data:
        job_text = f"{job['Title']} {job['Company']} {job['Location']} {job['Posted']}"
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        job['Similarity'] = similarity
        results.append({'Job': job, 'Similarity': similarity})

    results = sorted(results, key=lambda x: x['Similarity'], reverse=True)
    return results

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # Make sure to install the ChromeDriver

# Define the URL
url = "https://www.linkedin.com/jobs/search?keywords=data+scientist&location=United+States"

# Open the URL
driver.get(url)
time.sleep(5)  # Wait for the page to load

# Find job postings
jobs = driver.find_elements(By.CLASS_NAME, 'base-card')

# Prepare job data list
job_data = []
for job in jobs:
    try:
        title = job.find_element(By.CLASS_NAME, 'base-search-card__title').text.strip()
        company = job.find_element(By.CLASS_NAME, 'base-search-card__subtitle').text.strip()
        location = job.find_element(By.CLASS_NAME, 'job-search-card__location').text.strip()
        link = job.find_element(By.TAG_NAME, 'a').get_attribute('href')
        posted_day = job.find_element(By.CLASS_NAME, 'job-search-card__listdate').text.strip() if job.find_element(By.CLASS_NAME, 'job-search-card__listdate') else "N/A"

        # Validate extracted fields
        if title and company and location and link:
            job_data.append({'Title': title, 'Company': company, 'Location': location, 'Link': link, 'Posted': posted_day})
        else:
            print(f"Skipped incomplete job entry: Title={title}, Company={company}, Location={location}, Link={link}")

    except Exception as e:
        print(f"Error scraping job: {e}")

# Parse the resume
resume_data = parse_resume('resume_sample.pdf')

# Match jobs with the resume
matched_jobs = match_jobs_with_resume(resume_data, job_data)

# Save matched jobs with similarity to a CSV file
with open('linkedin_jobs.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Title', 'Company', 'Location', 'Link', 'Posted', 'Similarity'])
    writer.writeheader()
    for match in matched_jobs[:30]:  # Top 30 matches
        writer.writerow({
            'Title': match['Job']['Title'],
            'Company': match['Job']['Company'],
            'Location': match['Job']['Location'],
            'Link': match['Job']['Link'],
            'Posted': match['Job']['Posted'],
            'Similarity': f"{match['Similarity']:.2f}"
        })

print(f"Saved top 30 matched jobs to 'linkedin_jobs.csv'.")

# Display top 20 matches
for match in matched_jobs[:20]:
    print(f"Job Title: {match['Job']['Title']}, Company: {match['Job']['Company']}, Location: {match['Job']['Location']}, Similarity: {match['Similarity']:.2f}")

# Close the browser
driver.quit()
