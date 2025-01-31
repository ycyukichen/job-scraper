import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.cli import download
import logging
from typing import Dict, List, Union
from dataclasses import dataclass
import os
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResumeData:
    """Structure for parsed resume information"""
    skills: List[str]
    experience: int
    education: str
    text: str

@dataclass
class JobPosting:
    """Structure for job posting information"""
    title: str
    company: str
    location: str
    link: str
    posted_date: str
    experience: str
    preference: str
    similarity: float = 0.0

class ResumeParser:
    """Handles resume parsing and keyword extraction"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def read_file(file_path: str, file_type: str) -> str:
        """Read content from PDF/DOCX files"""
        try:
            if file_type == "pdf":
                reader = PdfReader(file_path)
                return " ".join([page.extract_text() for page in reader.pages])
            elif file_type == "docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"File read error: {e}")
            raise

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills with spaCy NLP"""
        doc = self.nlp(text.lower())
        skills = set()
        
        # Extract tokens and noun phrases
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                skills.add(token.lemma_)
        
        for chunk in doc.noun_chunks:
            clean_chunk = re.sub(r'\s+', ' ', chunk.text).strip()
            if len(clean_chunk) > 2:
                skills.add(clean_chunk)
                
        return list(skills)

    def parse_resume(self, file_path: str, file_type: str) -> ResumeData:
        """Parse resume with enhanced extraction"""
        text = self.read_file(file_path, file_type)
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        return ResumeData(
            skills=self.extract_skills(clean_text),
            experience=self._extract_experience(clean_text),
            education=self._extract_education(clean_text),
            text=clean_text
        )

    def _extract_experience(self, text: str) -> int:
        """Extract years of experience"""
        patterns = [
            r'(\d+)\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
            r'experience:\s*(\d+)\s*[+]?',
            r'worked\s+(\d+)\s+years'
        ]
        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                return int(match.group(1))
        return 0

    def _extract_education(self, text: str) -> str:
        """Extract education level"""
        levels = {
            'phd': 'PhD', 
            'master': "Master's",
            'bachelor': "Bachelor's",
            'associate': "Associate's"
        }
        for key in levels:
            if re.search(rf'\b{key}\b', text, re.IGNORECASE):
                return levels[key]
        return 'Not specified'

class LinkedInJobScraper:
    """Improved LinkedIn job scraper with anti-blocking measures"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        self.preference_map = {"Remote": "2", "Hybrid": "3", "Onsite": "1"}

    @lru_cache(maxsize=100)
    def fetch_page(self, url: str) -> str:
        """Cached page fetcher with rate limiting"""
        time.sleep(1.5)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return ""

    def parse_job_card(self, card: BeautifulSoup, preference: str) -> Union[JobPosting, None]:
        """Robust job card parser with multiple fallbacks"""
        try:
            # Primary selectors
            title_elem = card.select_one('h3.base-search-card__title, h4.job-card-title')
            company_elem = card.select_one('h4.base-search-card__subtitle, div.job-card-company-name')
            location_elem = card.select_one('span.job-search-card__location, div.job-card-location')
            link_elem = card.select_one('a.base-card__full-link, a.job-card-link')
            
            # Fallback selectors
            if not title_elem:
                title_elem = card.find('a', {'data-tracking-control-name': 'public_jobs_jserp-result_search-card'})
            if not company_elem:
                company_elem = card.find('a', {'data-tracking-control-name': 'public_jobs_company-name'})
            
            # Text extraction
            title = title_elem.get_text(strip=True) if title_elem else 'N/A'
            company = company_elem.get_text(strip=True) if company_elem else 'N/A'
            location = location_elem.get_text(strip=True) if location_elem else 'N/A'
            link = link_elem['href'].split('?')[0] if link_elem else 'N/A'
            
            # Date parsing
            date_elem = card.select_one('time.job-search-card__listdate, time.job-card-list__date')
            posted_date = date_elem['datetime'] if date_elem else 'N/A'
            
            # Experience parsing
            job_text = f"{title} {company} {location}".lower()
            exp_match = re.search(r'(\d+\+?[\s-]*\d*)\s*(years?|yrs?)|(senior|mid|junior)', job_text)
            experience = exp_match.group(1) if exp_match and exp_match.group(1) else 'N/A'

            return JobPosting(
                title=title,
                company=company,
                location=location,
                link=link,
                posted_date=posted_date,
                experience=experience,
                preference=preference
            )
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def scrape_jobs(self, position: str, location: str, preferences: List[str]) -> List[JobPosting]:
        """Main scraping function"""
        jobs = []
        for pref in preferences:
            url = f"https://www.linkedin.com/jobs/search?keywords={position}&location={location}&f_WT={self.preference_map[pref]}"
            page_content = self.fetch_page(url)
            if not page_content:
                continue
                
            soup = BeautifulSoup(page_content, 'html.parser')
            cards = soup.find_all('div', class_='base-card')
            
            for card in cards:
                if job := self.parse_job_card(card, pref):
                    jobs.append(job)
        return jobs

class JobMatcher:
    """Enhanced job matching algorithm"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.base_score = 0.25  # Minimum match score

    def calculate_match(self, resume: ResumeData, jobs: List[JobPosting], exp: int, location: str) -> List[JobPosting]:
        """Calculate matching scores"""
        resume_text = ' '.join(resume.skills)
        
        for job in jobs:
            job_text = f"{job.title} {job.company} {job.experience}".lower()
            
            # TF-IDF Similarity
            vectors = self.vectorizer.fit_transform([resume_text, job_text])
            content_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Experience match
            exp_score = self._match_experience(exp, job.experience)
            
            # Location match
            loc_score = self._match_location(location, job.location)
            
            # Skill match
            skill_score = len(set(resume.skills) & set(job_text.split())) / len(resume.skills)
            
            # Weighted final score
            final_score = (
                0.4 * content_score +
                0.3 * exp_score +
                0.2 * loc_score +
                0.1 * skill_score +
                self.base_score
            )
            job.similarity = min(1.0, final_score)
            
        return sorted(jobs, key=lambda x: x.similarity, reverse=True)

    def _match_experience(self, user_exp: int, job_exp: str) -> float:
        """Experience matching logic"""
        if 'senior' in job_exp.lower():
            return 0.8 if user_exp >= 5 else 0.4
        if 'mid' in job_exp.lower():
            return 0.7 if user_exp >= 3 else 0.5
        if 'junior' in job_exp.lower():
            return 0.6 if user_exp <= 2 else 0.4
        return 0.5

    def _match_location(self, user_loc: str, job_loc: str) -> float:
        """Location matching logic"""
        user_loc = user_loc.lower()
        job_loc = job_loc.lower()
        if 'remote' in job_loc:
            return 0.9
        if any(word in job_loc for word in user_loc.split()):
            return 0.7
        return 0.3

def main():
    """Streamlit UI Configuration"""
    st.set_page_config(page_title="AI Job Matcher", layout="wide")
    st.title("🔍 AI-Powered Job Matching System")
    
    with st.sidebar:
        st.header("Search Parameters")
        position = st.text_input("Desired Position", placeholder="Software Engineer")
        location = st.text_input("Preferred Location", placeholder="New York")
        experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=0)
        preferences = [pref for pref in ["Remote", "Hybrid", "Onsite"] if st.checkbox(pref)]
        resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    
    if st.sidebar.button("Find Matching Jobs"):
        if not all([resume_file, position, location, preferences]):
            st.error("Please complete all required fields")
            return
            
        try:
            # Process resume
            parser = ResumeParser()
            file_type = 'pdf' if resume_file.name.endswith('.pdf') else 'docx'
            temp_file = f"temp_resume.{file_type}"
            
            with open(temp_file, "wb") as f:
                f.write(resume_file.getbuffer())
                
            resume_data = parser.parse_resume(temp_file, file_type)
            
            # Scrape and match jobs
            scraper = LinkedInJobScraper()
            matcher = JobMatcher()
            
            with st.spinner("Scanning LinkedIn for opportunities..."):
                jobs = scraper.scrape_jobs(position, location, preferences)
                matched_jobs = matcher.calculate_match(resume_data, jobs, experience, location)
            
            # Display results
            st.subheader(f"🎯 Found {len(matched_jobs)} Relevant Positions")
            
            results_df = pd.DataFrame([{
                "Match %": f"{job.similarity * 100:.0f}%",
                "Title": job.title,
                "Company": job.company,
                "Location": job.location,
                "Type": job.preference,
                "Posted": job.posted_date,
                "Link": job.link
            } for job in matched_jobs[:50]])
            
            st.dataframe(
                results_df,
                column_config={"Link": st.column_config.LinkColumn()},
                hide_index=True,
                use_container_width=True
            )
            
            # Export option
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Export Results",
                csv,
                "job_matches.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"System error: {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()
