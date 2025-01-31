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
from datetime import datetime
import os
import time
from functools import lru_cache
import logging
import html

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResumeData:
    """Data structure for parsed resume information"""
    skills: List[str]
    experience: int
    education: str
    text: str

@dataclass
class JobPosting:
    """Data structure for job posting information"""
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
    def read_file_content(file_path: str, file_type: str) -> str:
        """Read content from PDF or DOCX file"""
        try:
            if file_type == "pdf":
                reader = PdfReader(file_path)
                return " ".join([page.extract_text() for page in reader.pages])
            elif file_type == "docx":
                document = Document(file_path)
                return "\n".join([para.text for para in document.paragraphs])
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords using spaCy with improved filtering"""
        doc = self.nlp(text.lower())
        keywords = set()
        
        # Include both single tokens and key phrases
        for token in doc:
            if (token.is_alpha and not token.is_stop and len(token.text) > 2):
                keywords.add(token.lemma_)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                keywords.add(chunk.text.lower())
                
        return list(keywords)

    def parse_resume(self, file_path: str, file_type: str) -> ResumeData:
        """Parse resume and extract structured information"""
        text = self.read_file_content(file_path, file_type)
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()
        
        skills = self.extract_keywords(text)
        
        # Enhanced experience extraction
        experience_patterns = [
            r"(\d+)\s*(?:\+\s*)?years?(?:\s+of)?\s+experience",
            r"(?:worked|working)\s+(?:for|since)\s+(\d+)\s+years",
        ]
        
        experience = 0
        for pattern in experience_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                experience = int(match.group(1))
                break
        
        # Enhanced education detection
        education_levels = {
            "phd": "PhD",
            "doctorate": "PhD",
            "master": "Master's",
            "mba": "Master's",
            "bachelor": "Bachelor's",
            "associate": "Associate's"
        }
        
        education = "Not specified"
        for key, value in education_levels.items():
            if key in clean_text:
                education = value
                break
                
        return ResumeData(skills=skills, experience=experience, 
                         education=education, text=clean_text)

class LinkedInJobScraper:
    """Handles job scraping from LinkedIn"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.preference_map = {"Remote": "2", "Onsite": "1", "Hybrid": "3"}
        
    @lru_cache(maxsize=100)
    def get_job_page(self, url: str) -> str:
        """Cached method to fetch job pages with rate limiting"""
        time.sleep(1)  # Rate limiting
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching jobs: {e}")
            return ""

    def parse_job_card(self, card: BeautifulSoup, preference: str) -> Union[JobPosting, None]:
        """Parse individual job card with error handling"""
        try:
            title = card.find("h3", class_="base-search-card__title")
            company = card.find("h4", class_="base-search-card__subtitle")
            location = card.find("span", class_="job-search-card__location")
            link = card.find("a", class_="base-card__full-link")
            posted_date = card.find("time")["datetime"] if card.find("time") else "Not specified"
            
            # Use `.get_text(strip=True)` only if the element exists, else provide fallback
            title = title.get_text(strip=True) if title else "Title not found"
            company = company.get_text(strip=True) if company else "Company not found"
            location = location.get_text(strip=True) if location else "Location not found"
            link = link["href"] if link else "Link not available"
            
            # Extract experience requirements
            job_description = f"{title} at {company} in {location}"
            experience_match = re.search(
                r"(\d+\s*-\s*\d+\s*years?|\d+\s*years?|entry-level|mid-level|senior)",
                job_description.lower()
            )
            experience = experience_match.group(0) if experience_match else "Experience not specified"
            
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
            logger.warning(f"Error parsing job card: {e}")
            return None

    def scrape_jobs(self, position: str, location: str, preferences: List[str]) -> List[JobPosting]:
        """Scrape jobs with improved error handling and rate limiting"""
        jobs = []
        
        for preference in preferences:
            base_url = (
                f"https://www.linkedin.com/jobs/search?"
                f"keywords={position}&location={location}"
                f"&f_WT={self.preference_map[preference]}"
            )
            
            html_content = self.get_job_page(base_url)
            if not html_content:
                continue
                    
            soup = BeautifulSoup(html_content, "html.parser")
            job_cards = soup.find_all("div", class_="base-card")
            
            for card in job_cards:
                if job := self.parse_job_card(card, preference):
                    jobs.append(job)
                else:
                    logger.warning(f"Skipped a job card due to parsing issues.")  # Fallback and log the issue
                    
        return jobs


class JobMatcher:
    """Enhanced job matching with multiple scoring factors"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # Consider both single words and pairs
            max_features=5000,    # Limit to most important features
            min_df=2              # Ignore very rare terms
        )
        
        # Skill importance weights by category
        self.skill_weights = {
            'technical': 1.5,    # Programming languages, tools, etc.
            'soft': 1.2,         # Communication, leadership, etc.
            'domain': 1.3,       # Industry-specific knowledge
            'certification': 1.4  # Professional certifications
        }
        
        # Common skill mappings
        self.skill_categories = {
            'technical': {
                'python', 'java', 'sql', 'aws', 'azure', 'docker', 'kubernetes',
                'react', 'javascript', 'machine learning', 'data analysis'
            },
            'soft': {
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum'
            },
            'domain': {
                'healthcare', 'finance', 'marketing', 'sales', 'consulting',
                'manufacturing', 'retail', 'education'
            },
            'certification': {
                'pmp', 'aws certified', 'cissp', 'cpa', 'six sigma',
                'scrum master', 'professional certified'
            }
        }

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        
        # Standardize variations of common terms
        replacements = {
            r'\b(yrs?|years)\b': 'years',
            r'\b(sr\.|senior)\b': 'senior',
            r'\b(jr\.|junior)\b': 'junior',
            r'\b(exp|experience)\b': 'experience',
            r'\b(dev|developer)\b': 'developer',
            r'\b(eng|engineer)\b': 'engineer',
            r'\b(mgr|manager)\b': 'manager'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
            
        return text

    def calculate_skill_match_score(self, resume_skills: List[str], job_text: str) -> float:
        """Calculate weighted skill match score"""
        job_text_lower = job_text.lower()
        total_weight = 0
        matched_weight = 0
        
        for category, weight in self.skill_weights.items():
            category_skills = self.skill_categories[category]
            relevant_skills = set(skill for skill in resume_skills 
                                if skill.lower() in category_skills)
            
            if relevant_skills:
                total_weight += weight
                matched_skills = sum(1 for skill in relevant_skills 
                                   if skill.lower() in job_text_lower)
                matched_weight += (matched_skills / len(relevant_skills)) * weight
        
        return matched_weight / total_weight if total_weight > 0 else 0

    def calculate_experience_match(self, user_experience: int, job_text: str) -> float:
        """Calculate experience match score"""
        # Extract experience requirements from job text
        patterns = [
            r'(\d+)\+?\s*(?:-\s*\d+)?\s*years?',  # e.g., "5+ years" or "5-7 years"
            r'(\d+)\s*years?\s*(?:of)?\s*experience',
            r'senior\s*level.*?(\d+)\s*years?'
        ]
        
        required_exp = None
        for pattern in patterns:
            if match := re.search(pattern, job_text.lower()):
                required_exp = int(match.group(1))
                break
        
        if required_exp is None:
            # If no explicit requirement, infer from level mentions
            if 'senior' in job_text.lower():
                required_exp = 5
            elif 'mid' in job_text.lower():
                required_exp = 3
            elif 'junior' in job_text.lower() or 'entry' in job_text.lower():
                required_exp = 0
            else:
                return 0.5  # Neutral score if no experience info
        
        # Calculate match score based on difference
        diff = abs(user_experience - required_exp)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.8
        elif diff <= 4:
            return 0.6
        else:
            return 0.4

    def calculate_location_match(self, preferred_location: str, job_location: str) -> float:
        """Calculate location match score"""
        preferred = preferred_location.lower()
        actual = job_location.lower()
        
        # Exact city match
        if preferred in actual or actual in preferred:
            return 1.0
            
        # State/region match
        preferred_parts = set(preferred.split())
        actual_parts = set(actual.split())
        common_parts = preferred_parts & actual_parts
        
        if common_parts:
            return 0.8
            
        # Remote flexibility
        if 'remote' in actual:
            return 0.7
            
        return 0.3

    def match_jobs(self, resume_data: ResumeData, jobs: List[JobPosting], 
                  user_experience: int, location_pref: str) -> List[JobPosting]:
        """Enhanced job matching with multiple weighted criteria"""
        for job in jobs:
            # Prepare job text
            job_text = self.preprocess_text(
                f"{job.title} {job.company} {job.location} {job.experience}"
            )
            
            # Calculate individual scores
            skill_score = self.calculate_skill_match_score(resume_data.skills, job_text)
            exp_score = self.calculate_experience_match(user_experience, job_text)
            location_score = self.calculate_location_match(location_pref, job.location)
            
            # Calculate content similarity using TF-IDF
            try:
                resume_text = " ".join(resume_data.skills)
                vectors = self.vectorizer.fit_transform([resume_text, job_text])
                content_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except Exception as e:
                logger.warning(f"Error in similarity calculation: {e}")
                content_score = 0
            
            # Calculate final weighted score
            weights = {
                'skill': 0.4,
                'experience': 0.25,
                'location': 0.15,
                'content': 0.2
            }
            
            final_score = (
                skill_score * weights['skill'] +
                exp_score * weights['experience'] +
                location_score * weights['location'] +
                content_score * weights['content']
            )
            
            job.similarity = min(1.0, final_score)
            
        return sorted(jobs, key=lambda x: x.similarity, reverse=True)

def create_streamlit_app():
    """Create the Streamlit user interface"""
    st.set_page_config(page_title="LinkedIn Job Matcher", layout="wide")
    
    st.title("🎯 LinkedIn Job Matcher")
    st.markdown("""
    Find your perfect job match by uploading your resume and setting your preferences.
    This tool analyzes your skills and experience to find the most relevant opportunities on LinkedIn.
    """)
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("📝 Job Search Parameters")
        position = st.text_input("Job Title:", placeholder="e.g., Data Scientist")
        location = st.text_input("Location:", placeholder="e.g., New York")
        
        st.subheader("🏢 Work Preferences")
        preferences = []
        for pref in ["Remote", "Hybrid", "Onsite"]:
            if st.checkbox(pref):
                preferences.append(pref)
                
        experience = st.number_input(
            "Years of Experience:",
            min_value=0,
            max_value=50,
            value=0,
            step=1
        )
        
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF/DOCX):",
            type=["pdf", "docx"]
        )
    
    # Main content area
    if st.sidebar.button("🔍 Find Matching Jobs"):
        if not all([uploaded_file, position, location, preferences]):
            st.error("Please fill in all required fields and upload your resume.")
            return
            
        try:
            # Initialize components
            parser = ResumeParser()
            scraper = LinkedInJobScraper()
            matcher = JobMatcher()
            
            # Process resume
            file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"
            temp_path = f"temp_resume.{file_type}"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Analyzing your resume..."):
                resume_data = parser.parse_resume(temp_path, file_type)
            
            # Fetch and match jobs
            with st.spinner("Searching for matching jobs..."):
                jobs = scraper.scrape_jobs(position, location, preferences)
                if not jobs:
                    st.warning("No jobs found. Try adjusting your search criteria.")
                    return
                    
                matched_jobs = matcher.match_jobs(
                    resume_data, jobs, experience, location
                )
            
            # Display results
            st.subheader(f"🎉 Found {len(matched_jobs)} Matching Jobs")
            
            results_df = pd.DataFrame([{
                "Match Score": f"{job.similarity * 100:.0f}%",
                "Posted": job.posted_date if job.posted_date else "N/A",
                "Title": html.escape(job.title) if job.title else "N/A",
                "Company": job.company if job.company else "N/A",
                "Location": job.location if job.location else "N/A",
                "Type": job.preference if job.preference else "N/A",
                "Link": job.link if job.link else "N/A"
            } for job in matched_jobs[:50]])
            
            # Create a styled version of the dataframe
            # styled_df = results_df.style.highlight_max(subset=["Match Score"])
            
            # Display the dataframe
            st.dataframe(
                data=results_df,
                column_config={
                    "Link": st.column_config.LinkColumn("Job Link")
                },
                hide_index=True
            )
            
            # Add download button
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results",
                csv,
                "matched_jobs.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Error in job matching process")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    create_streamlit_app()
