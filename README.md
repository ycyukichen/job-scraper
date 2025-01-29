# Job Scraper with Resume Matching

A dynamic web application that enables users to find and match job opportunities from LinkedIn with their resumes. The application uses natural language processing and machine learning techniques to provide intelligent job matching based on skills, experience, and preferences.

## Features

- **Resume Analysis**: Supports both PDF and DOCX format resumes.
- **Intelligent Matching**: Uses advanced algorithms to match your skills and experience with job requirements.
- **Customizable Search**: Filter by job type (Remote, Hybrid, Onsite) and location.
- **Interactive Results**: View matched jobs with similarity scores and direct links.
- **Downloadable Results**: Provides an option to download the top-matched jobs as a CSV file.
- **Streamlit Deployment**: The app can also be accessed directly via [this link](https://linkedinjob-scraper.streamlit.app/).

---

## Features

1. **Upload Resume**: Users upload their resumes in PDF format.
2. **Specify Job Search Criteria**: Users enter job position, location, preferences (e.g., Remote, Onsite, Hybrid), and years of experience.
3. **Job Scraping**: The application scrapes LinkedIn job postings based on the specified criteria.
4. **Resume Matching**: Extracted skills and experience from the resume are compared to job descriptions using a similarity algorithm.
5. **Results Displayed**: Top-matched jobs are displayed along with the following details:
   - Match Score
   - Posted Date
   - Job Title
   - Company
   - Location
   - Type
   - Link
6. **Download Results**: Option to download the top matches as a CSV file.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ycyukichen/job-scraper.git
   cd job-scraper
   ```

2. **Install Dependencies**:
   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit app:

   ```bash
   streamlit run jobmatch_app.py
   ```

---

## Project Files

- **`jobmatch_app.py`**: Main application code that integrates job scraping, resume parsing, and similarity matching
- **`requirements.txt`**: List of dependencies for the project

---

## How It Works
### Resume Processing

- Extracts key information from your resume using natural language processing
- Identifies skills, experience level, and education
- Creates a structured profile for matching

### Job Matching Algorithm
The matching algorithm considers multiple factors:

- Skill match (40%): Analyzes technical, soft, domain, and certification skills
- Experience match (25%): Compares your experience with job requirements
- Content similarity (20%): Uses TF-IDF vectorization for text comparison
- Location match (15%): Considers location preferences and remote options

### Job Scraping

- Fetches recent job postings from LinkedIn
- Implements rate limiting to prevent blocking
- Handles pagination and error recovery

### Best Practices for Use

1. **Resume Optimization**
   - Include a clear skills section
   - List relevant technical and soft skills
   - Specify years of experience clearly
   - Include education details


2. **Search Tips**
   - Start with broader location searches
   - Try different job title variations
   - Select multiple work preferences for more results

### Limitations

- LinkedIn's job posting format changes may affect scraping
- Rate limiting may slow down large searches
- Resume parsing accuracy depends on document formatting
- Results limited to publicly available LinkedIn job posts

---

## Usage Instructions

1. **Launch the Application**:

   - Start the app with the command: `streamlit run jobmatch_app.py`.
   - Alternatively, access the app directly at [this link](https://linkedinjob-scraper.streamlit.app/).

2. **Upload Resume**:

   - Upload your resume in PDF format.

3. **Input Job Criteria**:

   - Enter job title, location, preferences (e.g., Remote, Onsite, Hybrid) and years of experience.

4. **View and Download Results**:

   - View the top-matched jobs and download them as a CSV file.

---

## Example Outputs

### Uploaded Resume: `resume_sample.pdf`

**Sample Result:**

| Similarity | Posted     | Title                | Company           | Location     | Type     | Link         | 
| ---------- | ---------- | ---------------------| ----------------- |------------- | -------- | ------------ | 
| 0.87       | 2024-12-22 | Data Scientist       | Tech Solutions    | New York, NY | Remote   | Link to job  |
| 0.85       | 2025-01-16 | Machine Learning Eng | Insight Analytics | Boston, MA   | Hybrid   | Link to job  |

---

## Disclaimer

This tool is for educational purposes and personal use. Please respect LinkedIn's terms of service and rate limiting when using this application. This is not affiliated with or endorsed by LinkedIn.

---

## Acknowledgments

- LinkedIn for providing job posting data
- Streamlit for the web framework
- spaCy for NLP capabilities
- scikit-learn for ML functionality

## Project Status
Active development - Contributions and feedback welcome
