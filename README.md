# Job Scraper with Resume Matching

A dynamic web application that enables users to find and match job opportunities from LinkedIn with their resumes. The application uses Natural Language Processing (NLP) to extract skills and experience from resumes and dynamically matches them with job descriptions. It is industry-agnostic and works for all types of job seekers.

## Features

- **Dynamic Skill Extraction**: Uses NLP to extract skills and experience directly from user resumes.
- **Job Scraping**: Retrieves job postings from LinkedIn based on user input (position, location, and preference).
- **Resume Matching**: Calculates similarity scores between the resume and job descriptions.
- **Experience Level Identification**: Extracts required experience levels from job postings (e.g., `2-5 years`, `Mid-level`).
- **Downloadable Results**: Provides an option to download the top-matched jobs as a CSV file.

---

## How It Works

1. **Upload Resume**: Users upload their resumes in PDF format.
2. **Specify Job Search Criteria**: Users enter job position, location, and preferences (e.g., remote, onsite, hybrid).
3. **Job Scraping**: The application scrapes LinkedIn job postings based on the specified criteria.
4. **Resume Matching**: Extracted skills and experience from the resume are compared to job descriptions using a similarity algorithm.
5. **Results Displayed**: Top-matched jobs are displayed along with the following details:
   - Job Title
   - Company
   - Location
   - Posted Date
   - Experience Level
   - Similarity Score
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
   pip install -r r
   ```

3. requirements.txt**Set Up ChromeDriver**:
   Download the appropriate version of [ChromeDriver](https://chromedriver.chromium.org/downloads) for your system and ensure it is in your PATH.

4. **Run the Application**:
   Start the Streamlit app:

   ```bash
   streamlit run jobmatch_chatbot.py
   ```

---

## Project Files

- **`jobmatch_chatbot.py`**: Main application code that integrates job scraping, resume parsing, and similarity matching.
- **`resume_sample.pdf`**: Sample resume to test the application.
- **`linkedin_jobs.csv`**: Example CSV output of matched jobs.
- **`requirements.txt`**: List of dependencies for the project.

---

## Dependencies

The project uses the following Python libraries:

- `streamlit`: For building the web interface.
- `selenium`: For scraping job postings from LinkedIn.
- `PyPDF2`: For extracting text from PDF resumes.
- `spacy`: For NLP-based skill extraction.
- `sklearn`: For calculating similarity scores using vectorization.
- `pandas`: For handling job data and generating results.

---

## Usage Instructions

1. **Launch the Application**:

   - Start the app with the command: `streamlit run jobmatch_chatbot.py`.

2. **Upload Resume**:

   - Upload your resume in PDF format.

3. **Input Job Criteria**:

   - Enter job title, location, and preferences (e.g., Remote, Onsite, Hybrid).

4. **View and Download Results**:

   - View the top-matched jobs and download them as a CSV file.

---

## Example Outputs

### Uploaded Resume: `resume_sample.pdf`

**Sample Result:**

| Title                | Company           | Location     | Experience | Similarity |
| -------------------- | ----------------- | ------------ | ---------- | ---------- |
| Data Scientist       | Tech Solutions    | New York, NY | 2-5 years  | 0.87       |
| Machine Learning Eng | Insight Analytics | Boston, MA   | Mid-level  | 0.85       |

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

