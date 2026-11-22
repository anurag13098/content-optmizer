# Content Optimizer

A Streamlit prototype that compares a pasted article with pages appearing in Google search results.

## Features
- Accepts a target search query and article text.
- Retrieves organic Google results using SerpAPI.
- Downloads result pages and extracts main text using Trafilatura.
- Displays extraction status and word counts.
- Uses TF-IDF (single words and two-word phrases) to find recurring competitor wording.
- Compares phrases with the user's article and highlights phrases not found.
- Exports the content-gap report as CSV.

## Setup on Windows (PowerShell)

1. Open this folder in VS Code and open its terminal.
2. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Copy `.env.example` to a new file named `.env`. Put your real SerpAPI key there:

```text
SERPAPI_KEY=your_actual_key_here
```

Do not share `.env` or upload it to GitHub. The `.gitignore` file excludes it.

5. Start the app:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501). Stop the app with Ctrl+C.

## How the analysis works
The app creates a TF-IDF document-term matrix from successfully extracted competitor pages. It counts how many competitor pages contain each phrase, then checks whether the phrase appears in the user's article. Phrases found on multiple competitor pages but not found in the draft are surfaced for editorial review.

## Limitations
- A SerpAPI key is required; none is bundled.
- Some sites block automated downloads, require JavaScript, or expose little readable text.
- Search results vary by location, language, and time.
- TF-IDF detects recurring wording, not full semantic meaning or search intent.
- Suggestions are not mandatory keywords and do not guarantee rankings.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `.env.example` — API key template
- `.gitignore` — excludes secrets and environment files
