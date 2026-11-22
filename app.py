import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
import trafilatura
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY", "").strip()

st.set_page_config(page_title="Content Optimizer", page_icon="📝", layout="wide")
st.title("📝 Content Optimizer")
st.write("Compare your article with pages ranking for your target search query.")

with st.sidebar:
    st.header("Settings")
    result_limit = st.slider("Search results", 3, 10, 10)
    max_terms = st.slider("Suggestions to show", 10, 40, 25)
    min_coverage = st.slider("Minimum competitor page coverage", 1, 10, 2)

keyword = st.text_input("Target search query", placeholder="e.g. amino club coupon code")
content = st.text_area("Paste your article", height=300, placeholder="Paste your complete article here...")

def search_google(query, count):
    response = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "api_key": API_KEY, "num": count},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("organic_results", [])

def extract_page(url):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 ContentOptimizer/1.0"},
            timeout=(8, 18),
        )
        response.raise_for_status()
        text = trafilatura.extract(
            response.text, include_comments=False, include_tables=True, favor_precision=True
        )
        if not text or len(text.split()) < 40:
            return None, "Too little readable text was found."
        return text, None
    except requests.exceptions.Timeout:
        return None, "The page timed out."
    except requests.exceptions.RequestException as exc:
        return None, f"Download failed ({exc.__class__.__name__})."
    except Exception as exc:
        return None, f"Extraction failed ({exc.__class__.__name__})."

def make_gap_report(competitor_texts, article, minimum, limit):
    docs = [re.sub(r"\s+", " ", t.lower()).strip() for t in competitor_texts if t]
    if not docs:
        return pd.DataFrame(), "No competitor text was extracted."
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            min_df=2 if len(docs) >= 3 else 1,
            max_df=0.95 if len(docs) >= 4 else 1.0,
            max_features=2500, sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(), "Not enough varied text was found to identify recurring terms."
    terms = vectorizer.get_feature_names_out()
    scores = matrix.mean(axis=0).A1
    analyzer = vectorizer.build_analyzer()
    article_terms = set(analyzer(article.lower()))
    rows = []
    for j, term in enumerate(terms):
        coverage = int((matrix[:, j] > 0).sum())
        if coverage < minimum:
            continue
        rows.append({
            "Topic / phrase": term,
            "Competitor pages": f"{coverage}/{len(docs)}",
            "Your article": "Covered" if term in article_terms else "Not found",
            "Priority score": round(float(scores[j]) * coverage / len(docs), 4),
            "_covered": term in article_terms,
            "_coverage": coverage,
        })
    if not rows:
        return pd.DataFrame(), "No phrases met the selected coverage threshold. Lower it or extract more pages."
    report = pd.DataFrame(rows).sort_values(
        ["_covered", "_coverage", "Priority score"], ascending=[True, False, False]
    ).head(limit)
    return report.drop(columns=["_covered", "_coverage"]).reset_index(drop=True), None

def run():
    if not keyword.strip():
        st.warning("Enter a target search query.")
        return
    if len(content.split()) < 50:
        st.warning("Please paste at least 50 words of article content.")
        return
    if not API_KEY:
        st.error("SerpAPI key missing. Add SERPAPI_KEY=your_key to a .env file.")
        return

    try:
        with st.spinner("Retrieving Google results..."):
            results = search_google(keyword.strip(), result_limit)
    except Exception as exc:
        st.error(f"Search request failed: {exc}")
        return
    if not results:
        st.warning("No organic results were returned.")
        return

    st.success(f"Retrieved {len(results)} organic results.")
    st.subheader("Search results and extracted page text")
    jobs = {}
    pages = []
    progress = st.progress(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pos, item in enumerate(results, 1):
            url = item.get("link", "")
            if url:
                jobs[pool.submit(extract_page, url)] = (pos, item)
        finished = 0
        for future in as_completed(jobs):
            pos, item = jobs[future]
            try:
                text, error = future.result()
            except Exception:
                text, error = None, "Unexpected extraction error."
            pages.append({
                "position": pos, "title": item.get("title", "Untitled"),
                "url": item.get("link", ""), "snippet": item.get("snippet", ""),
                "text": text, "error": error,
            })
            finished += 1
            progress.progress(finished / max(len(jobs), 1))
    progress.empty()
    pages.sort(key=lambda p: p["position"])
    good = [p for p in pages if p["text"]]
    st.metric("Pages with extracted text", f"{len(good)} / {len(results)}")

    for p in pages:
        with st.expander(f"{p['position']}. {p['title']}"):
            st.write(p["url"])
            if p["snippet"]:
                st.caption(p["snippet"])
            if p["text"]:
                st.success(f"Content extracted: {len(p['text'].split()):,} words")
                st.text_area("Extracted text preview", p["text"][:4000],
                             height=150, key=f"page_{p['position']}", disabled=True)
            else:
                st.warning(p["error"] or "Could not extract text.")

    if not good:
        st.error("No page text was extracted. Try another query or retry later.")
        return

    st.subheader("Content gap report")
    report, error = make_gap_report([p["text"] for p in good], content, min_coverage, max_terms)
    if error:
        st.warning(error)
        return
    gaps = report[report["Your article"] == "Not found"]
    a, b, c = st.columns(3)
    a.metric("Pages analyzed", len(good))
    b.metric("Topics displayed", len(report))
    c.metric("Potential gaps", len(gaps))
    st.caption("These phrases are research prompts, not mandatory keywords.")
    st.dataframe(report, use_container_width=True, hide_index=True)
    if not gaps.empty:
        st.write("Topics to review:", ", ".join(gaps["Topic / phrase"].head(15)))
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download report as CSV", csv, "content_gap_report.csv", "text/csv")
    st.info(
        "Limitations: some sites block automated access or require JavaScript. "
        "TF-IDF finds recurring wording, not complete meaning. Search results vary, "
        "and this tool cannot guarantee rankings."
    )

if st.button("Analyze my content", type="primary", use_container_width=True):
    run()

st.divider()
st.caption("Python • Streamlit • SerpAPI • Trafilatura • scikit-learn")
