import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
import trafilatura
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()


def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


SERPAPI_KEY = get_secret("SERPAPI_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

st.set_page_config(page_title="Content Optimizer", page_icon="📝", layout="wide")
st.title("📝 Content Optimizer")
st.write(
    "Research pages ranking for your target query, identify content gaps, "
    "and generate a rewritten draft with newly added phrases highlighted in bold."
)

with st.sidebar:
    st.header("Research settings")
    result_limit = st.slider("Search results", 3, 10, 10)
    max_terms = st.slider("Suggestions to show", 10, 40, 25)
    min_coverage = st.slider("Minimum competitor page coverage", 1, 10, 2)
    st.divider()
    st.caption("AI rewriting uses Gemini. Search research uses SerpAPI.")

keyword = st.text_input("Target search query", placeholder="e.g. amino club coupon code")
content = st.text_area(
    "Paste your article", height=300,
    placeholder="Paste your complete article here..."
)


def search_google(query, count):
    response = requests.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": count},
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


def rewrite_with_gemini(article, target_query, missing_terms, competitor_pages):
    client = genai.Client(api_key=GEMINI_API_KEY)
    keyword_list = ", ".join(missing_terms[:30]) if missing_terms else "No clear missing phrases were detected."
    competitor_context = "\n".join(
        f"- {p['title']}: {p['snippet']}" for p in competitor_pages[:10] if p.get("title")
    )
    prompt = f"""
You are an expert SEO content editor. Rewrite the user's complete article for the target search query.

TARGET SEARCH QUERY:
{target_query}

POTENTIAL MISSING TOPICS / PHRASES FROM COMPETITOR RESEARCH:
{keyword_list}

COMPETITOR TITLES AND SEARCH SNIPPETS (research context only; do not copy them):
{competitor_context}

ORIGINAL ARTICLE:
{article}

REQUIREMENTS:
1. Produce a complete, polished, original rewritten article, not an outline or list of suggestions.
2. Preserve the original article's factual meaning, named entities, prices, discount codes, dates, links, and claims. Do not invent facts, offers, statistics, testimonials, or product details.
3. Add relevant missing phrases naturally where they genuinely fit. Do not force irrelevant phrases or keyword-stuff.
4. Put **Markdown bold** around each newly added target phrase or meaningful new wording. Do not bold text already present unless needed for a heading.
5. Preserve useful headings and improve the structure where appropriate. Use clear H2/H3 headings.
6. Do not mention competitor websites or say that you used AI.
7. Return only the finished article in Markdown. No preface, analysis, checklist, or explanation.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.5},
    )
    result = (response.text or "").strip()
    if not result:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return result


def perform_research(query, article, count, minimum, limit):
    if not query.strip():
        st.warning("Enter a target search query.")
        return
    if len(article.split()) < 50:
        st.warning("Please paste at least 50 words of article content.")
        return
    if not SERPAPI_KEY:
        st.error("SerpAPI key missing. Add SERPAPI_KEY to Streamlit Secrets.")
        return

    try:
        with st.spinner("Retrieving Google results..."):
            results = search_google(query.strip(), count)
    except Exception as exc:
        st.error(f"Search request failed: {exc}")
        return
    if not results:
        st.warning("No organic results were returned.")
        return

    pages = []
    jobs = {}
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
                page_text, error = future.result()
            except Exception:
                page_text, error = None, "Unexpected extraction error."
            pages.append({
                "position": pos,
                "title": item.get("title", "Untitled"),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "text": page_text,
                "error": error,
            })
            finished += 1
            progress.progress(finished / max(len(jobs), 1))
    progress.empty()
    pages.sort(key=lambda p: p["position"])
    good_pages = [p for p in pages if p["text"]]
    if not good_pages:
        st.error("No page text was extracted. Try another query or retry later.")
        return

    report, report_error = make_gap_report(
        [p["text"] for p in good_pages], article, minimum, limit
    )
    missing_terms = []
    if not report.empty:
        missing_terms = report.loc[
            report["Your article"] == "Not found", "Topic / phrase"
        ].tolist()

    st.session_state["research_data"] = {
        "pages": pages,
        "report": report.to_dict(orient="records"),
        "report_error": report_error,
        "missing_terms": missing_terms,
        "query": query.strip(),
        "article": article,
    }
    # A new research run should clear any previous generated draft.
    st.session_state.pop("optimized_article", None)


if "research_data" not in st.session_state:
    st.session_state["research_data"] = None

if st.button("Analyze my content", type="primary", use_container_width=True):
    perform_research(keyword, content, result_limit, min_coverage, max_terms)

# Render saved research on every rerun. This keeps the rewrite button available
# after Streamlit reruns the script when the user clicks it.
data = st.session_state.get("research_data")
if data:
    pages = data["pages"]
    good_pages = [p for p in pages if p["text"]]
    st.success(f"Retrieved {len(pages)} organic results; extracted readable text from {len(good_pages)} pages.")
    st.subheader("Search results and extracted page text")
    for page in pages:
        with st.expander(f"{page['position']}. {page['title']}"):
            st.write(page["url"])
            if page["snippet"]:
                st.caption(page["snippet"])
            if page["text"]:
                st.success(f"Content extracted: {len(page['text'].split()):,} words")
                st.text_area(
                    "Extracted text preview", page["text"][:4000], height=150,
                    key=f"page_preview_{page['position']}", disabled=True
                )
            else:
                st.warning(page["error"] or "Could not extract text.")

    st.subheader("Content gap report")
    if data["report"]:
        report_df = pd.DataFrame(data["report"])
        a, b, c = st.columns(3)
        a.metric("Pages analyzed", len(good_pages))
        b.metric("Topics displayed", len(report_df))
        c.metric("Potential gaps", len(data["missing_terms"]))
        st.caption("These phrases are research prompts, not mandatory keywords.")
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download report as CSV", report_df.to_csv(index=False).encode("utf-8"),
            "content_gap_report.csv", "text/csv"
        )
        if data["missing_terms"]:
            st.write("Topics to review:", ", ".join(data["missing_terms"][:15]))
    else:
        st.warning(data["report_error"] or "No keyword gaps were identified.")

    st.divider()
    st.subheader("✨ Generate your optimized article")
    st.write("Your research is saved. Click below to ask Gemini to rewrite the complete article and bold newly added relevant phrases.")

    if not GEMINI_API_KEY:
        st.error("Gemini API key missing. Add GEMINI_API_KEY to Streamlit Secrets.")
    else:
        if st.button("Generate optimized article with Gemini", type="primary", use_container_width=True):
            try:
                with st.spinner("Gemini is rewriting your article. This may take a little while..."):
                    optimized = rewrite_with_gemini(
                        data["article"], data["query"], data["missing_terms"], pages
                    )
                st.session_state["optimized_article"] = optimized
                st.success("Your optimized article has been generated.")
            except Exception as exc:
                st.error(f"Gemini rewriting failed: {exc}")

if st.session_state.get("optimized_article"):
    st.subheader("Your optimized article")
    st.markdown(st.session_state["optimized_article"])
    st.download_button(
        "Download optimized article (.md)",
        st.session_state["optimized_article"],
        file_name="optimized_article.md",
        mime="text/markdown",
    )
    st.caption("Review all facts, links, prices, dates, and promotional claims before publishing.")

st.divider()
st.info(
    "Limitations: some sites block automated access or require JavaScript. "
    "TF-IDF finds recurring wording, not complete meaning. Search results vary, "
    "and this tool cannot guarantee rankings. Gemini may also make mistakes, "
    "so review the rewritten article before publishing."
)
st.caption("Python • Streamlit • SerpAPI • Gemini • Trafilatura • scikit-learn")
