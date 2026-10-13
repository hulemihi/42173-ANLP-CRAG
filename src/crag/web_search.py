"""
Web search fallback: rewrite the question into a search query and fetch passages from the web.

Searches both Tavily (general web fact-checking) and NCBI PubMed (peer-reviewed biomedical abstracts).
Configured via environment variables:
  - TAVILY_API_KEY: API key for Tavily search (https://tavily.com)
  - NCBI_EMAIL: User email for NCBI E-utilities (default: student@uts.edu.au)
  - NCBI_API_KEY: Optional NCBI API key to increase rate limit
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_PASSAGE_WORDS = 200
NCBI_TOOL = "42173_anlp_crag"
DEFAULT_EMAIL = "student@uts.edu.au"
CACHE_PATH = Path(__file__).resolve().parents[2] / "indexes" / "web_search_cache.json"

_LAST_PUBMED_CALL = 0.0

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "it", "its", "that",
    "this", "these", "those", "of", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "out",
    "what", "which", "who", "whom", "whose", "why", "how", "true", "false",
    "according", "claims", "claim", "reported", "shows", "say", "says",
}


def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _clean_passage(text, max_words=MAX_PASSAGE_WORDS):
    """
    Normalize whitespaces and clip text to at most max_words words.
    """
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


def _rate_limit_pubmed(min_interval=0.35):
    """
    Enforce politeness to prevent exceeding NCBI's 3 requests/sec rate limit.
    """
    global _LAST_PUBMED_CALL
    now = time.time()
    elapsed = now - _LAST_PUBMED_CALL
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_PUBMED_CALL = time.time()


def rewrite_query(question, model=None, tokenizer=None):
    """
    Rewrite a question or claim into a keyword search query (str).
    Uses Qwen when available, with a fast heuristic fallback.
    """
    if not question or not question.strip():
        return ""

    if model is not None and tokenizer is not None:
        message = (
            "Extract 2 to 5 essential search keywords from the following biomedical question or health claim. "
            "Focus on the core entities, condition, and treatment. "
            "Do not explain, do not use quotes or punctuation, and answer with only the search keywords.\n\n"
            f"Input: {question}\nSearch keywords:"
        )
        try:
            from crag.generator import _generate_text
            rewritten = _generate_text(model, tokenizer, message, max_new_tokens=30)
            rewritten = re.sub(r"^(Keywords|Search keywords|Query):\s*", "", rewritten, flags=re.IGNORECASE)
            rewritten = rewritten.strip(" \"'\n\r")
            if rewritten:
                return rewritten
        except Exception:
            pass

    # Rule-based fallback (e.g. for check_crag_ready or when model is None)
    clean = re.sub(r"[^\w\s-]", " ", question)
    tokens = [t for t in clean.split() if t.lower() not in STOPWORDS]
    return " ".join(tokens[:8]) if tokens else question


def _search_tavily(query, k=5):
    """
    Search general web using Tavily REST API. Returns list[str].
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": k,
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        passages = []
        for item in data.get("results", []):
            title = item.get("title", "").strip()
            content = item.get("content", "").strip()
            if content:
                text = f"[Web] {title}: {content}" if title else f"[Web] {content}"
                passages.append(_clean_passage(text))
        return passages
    except Exception as e:
        print(f"[Warning] Tavily search failed: {e}")
        return []


def _search_pubmed(query, k=5):
    """
    Search peer-reviewed literature via NCBI PubMed E-utilities. Returns list[str].
    """
    email = os.getenv("NCBI_EMAIL", DEFAULT_EMAIL).strip()
    api_key = os.getenv("NCBI_API_KEY", "").strip()

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": k,
        "tool": NCBI_TOOL,
        "email": email,
    }
    if api_key:
        search_params["api_key"] = api_key

    try:
        _rate_limit_pubmed()
        resp = requests.get(search_url, params=search_params, timeout=10)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": NCBI_TOOL,
            "email": email,
        }
        if api_key:
            fetch_params["api_key"] = api_key

        _rate_limit_pubmed()
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()

        root = ET.fromstring(fetch_resp.text)
        passages = []
        for article in root.findall(".//PubmedArticle"):
            title_node = article.find(".//ArticleTitle")
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""
            abstract_nodes = article.findall(".//AbstractText")
            abstract = " ".join("".join(n.itertext()).strip() for n in abstract_nodes).strip()

            if abstract:
                text = f"[PubMed] {title}: {abstract}" if title else f"[PubMed] {abstract}"
                passages.append(_clean_passage(text))
            elif title:
                passages.append(_clean_passage(f"[PubMed] {title}"))

        return passages
    except Exception as e:
        print(f"[Warning] PubMed search failed: {e}")
        return []


def web_search(query, k=5, use_cache=True):
    """
    Search both Tavily and PubMed databases and return up to k combined passages (list[str]).
    Passages from both sources are interleaved to ensure balanced coverage.
    Results are cached locally to save API quota and network calls.
    """
    if not query or not query.strip():
        return []

    query_key = query.strip().lower()
    if use_cache:
        cache = _load_cache()
        if query_key in cache and len(cache[query_key]) >= k:
            return cache[query_key][:k]

    # Fetch from both providers
    tavily_results = _search_tavily(query, k=k)
    pubmed_results = _search_pubmed(query, k=k)

    # Interleave results: [tavily_0, pubmed_0, tavily_1, pubmed_1, ...]
    combined = []
    max_len = max(len(tavily_results), len(pubmed_results))
    for i in range(max_len):
        if i < len(tavily_results):
            combined.append(tavily_results[i])
        if i < len(pubmed_results):
            combined.append(pubmed_results[i])

    if use_cache and combined:
        cache = _load_cache()
        cache[query_key] = combined
        _save_cache(cache)

    return combined[:k] if combined else []
