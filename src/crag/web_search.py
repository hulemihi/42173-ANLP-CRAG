"""
Web search fallback: rewrite the question into a search query and fetch passages from the web.

TODO: choose a search API (e.g. Tavily, SerpAPI, or PubMed E-utilities) and read its key from an env var.
"""


def rewrite_query(question):
    """
    Rewrite a question into a keyword search query (str).
    """
    raise NotImplementedError


def web_search(query, k=5):
    """
    Return up to k passages (list[str]) from the web for the query.
    """
    raise NotImplementedError
