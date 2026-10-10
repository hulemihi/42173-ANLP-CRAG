"""
Retriever: return the top-k passages for a question from a corpus (e.g. all PubMedQA abstracts).

TODO: implement with BM25 (rank_bm25) or a dense retriever.
"""


def build_index(corpus):
    """
    Build a search index over a list of passage strings. Returns the index object.
    """
    raise NotImplementedError


def retrieve(index, question, k=5):
    """
    Return the top-k passages (list[str]) for the question, most relevant first.
    """
    raise NotImplementedError
