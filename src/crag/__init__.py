"""
CRAG (Corrective Retrieval-Augmented Generation) on PubMedQA.

Pipeline: retriever -> evaluator -> actions -> refinement / web_search -> generator.
See pipeline.run_crag for how the stages connect.
"""
