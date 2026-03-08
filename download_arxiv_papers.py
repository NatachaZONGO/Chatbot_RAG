import arxiv
import os

os.makedirs("data", exist_ok=True)

search = arxiv.Search(
    query="machine learning OR retrieval augmented generation OR semantic search",
    max_results=30,
    sort_by=arxiv.SortCriterion.Relevance
)

for result in search.results():
    print("Téléchargement :", result.title)
    result.download_pdf(dirpath="data")