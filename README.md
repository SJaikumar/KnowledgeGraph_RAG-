# Tamil Nadu Government Schemes – RAG + Knowledge Graph Starter

A local Python project for crawling official scheme-detail links, extracting structured information, persisting a Neo4j knowledge graph, indexing page excerpts in FAISS, and answering English/Tamil questions in Streamlit with citations.

**Status:** This is a starter implementation. The source pages have not been ingested here. It cannot answer scheme questions until successful extraction and indexing. In particular, if government URLs redirect or time out, this code records errors and will not fabricate content. The chatbot currently retrieves FAISS + BM25 passages using RRF; Neo4j graph facts are also retrieved for schemes shortlisted by the passage retriever. This is **graph-enhanced RAG**, not yet a full multi-hop graph-planning agent.

## Setup (Windows PowerShell)

```powershell
cd tn_schemes_graphrag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Insert real `OPENAI_API_KEY`, Neo4j connection URI, username and password in `.env`. Do not commit this file.

## Run pipeline

```powershell
python pipeline.py crawl
python pipeline.py extract
python pipeline.py build
streamlit run app.py
```

Inspect `data/crawl.json` **before** running `extract`: only `status=ok` pages are extracted. A redirected/blocked page is not counted as valid source evidence. The crawler is rate-limited and only contacts URLs from `schemes.csv`. Stop and use authorized exported pages if website access restrictions apply.

- `schemes.csv`: user-provided seed list and source URLs.
- `data/crawl.json`: fetched URLs, redirect status and extracted raw text.
- `data/schemes.json`: LLM-extracted facts with source provenance.
- `data/chunks.json` and `data/faiss.index`: retrieved passages and local persistent vector index.
- Neo4j: `Scheme`, `Department`, `Beneficiary`, `Benefit`, `Eligibility`, `Document`, `ApplicationStep`, and `Chunk` nodes and edges.

## Example queries

- What training programmes are listed for farmers?
- What does the source say about the soil health card scheme?
- What subsidies are mentioned for MSMEs?
- விவசாயிகளுக்கு என்ன பயிற்சி திட்டங்கள் உள்ளன?

## Known limitations / next steps

1. Access to legacy `scheme_details.php` links may fail or redirect. Prefer current official pages or authorized PDF downloads when that happens.
2. Crawling checks page/title consistency heuristically; manually audit extracted samples. No OCR is implemented for scanned PDFs.
3. The extraction schema is LLM-generated and requires human quality checking, especially for eligibility and active/inactive programme status.
4. No active-scheme inference: all facts reflect only fetched source material, not a live eligibility verdict.
5. Improve by adding multi-hop graph query planning, entity linking, change tracking, better multilingual keyword segmentation, and a dataset of graded answers.
6. Do not store private applicants' identity documents or sensitive personal data in Neo4j without appropriate controls.
