# 🏥 Medical CRAG QA System
### Production-Grade Corrective RAG · LangGraph · Gemini · ChromaDB

---

## 🏗️ Architecture Overview

```
Query → RETRIEVE → GRADE DOCS → [CONDITIONAL ROUTE]
                                   ├─ ALL RELEVANT   → GENERATE → HALLUCINATION CHECK → ✅ Answer
                                   ├─ SOME IRRELEVANT → TRANSFORM QUERY → RE-RETRIEVE  → GENERATE
                                   └─ ALL IRRELEVANT  → WEB SEARCH (Tavily) → GENERATE
```

## 📁 Project Structure

```
medical-crag/
├── medical_crag.ipynb    # Phase 1: Full Jupyter notebook
├── streamlit_app.py      # Phase 2: Streamlit frontend
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export LANGSMITH_API_KEY="your-langsmith-key"  # Optional
```

### 3. Run Phase 1 (Jupyter Notebook)
```bash
jupyter notebook medical_crag.ipynb
```

### 4. Run Phase 2 (Streamlit App)
```bash
streamlit run streamlit_app.py
```

---

## 🧠 CRAG Pipeline Nodes

| Node | Function | Routing |
|------|----------|---------|
| **RETRIEVE** | k-NN similarity search (ChromaDB, k=5) | Always first |
| **GRADE DOCUMENTS** | Gemini LLM: RELEVANT/IRRELEVANT/AMBIGUOUS per doc | Always after retrieve |
| **TRANSFORM QUERY** | Rewrite with clinical terminology (ICD codes, generics) | Mixed relevance |
| **WEB SEARCH** | Tavily API fallback | All irrelevant |
| **GENERATE** | Gemini grounded answer with source citations | After context ready |
| **HALLUCINATION CHECK** | Verify all claims against retrieved context | After generate |

---

## 📐 Evaluation Metrics

### Retrieval
- **Recall@3 / Recall@5**: `relevant_retrieved_K / total_relevant`
- **MRR**: `1 / rank_of_first_relevant_doc`

### Generation
- **Faithfulness**: `grounded_claims / total_claims` (LLM-as-judge)
- **Answer Relevance**: `cosine_similarity(embed(question), embed(answer))`
- **Hallucination Rate**: `hallucinated_answers / total_answers`

### Summary Table (expected after full eval)
```
| Metric           | Naive RAG | CRAG   | Improvement |
|------------------|-----------|--------|-------------|
| Recall@5         | ~70%      | ~85%   | +15%        |
| MRR              | 0.72      | 0.88   | +0.16       |
| Faithfulness     | ~65%      | ~82%   | +17%        |
| Hallucination    | ~25%      | ~8%    | -17%        |
| Answer Relevance | 0.74      | 0.86   | +0.12       |
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Google Gemini 1.5 Flash/Pro |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (local, persistent) |
| Web Fallback | Tavily Search API |
| Framework | LangChain |
| Observability | LangSmith |
| Frontend | Streamlit |
| Notebook | Jupyter |

---

## 📊 Evaluation Dataset

50 synthetic QA pairs covering:
- **Drug Dosage** (15 questions): metformin, lisinopril, warfarin, amoxicillin, atorvastatin...
- **Clinical Guidelines** (12 questions): sepsis bundles, HTN targets, diabetes management...
- **Symptom/Diagnosis** (12 questions): FAST acronym, AKI criteria, chest pain DDx...
- **Treatment Protocols** (11 questions): DVT treatment, STEMI management, pain protocols...

---

## 🔑 API Keys Required

1. **Google AI Studio** (free tier available): https://makersuite.google.com/app/apikey
2. **Tavily** (free tier: 1000 req/month): https://tavily.com
3. **LangSmith** (optional, for tracing): https://smith.langchain.com
