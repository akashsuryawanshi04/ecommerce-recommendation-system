# Quick Start Guide

This guide gets you from zero to running recommendations in under 10 minutes.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.12 | 3.11 recommended |
| pip | ≥ 23.0 | `pip install --upgrade pip` |
| RAM | ≥ 4 GB | 8 GB recommended for ALS |
| OS | Linux / macOS / Windows | WSL2 recommended on Windows |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<username>/ecommerce-recommendation-system.git
cd ecommerce-recommendation-system
```

---

## Step 2 — Create a Virtual Environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `implicit` (ALS model) may take 1–2 minutes to install.
> If GPU acceleration is desired, install the CUDA build:
> `pip install implicit[gpu]`

---

## Step 4 — Generate the Dataset

```bash
python data/generate_dataset.py
```

Expected output:
```
✅ Dataset saved → data/raw/ecommerce_data.csv
   Rows       : 25,001
   Customers  : 1,000
   Products   : 30
   Date range : 2023-01-01 → 2023-12-31
```

> **Using your own data?** Replace this step by placing your CSV in
> `data/raw/ecommerce_data.csv` with columns:
> `TransactionID, CustomerID, ProductID, ProductName, Quantity, UnitPrice, Timestamp, Country`

---

## Step 5 — Preprocess the Data

```bash
python src/preprocessing/data_processor.py
```

Expected output:
```
✅ Preprocessing complete.
   Clean rows      : 23,816
   Unique customers: 999
   Unique products : 30
   Matrix density  : 52.93%
```

Artefacts saved to `data/processed/`:
```
data/processed/
├── clean_transactions.csv
├── interaction_matrix.npz   ← sparse CSR user-item matrix
├── customer_encoder.npy     ← CustomerID → integer index
└── product_encoder.npy      ← ProductID  → integer index
```

---

## Step 6 — Train All Models

```bash
python src/train.py
```

This single command trains all four models and prints evaluation metrics:

```
════════════════════════════════════════════════════════════════
   🛒  E-Commerce Recommendation System — Training Pipeline
════════════════════════════════════════════════════════════════
[INFO] Training UserUserCF ...
[INFO] Training ItemItemCF ...
[INFO] Training ALS ...
[INFO] Training Association Rules ...
[INFO] Running evaluation ...

  📊  ALS Evaluation Report
  Precision@10 : 0.1810
  NDCG@10      : 0.2470
  Hit Rate@10  : 0.4910
  MRR          : 0.2110

✅  Training complete. Models saved to models/saved/
════════════════════════════════════════════════════════════════
```

Saved models:
```
models/saved/
├── uu_cf.pkl              ← User-User CF
├── ii_cf.pkl              ← Item-Item CF
├── als_model.pkl          ← ALS Matrix Factorisation
└── association_rules.pkl  ← Apriori Rules
```

---

## Step 7 — Start the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API starts at **http://localhost:8000**

Interactive Swagger docs: **http://localhost:8000/docs**

Test the API:
```bash
# Get personalised recommendations for user C0042
curl "http://localhost:8000/recommend/user/C0042?n=5&method=hybrid"

# Get products similar to P004 (Mechanical Keyboard)
curl "http://localhost:8000/recommend/product/P004?n=5"

# Get "Customers Also Bought" for P004
curl "http://localhost:8000/recommend/also-bought/P004?n=5"
```

---

## Step 8 — Launch the Streamlit UI

Open a **new terminal** (keep the API running in the first one):

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at **http://localhost:8501**

UI features:
- **Home** — live metrics, architecture overview, quick demo
- **For You** — personalised recommendations by user ID and algorithm
- **Product Page** — product details + "You May Also Like" + "Also Bought"
- **Model Insights** — benchmark table and metric explanations
- **About** — full technical documentation

---

## Step 9 — Run the Test Suite (Optional)

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Step 10 — Run EDA (Optional)

```bash
python notebooks/01_exploratory_data_analysis.py
```

Charts are saved to `notebooks/figures/`:
- Top 15 best-selling products
- Customer purchase frequency distribution
- Product popularity (long-tail) curve
- Monthly revenue trend
- Basket size distribution
- Revenue by country
- Interaction matrix sparsity heatmap

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: No module named 'implicit'` | `pip install implicit` |
| `ModuleNotFoundError: No module named 'mlxtend'` | `pip install mlxtend` |
| API returns 404 for a user | The user ID may not exist in the training data. Use `/customers` endpoint to list valid IDs. |
| `FileNotFoundError` for model `.pkl` | Run `python src/train.py` first. |
| Streamlit can't connect to API | Ensure the API is running (`uvicorn api.main:app --port 8000`) before starting Streamlit. |
| Slow ALS training | Reduce `factors` or `iterations` in `configs/config.yaml`. |
