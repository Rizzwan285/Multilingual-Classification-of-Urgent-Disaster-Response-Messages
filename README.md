# Multilingual Classification of Urgent Disaster Response Messages

A multilingual NLP pipeline that classifies disaster-related social media messages across English, Malayalam, and transliterated Manglish into five operational categories.

**Course:** Natural Language Processing
**Institution:** IIT Palakkad

**Team:**
- Muhamed Rizwan Mehaboob (142301026)
- Anju Sasikumar (142301004)
- Kotha Adarsh Reddy (102301018)

---

## What This Project Does

During natural disasters, social media is flooded with messages — most of them informational, but a small subset are urgent calls for rescue or supplies that emergency responders need to find quickly. Simple keyword filtering does not work, because words like *water* appear in pleas for help, news updates, and thank-you messages alike.

This project builds and evaluates a complete classification pipeline:

1. **Five target classes:** Critical Rescue, Resource Requests, Volunteering and Donations, Situational Awareness, Irrelevant.
2. **Three model families compared:** Traditional ML baselines (Naive Bayes, Logistic Regression, Linear SVM), four fine-tuned multilingual transformers (mBERT, XLM-RoBERTa, IndicBERT, MuRIL), and a prompted LLM classifier (Llama 3.3 70B via Groq).
3. **Cross-lingual evaluation:** All models trained only on English HumAID, then tested zero-shot on a custom Malayalam dataset and on a transliterated Manglish version generated from it at evaluation time.
4. **Class imbalance handling:** LLM-generated synthetic data for the minority *Resource Requests* class plus algorithmic class weighting, with an ablation that quantifies the synthetic data's contribution.

---

## Headline Results

| Setting | Best Model | Macro F1 |
|---|---|---|
| English HumAID test | Fine-Tuned XLM-RoBERTa | **0.778** |
| Native Malayalam (zero-shot) | Fine-Tuned MuRIL | **0.409** |
| Transliterated Manglish (zero-shot) | Fine-Tuned MuRIL | **0.188** |
| Best ML baseline (English) | Logistic Regression | 0.749 |
| Best LLM (English, one-shot, n=50) | Llama 3.3 70B | 0.62 (accuracy) |

Full methodology, tables, and analysis are in [`docs/NLP_Project.pdf`](docs/NLP_Project.pdf).

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .env.example                           # Template for API keys (do not commit .env)
│
├── datasets/
│   ├── raw/                               # HumAID TSVs (gitignored, see "Dataset Setup")
│   └── processed/
│       ├── humaid_processed.csv           # Output of preprocessing
│       ├── humaid_train_augmented.csv     # Real train + 2,500 synthetic Resource Requests
│       ├── synthetic_resource_requests.csv# LLM-generated minority-class data
│       └── dataset_malayalam.csv          # Custom 478-tweet Malayalam test set (pre-annotated)
│
├── docs/
│   ├── NLP_Project.pdf                    # Final project report
│   └── NLP_Project_Proposal.pdf
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb           # HumAID loading, label mapping, cleaning, EDA
│   ├── Training_ML_Baselines.ipynb        # All three ML baselines (with ablation)
│   ├── Training_Transformers.ipynb        # Fine-tunes the four multilingual transformers (GPU)
│   └── Models_Evaluation.ipynb            # Final evaluation across English + Malayalam + Manglish
│
├── src/
│   ├── download_base_transformers.py      # Downloads mBERT/XLM-R/MuRIL/IndicBERT to offline_models/
│   ├── generate_synthetic_data.py         # LLM-based synthesis for Resource Requests
│   ├── merge_synthetic_data.py            # Merges synthetic into augmented training CSV
│   ├── llm_classification.py              # Llama 3.3 70B zero-shot/few-shot evaluation
│   ├── tsne.py                            # Optional: t-SNE plot of MuRIL embeddings (for report)
│   └── upload_model.py                    # Optional: uploads fine-tuned models to Hugging Face
│
├── trained_models/
│   └── ml_baselines/                      # Trained .joblib baselines (committed)
│       ├── real_only/                     # Trained on HumAID train split alone
│       └── augmented/                     # Trained on humaid_train_augmented.csv
|
├── trained_models/                        
│
└── results/                               # Generated when notebooks/scripts run (see note below)
```

**About the empty `results/` folder.** Its contents are intentionally gitignored, since plots and CSVs change every run. When you run the notebooks and scripts, this folder is populated with metrics, plots, and intermediate outputs. An empty `results/` on a fresh clone is expected.

**Other gitignored paths:**
- `datasets/raw/` — HumAID raw data (download separately, see Setup → step 4)
- `offline_models/local_*` — downloaded transformer weights (too large for git)
- `trained_models/` — fine-tuned transformer checkpoints
- `results/` contents
- `nlp_env/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`

---

## Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A CUDA-capable GPU **is required** for `Training_Transformers.ipynb`. Everything else (preprocessing, ML baselines, LLM evaluation, final evaluation) runs on CPU.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
```

**Linux / macOS:**
```bash
python3 -m venv nlp_env
source nlp_env/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv nlp_env
nlp_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

Some scripts require external API keys. Copy the template and fill in your values:

**Linux / macOS:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Edit `.env` and add the keys you need:

```
HF_TOKEN=your_huggingface_token       # required for IndicBERT (gated) and uploading models
GROQ_API_KEY=your_groq_key            # required for synthetic data generation and LLM classification
```

**What each key is for:**
- `GROQ_API_KEY` is used by `generate_synthetic_data.py` (Llama 3.1 8B) and `llm_classification.py` (Llama 3.3 70B). Get one free at [console.groq.com](https://console.groq.com).
- `HF_TOKEN` is needed to download `ai4bharat/indic-bert` (a gated model on Hugging Face) and to upload fine-tuned models. You must also click "Agree and access" on the IndicBERT model page on Hugging Face before download will work.

### 4. Download HumAID dataset

The raw HumAID dataset is not committed to this repository due to licensing. Download it from the official source:

- **HumAID dataset:** https://crisisnlp.qcri.org/humaid_dataset.html

Download both `HumAID_data_events_set1_47K.tar.gz` and `HumAID_data_events_set2_29K.tar.gz`, then extract them under `datasets/raw/` so the structure looks like:

```
datasets/raw/
├── HumAID_data_events_set1_47K/events_set1/<event>/<event>_{train,dev,test}.tsv
└── HumAID_data_events_set2_29K/events_set2/<event>/<event>_{train,dev,test}.tsv
```

The processed CSV (`datasets/processed/humaid_processed.csv`) is included in the repo, so you only need the raw download if you want to re-run preprocessing from scratch.

The Malayalam test dataset (`datasets/processed/dataset_malayalam.csv`) is already committed — no download or annotation step is needed. The Manglish version is generated automatically inside `Models_Evaluation.ipynb` at evaluation time, so there is no separate Manglish file.

---

## Running the Pipeline

The pipeline is split into stages. Each stage produces an artifact that the next stage consumes. Many intermediate artifacts are already committed (the processed HumAID CSV, the augmented training CSV, the synthetic data, the trained baselines, the Malayalam test set), so you can skip ahead to evaluation if that's all you need.

The fastest path to results is just **Stage 5 (Download Transformers) → Stage 6 (Train Transformers) → Stage 8 (Evaluate)**.

### Stage 1 — Data Preprocessing

Loads all HumAID TSVs, maps the 11 original labels to our 5 target classes, cleans tweet text, runs exploratory analysis, and saves `datasets/processed/humaid_processed.csv`.

```bash
jupyter notebook notebooks/Data_Preprocessing.ipynb
```

Run all cells. Takes 1–2 minutes. **Skip this stage if `humaid_processed.csv` is already present** (it is, in this repo).

### Stage 2 — Generate Synthetic Resource Requests Data

Uses Llama 3.1 8B via Groq to generate ~2,500 synthetic tweets for the minority class. Requires `GROQ_API_KEY` in `.env`. Takes 30–60 minutes due to API rate limits.

```bash
python src/generate_synthetic_data.py
```

Output: `datasets/processed/synthetic_resource_requests.csv`. **Skip if this file is already present** (it is).

### Stage 3 — Merge Synthetic into Training Set

Combines real training data with synthetic rows, shuffles, and saves the augmented training CSV. Dev and test sets stay untouched.

```bash
python src/merge_synthetic_data.py
```

Output: `datasets/processed/humaid_train_augmented.csv`. **Skip if this file is already present** (it is).

### Stage 4 — Train ML Baselines

Trains Logistic Regression, Linear SVM, and Naive Bayes twice — once on real-only data and once on the augmented training set — to enable the synthetic-data ablation. CPU-only.

```bash
jupyter notebook notebooks/Training_ML_Baselines.ipynb
```

Outputs: trained models in `offline_models/ml_baselines/{real_only,augmented}/`. **These are already committed**, so you only need to re-run this stage if you want to verify reproducibility.

### Stage 5 — Download Base Transformer Models

Downloads pre-trained weights for mBERT, XLM-RoBERTa, MuRIL, and IndicBERT into `offline_models/`. Required before fine-tuning. Needs `HF_TOKEN` for IndicBERT.

```bash
python src/download_base_transformers.py
```

Skips models already present locally. The downloaded folders are gitignored due to size, so this is required on every fresh clone if you want to fine-tune transformers.

### Stage 6 — Fine-Tune Transformers (GPU required)

Fine-tunes all four transformers on the augmented training set. Each model takes 5–10 minutes on a modern GPU. CPU runs are impractically slow.

```bash
jupyter notebook notebooks/Training_Transformers.ipynb
```

Output: `trained_models/<model_name>/` (gitignored due to size). You must run this stage at least once before final evaluation can use the fine-tuned transformers.

### Stage 7 — LLM Classification (Optional)

Evaluates Llama 3.3 70B as a zero-shot and few-shot classifier on a 50-tweet stratified sample. Requires `GROQ_API_KEY`.

```bash
python src/llm_classification.py
```

Runs four experiments (different prompting strategies and seeds) and produces a comparison table in `results/llm_classification/`. Takes about 8 minutes due to rate limits.

### Stage 8 — Final Evaluation

Evaluates every model on the English test set, the native Malayalam test set, and a transliterated Manglish test set generated on the fly from the Malayalam data. Produces all comparison plots used in the report.

```bash
jupyter notebook notebooks/Models_Evaluation.ipynb
```

Outputs land under `results/`. This is the notebook that produces the headline numbers and figures.

### Optional: t-SNE Visualization

To regenerate the t-SNE embedding plot from the report:

```bash
python src/tsne.py
```

This requires fine-tuned MuRIL (Stage 6). Output goes to `results/`.

### Optional: Upload Fine-Tuned Models to Hugging Face

To share trained transformer weights:

```bash
python src/upload_model.py
```

Requires `HF_TOKEN` with write access. Repository names are configured inside the script.

---

## Recommended Path for a Reviewer

If you've cloned this repo to verify the work, the fastest reproduction path is:

```bash
# 1. Setup (one time)
python3 -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add HF_TOKEN to .env (needed for IndicBERT)

# 2. Download base transformer weights (needs HF_TOKEN)
python src/download_base_transformers.py

# 3. Fine-tune the four transformers (GPU required, ~30-40 min total)
jupyter notebook notebooks/Training_Transformers.ipynb

# 4. Run final evaluation across all three languages
jupyter notebook notebooks/Models_Evaluation.ipynb
```

This produces all numbers and plots used in the report, using committed intermediate data (preprocessed HumAID, augmented training set, Malayalam test set, trained ML baselines).

To reproduce **everything** from scratch (including preprocessing and synthetic data generation), also run Stages 1–4 first, in order, before Stage 5.

---

## Methodology Notes

A few decisions worth flagging for anyone reading the code or report:

**Label mapping correction.** An earlier version of this pipeline mapped HumAID's `sympathy_and_support` to *Situational Awareness*. We later reclassified it to *Irrelevant* because sympathy/prayer messages are not directly actionable. This shifted around 8,500 tweets and made the *Situational Awareness* class cleaner. All committed results use the corrected mapping.

**Synthetic data generation safeguards.** To avoid the common pitfalls of LLM-based augmentation: seed examples rotate per call (5 fresh examples each time), seeds come from the training split only (no dev or test leakage), the prompt explicitly requests social-media style with typos and hashtags, and a near-duplicate filter removes generations that closely repeat earlier ones.

**Ablation methodology.** The augmented training CSV carries an `is_synthetic` flag, which lets us train both with and without synthetic data using the same code path. The result: synthetic data improved Resource Requests F1 by 0.05–0.06 across baselines but the macro-F1 gain was modest (0.01–0.013).

**Why LLM evaluation is English-only.** Llama 3.1/3.3 has weak Malayalam capability. Including it in cross-lingual tests would have conflated model capability with language support, producing misleading numbers.

**Class weights are applied even with synthetic data.** Both methods are used together. The ablation confirms each contributes; they are not redundant.

**Manglish generation.** We do not commit a separate Manglish CSV. The Manglish test set is generated from `dataset_malayalam.csv` inside `Models_Evaluation.ipynb` using transliteration, so there is exactly one source of Malayalam ground truth and no risk of label drift between native and transliterated versions.

---

## Limitations

- **Synthetic data is not a substitute for real data.** Improvements from LLM-generated tweets are real but small.
- **Single-label assumption.** Real tweets often plausibly belong to multiple classes; we force one.
- **Cross-lingual test set is small** (478 manually annotated Malayalam tweets), limiting statistical precision.
- **LLM classifier evaluation uses only 50 tweets** due to API cost, so prompting-strategy comparisons are suggestive rather than definitive.
- **Resource Requests remains hardest.** Even after augmentation and class weighting, this minority class shows the lowest F1 on every model.

---

## Dependencies

Core libraries used in this project:

```
pandas, numpy
matplotlib, seaborn
scikit-learn
nltk
torch, transformers, huggingface_hub, datasets
joblib, tqdm
groq
indic_transliteration
python-dotenv
```

Always install via `pip install -r requirements.txt` to get the pinned versions.

---

## Citation

```bibtex
@inproceedings{alam2021humaid,
  title={HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks},
  author={Alam, Firoj and Qazi, Umair and Imran, Muhammad and Ofli, Ferda},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)},
  volume={15},
  number={1},
  pages={933--942},
  year={2021}
}
```

Full reference list is in the report ([`docs/NLP_Project.pdf`](docs/NLP_Project.pdf)).

---

## License

This project was developed for academic purposes as part of the NLP course at IIT Palakkad. The HumAID dataset is used under its original licensing terms; please consult [crisisnlp.qcri.org](https://crisisnlp.qcri.org/humaid_dataset.html) for redistribution policy.

---

## Contact

- Muhamed Rizwan Mehaboob — 142301026@smail.iitpkd.ac.in
- Anju Sasikumar — 142301004@smail.iitpkd.ac.in
- Kotha Adarsh Reddy — 102301018@smail.iitpkd.ac.in

For issues with reproducing the results, please open a GitHub issue.