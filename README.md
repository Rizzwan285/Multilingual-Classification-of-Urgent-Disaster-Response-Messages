# Multilingual Classification of Urgent Disaster Response Messages from Social Media

**Course:** Natural Language Processing  
**Instructor:** Dr. Swapnil Hingmire  

**Team:**
- Muhamed Rizwan Mehaboob (142301026)
- Anju Sasikumar (142301004)
- Kotha Adarsh Reddy (102301018)

---

> **Project root:** Run every command from the repository root, which is also the folder shown as `.` in `path.txt`.
>
> **Virtual environment:** The recommended local environment name is `nlp_env`.
>
> **Generated folders:** `trained_models/`, `results/local*/`, and most of `offline_models/` are generated locally and are ignored by git.

---

## Introduction

Social media platforms like Twitter, Telegram, and WhatsApp become critical communication channels during natural disasters. Affected people post urgent appeals for rescue, food, medical aid, and shelter. However, the massive volume of messages — most of which are non-urgent or informational — overwhelms emergency response organizations.

Simple keyword-based filtering fails because semantic intent, not the presence of specific words, determines urgency. Additionally, disaster-related messages in India are frequently written in regional languages not adequately supported by existing systems.

This project builds a multilingual classification system that categorizes disaster-related social media messages into five actionable classes:

| Class | Description |
|---|---|
| **Critical Rescue** | People displaced, injured, dead, or missing — immediate response needed |
| **Resource Requests** | Requests for food, water, shelter, medical supplies, or other aid |
| **Volunteering and Donations** | Offers of help, donation drives, volunteer coordination |
| **Situational Awareness** | Infrastructure damage, weather updates, caution advisories, general info |
| **Irrelevant** | Sympathy messages, unrelated content, unclear/unjudgeable posts |

---

## Dataset

### HumAID

The [HumAID](https://crisisnlp.qcri.org/humaid_dataset.html) dataset contains about 77K manually annotated disaster-related tweets from 17 major natural disaster events (2016–2019), including earthquakes, hurricanes, wildfires, and floods. Each tweet is labeled with one of 11 humanitarian categories. This is used as the primary dataset for training and evaluating all models.

### Label Mapping (HumAID)

HumAID's 11 original humanitarian labels are mapped to our 5 target classes:

| HumAID Label | → Target Class |
|---|---|
| `displaced_people_and_evacuations`, `injured_or_dead_people`, `missing_or_found_people` | Critical Rescue |
| `requests_or_urgent_needs` | Resource Requests |
| `rescue_volunteering_or_donation_effort` | Volunteering and Donations |
| `caution_and_advice`, `infrastructure_and_utility_damage`, `other_relevant_information` | Situational Awareness |
| `sympathy_and_support`, `not_humanitarian`, `dont_know_cant_judge` | Irrelevant |

---

## Repository Structure

```text
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .env.example
├── path.txt
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── Baseline_SVM.ipynb
│   ├── Baseline_NaiveBayes.ipynb
│   ├── Baseline_Logistic_Regression.ipynb
│   └── Models_Evaluation.ipynb
│
├── src/
│   ├── download_base_transformers.py
│   ├── upload_model.py
│   └── ...
│
├── datasets/
│   ├── raw/
│   │   ├── HumAID_data_events_set1_47K/
│   │   ├── HumAID_data_events_set2_29K/
│   │   └── disaster_response_messages/
│   ├── processed/
│   │   └── humaid_processed.csv
│   └── indian_language/
│
├── offline_models/
│   └── ml_baselines/            # tracked exception inside offline_models/
│
├── trained_models/              # generated locally; ignored by git
│
├── results/
│   ├── plots/
│   ├── evaluation/
│   └── local*/                  # generated locally; ignored by git
│
└── docs/
    ├── project_proposal.pdf
    └── weekly_reports/
```

> **Note:** `trained_models/` stores fine-tuned checkpoints and is ignored by git. The same applies to most transformer run outputs under `results/local*/`. The only folder inside `offline_models/` intended to remain tracked is `offline_models/ml_baselines/`.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- CUDA-capable GPU recommended for transformer fine-tuning

### 1. Clone the Repository

```bash
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv nlp_env
nlp_env\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv nlp_env
source nlp_env/bin/activate
```

### 3. Install the Requirements

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Some scripts use API tokens. Do not commit real secrets to git.

### 4. Create a `.env` file

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and add your tokens:

```bash
HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Important `.env` rules

- Do **not** add spaces around `=`
- Do **not** put quotes around the values
- Do **not** commit `.env`
- Keep `.env.example` in the repo as a placeholder only

### 6. Hugging Face access for gated models

Some Hugging Face models are public, while others are gated.

For this project:

- `xlm-roberta-base` is public
- `google/muril-base-cased` is public
- `bert-base-multilingual-cased` is public
- `ai4bharat/indic-bert` is gated and requires access approval

Before downloading `ai4bharat/indic-bert`, log in and accept access on Hugging Face:

1. Open the model page on Hugging Face
2. Click **Agree and access** / **Accept terms**
3. Make sure your `HF_TOKEN` is added in `.env`

The download and evaluation scripts will use `HF_TOKEN` automatically when needed.

---

## Prepare the Dataset

The project expects the HumAID dataset to be available in `datasets/raw/`.  
If it is not already present, download it from the official HumAID page:

- https://crisisnlp.qcri.org/humaid_dataset.html

After downloading, place the extracted files inside the `datasets/raw/` directory.

---

## Run the Project Step by Step

### 1. Run Data Preprocessing

Open and run:

```bash
jupyter notebook notebooks/Data_Preprocessing.ipynb
```

This notebook loads the raw data, maps labels, preprocesses text, and saves the processed dataset.

---

### 2. Download the Base Transformer Models

Run:

```bash
python src/download_base_transformers.py
```

This downloads the base Hugging Face models into `offline_models/`.

It will:
- download public models directly
- use `HF_TOKEN` for gated models if needed
- skip models that are already present locally

---

### 3. Train the ML Baselines

Run the baseline notebooks from the same virtual environment:

- `notebooks/Baseline_SVM.ipynb`
- `notebooks/Baseline_NaiveBayes.ipynb`
- `notebooks/Baseline_Logistic_Regression.ipynb`

These do not require a GPU.

---

### 4. Train the Transformer Models

Run the transformer training notebook(s) from the same virtual environment.

A GPU is strongly recommended here. If you do not have a GPU, you can skip transformer training and still evaluate the ML baselines.

The trained checkpoints will be saved in `trained_models/`, and these folders are ignored by git.

---

### 5. Upload Trained Models to Hugging Face

After training, run:

```bash
python src/upload_model.py
```

This will upload the fine-tuned models to Hugging Face so that other users can run the project without retraining.

Important:
- `HF_TOKEN` must be present in `.env`
- the token must have **Write** access
- the Hugging Face repos must match the names used in the script

---

### 6. Evaluate the Models

Run:

```bash
jupyter notebook notebooks/Models_Evaluation.ipynb
```

This notebook:
- evaluates the ML baselines,
- evaluates the transformer models if the fine-tuned checkpoints are available,
- falls back to local offline transformer checkpoints when needed,
- and can download missing Hugging Face models if your environment has access.

The evaluation outputs, confusion matrices, and summary tables are saved inside `results/evaluation/`.

If a local transformer checkpoint is missing, the evaluation notebook can download the corresponding model from Hugging Face using the repo mapping inside the notebook.

---

## Reproducing Results from Scratch

```bash
# 1. Clone and set up environment
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
python3 -m venv nlp_env
source nlp_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Add HF_TOKEN and GROQ_API_KEY inside .env

# 4. Prepare dataset
# Download HumAID if needed and place it under datasets/raw/

# 5. Preprocess data
jupyter notebook notebooks/Data_Preprocessing.ipynb

# 6. Download base transformer models
python src/download_base_transformers.py

# 7. Train baselines
jupyter notebook notebooks/Baseline_SVM.ipynb
jupyter notebook notebooks/Baseline_NaiveBayes.ipynb
jupyter notebook notebooks/Baseline_Logistic_Regression.ipynb

# 8. Train transformer models on GPU if available
jupyter notebook notebooks/<transformer_training_notebook>.ipynb

# 9. Upload trained models to Hugging Face
python src/upload_model.py

# 10. Evaluate all available models
jupyter notebook notebooks/Models_Evaluation.ipynb
```

---

## Dependencies

Core dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
torch
transformers
huggingface_hub
joblib
tqdm
indic_transliteration
python-dotenv
```

Install them through `requirements.txt` whenever possible.

---

## Git Ignore Notes

The following paths are intentionally not tracked by git according to `.gitignore`:

- `trained_models/`
- `results/local*/`
- `offline_models/*` except `offline_models/ml_baselines/`
- `.env`
- `nlp_env/`, `.venv/`, `nlp_master/`
- `nlp_logs/`, `logs_*/`, `*.out`, `*.err`
- cache and IDE folders such as `__pycache__/`, `.ipynb_checkpoints/`, `.vscode/`, `.idea/`

These directories are created locally when you train or evaluate models and should not be committed.

---

## Evaluation Metrics

Given the high-stakes nature of disaster response, we prioritize **recall** for urgent categories (Critical Rescue and Resource Requests). A missed urgent message can result in delayed assistance and serious harm.

Metrics reported:
- Precision, Recall, and F1-score per class
- Macro and Weighted F1-score
- Confusion matrix analysis
- Dangerous false negative rate (urgent messages classified as non-urgent)

---

## Known Issues

- **Class imbalance:** Resource Requests is the smallest class. Balanced class weights are used to mitigate this.
- **Transformer training:** Fine-tuning is much slower on CPU and is best done on GPU.
- **IndicBERT access:** `ai4bharat/indic-bert` is gated on Hugging Face and requires accepted access before download.
- **HumAID tweet text:** Some tweets may have been deleted from Twitter since the dataset was created. The dataset provides the tweet text directly, so this does not affect the experiments.

---

## Dataset Sources

- **HumAID:** https://crisisnlp.qcri.org/humaid_dataset.html

---

## Citation

```bibtex
@inproceedings{alam2021humaid,
  title={HumAID: Human-Annotated Disaster Incidents Data from Twitter},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  year={2021}
}
```

---

## License

This project is for academic purposes as part of the NLP course at IIT Palakkad. The HumAID dataset is used under its respective licensing terms.

---

## Contact

For questions or issues, contact any team member or raise an issue on this repository.

- Muhamed Rizwan Mehaboob - 142301026@smail.iitpkd.ac.in
- Anju Sasikumar - 142301004@smail.iitpkd.ac.in
- Kotha Adarsh Reddy - 102301018@smail.iitpkd.ac.in