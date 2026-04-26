# Multilingual Classification of Urgent Disaster Response Messages from Social Media

**Course:** Natural Language Processing  
**Instructor:** Dr. Swapnil Hingmire  

**Team:**
- Muhamed Rizwan Mehaboob (142301026)
- Anju Sasikumar (142301004)
- Kotha Adarsh Reddy (102301018)

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

## Repository Structure

```text
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── path.txt
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── Baseline_SVM.ipynb
│   ├── Baseline_NaiveBayes.ipynb
│   ├── Baseline_Logistic_Regression.ipynb
│   └── Models_Evaluation.ipynb / .md
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
│   └── ml_baselines/            # only tracked exception inside offline_models/
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
> **Virtual environment:** The recommended local environment name is `nlp_env`.
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

### 4. Prepare the Dataset

The project expects the HumAID dataset to be available in `datasets/raw/`.  
If it is not already present, download it from the official HumAID page:

- https://crisisnlp.qcri.org/humaid_dataset.html

After downloading, place the extracted files inside the `datasets/raw/` directory.

### 5. Hugging Face Login for Model Downloads

The base transformer download script can fetch public models directly. If a model repository is gated or private, authenticate first.

Recommended setup:

1. Create a Hugging Face account.
2. Open your Hugging Face settings and create a **User Access Token** with **read** access.
3. Log in once on your machine:

```bash
huggingface-cli login
```

Paste the token when prompted.

You can also log in from Python notebooks with:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Once logged in, the token is stored locally and can be reused by the download and evaluation scripts.

### 6. Run Data Preprocessing

Run:

```bash
jupyter notebook notebooks/Data_Preprocessing.ipynb
```

This notebook loads the raw data, maps labels, preprocesses text, and saves the processed dataset.

### 7. Download the Base Transformer Models

Run the download script:

```bash
python download_base_transformers.py
```

This downloads the base Hugging Face models into `offline_models/`.  
If any model is gated or private, make sure you are logged in to Hugging Face before running this step.

### 8. Train the ML Baselines

Run the baseline notebooks from the same virtual environment:

- `notebooks/Baseline_SVM.ipynb`
- `notebooks/Baseline_NaiveBayes.ipynb`
- `notebooks/Baseline_Logistic_Regression.ipynb`

These do not require a GPU.

### 9. Train the Transformer Models

Run the transformer training notebook(s) from the same virtual environment.

A GPU is strongly recommended here. If you do not have a GPU, you can skip transformer training and still evaluate the ML baselines.

The trained checkpoints will be saved in `trained_models/`, and these folders are ignored by git.

### 10. Evaluate the Models

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

## Reproducing Results

```bash
# 1. Clone and set up environment
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
python3 -m venv nlp_env
source nlp_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
# Download HumAID if needed and place it under datasets/raw/

# 4. Preprocess data
jupyter notebook notebooks/Data_Preprocessing.ipynb

# 5. Download base transformer models
python download_base_transformers.py

# 6. Train baselines
jupyter notebook notebooks/Baseline_SVM.ipynb
jupyter notebook notebooks/Baseline_NaiveBayes.ipynb
jupyter notebook notebooks/Baseline_Logistic_Regression.ipynb

# 7. Train transformer models on GPU if available
jupyter notebook notebooks/<transformer_training_notebook>.ipynb

# 8. Evaluate all available models
jupyter notebook notebooks/Models_Evaluation.ipynb
```

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
```

Install them through `requirements.txt` whenever possible.

## Git Ignore Notes

The following paths are intentionally not tracked by git according to `.gitignore`:

- `trained_models/`
- `results/local*/`
- `offline_models/*` except `offline_models/ml_baselines/`
- `nlp_env/`, `.venv/`, `nlp_master/`
- `nlp_logs/`, `logs_*/`, `*.out`, `*.err`
- cache and IDE folders such as `__pycache__/`, `.ipynb_checkpoints/`, `.vscode/`, `.idea/`

These directories are created locally when you train or evaluate models and should not be committed.

## Evaluation Metrics

Given the high-stakes nature of disaster response, we prioritize **recall** for urgent categories (Critical Rescue and Resource Requests). A missed urgent message can result in delayed assistance and serious harm.

Metrics reported:
- Precision, Recall, and F1-score per class
- Macro and Weighted F1-score
- Confusion matrix analysis
- Dangerous false negative rate (urgent messages classified as non-urgent)

## Known Issues

- **Class imbalance:** Resource Requests is the smallest class. Balanced class weights are used to mitigate this.
- **Transformer training:** Fine-tuning is much slower on CPU and is best done on GPU.
- **HumAID tweet text:** Some tweets may have been deleted from Twitter since the dataset was created. The dataset provides the tweet text directly, so this does not affect the experiments.

## Dataset Sources

- **HumAID:** https://crisisnlp.qcri.org/humaid_dataset.html

## Citation

```bibtex
@inproceedings{alam2021humaid,
  title={HumAID: Human-Annotated Disaster Incidents Data from Twitter},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  year={2021}
}
```

## License

This project is for academic purposes as part of the NLP course at IIT Palakkad. The HumAID dataset is used under its respective licensing terms.

## Contact

For questions or issues, contact any team member or raise an issue on this repository.

- Muhamed Rizwan Mehaboob - 142301026@smail.iitpkd.ac.in
- Anju Sasikumar - 142301004@smail.iitpkd.ac.in
- Kotha Adarsh Reddy - 102301018@smail.iitpkd.ac.in
