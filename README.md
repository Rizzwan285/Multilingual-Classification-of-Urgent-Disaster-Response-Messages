# Multilingual Classification of Urgent Disaster Response Messages from Social Media

**Course:** Natural Language Processing  
**Instructor:** Dr. Swapnil Hingmire  

**Team:**
- Anju Sasikumar (142301004)
- Kotha Adarsh Reddy (102301018)
- Muhamed Rizwan Mehaboob (142301026)

---

> **Project root:** The repository is meant to be used from the project root (the folder shown as `.` in `path.txt`).
> **Virtual environment:** The recommended local environment name is `nlp_env`.
> **Ignored/generated folders:** `trained_models/`, `results/local*/`, and almost all of `offline_models/` are treated as local/generated artifacts and are excluded by `.gitignore`.

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

## Dataset

### HumAID

The [HumAID](https://crisisnlp.qcri.org/humaid_dataset.html) dataset contains ~77K manually annotated disaster-related tweets from 17 major natural disaster events (2016–2019), including earthquakes, hurricanes, wildfires, and floods. Each tweet is labeled with one of 11 humanitarian categories. This is used as the primary dataset for training and evaluating all models.

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
│   └── Models_Evaluation.md / .ipynb
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


> **Note:** `trained_models/` is where fine-tuned checkpoints are saved, but it is ignored by git. The same applies to most transformer run outputs under `results/local*/`. The only folder inside `offline_models/` intended to remain tracked is `offline_models/ml_baselines/`.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- (Optional) NVIDIA GPU with CUDA for transformer fine-tuning (needed later)

### 1. Clone the Repository

```bash
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
```

> **Note:** The raw datasets are already included in this repository under `datasets/raw/`. You don't need to download them separately after cloning.

### 2. Create a Virtual Environment (Recommended)

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

### 3. Update the Base Path

Open `notebooks/Data_Preprocessing.ipynb` and update `BASE_DIR` in the configuration cell so it matches the project root logic used in the notebooks:

```python
from pathlib import Path
BASE_DIR = Path.cwd().parent if Path.cwd().name in ("src", "notebooks") else Path.cwd()
```

Update the same `BASE_DIR` logic in all other notebooks as well.


### 4. Run the Notebooks

Run the notebooks **in order**. Each notebook depends on the output of the previous one.

| Order | Notebook | Description | GPU Required |
|---|---|---|---|
| 1 | `Data_Preprocessing.ipynb` | Loads raw data, maps labels, preprocesses text, generates EDA plots, saves `humaid_processed.csv` | No |
| 2 | `Baseline_SVM.ipynb` | TF-IDF + LinearSVC baseline | No |


## Reproducing Results

```bash
# 1. Clone and set up environment
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd Multilingual-Classification-of-Urgent-Disaster-Response-Messages
python -m venv nlp_env
nlp_env\Scripts\activate          # Windows
# source nlp_env/bin/activate      # Linux / macOS

# 2. Update BASE_DIR in all notebooks

# 3. Run notebooks in order
jupyter notebook notebooks/

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
nltk>=3.8.0
```

Transformer-related dependencies (`transformers`, `torch`) will be needed for later notebooks.

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

- **Class imbalance:** Resource Requests is the smallest class (~3.4% of data). All models use balanced class weights to mitigate this.
- **HumAID tweet text:** Some tweets may have been deleted from Twitter since the dataset was created. The dataset provides the tweet text directly, so this does not affect our experiments.

## Dataset Sources

If you want to download the datasets independently:
- **HumAID:** [https://crisisnlp.qcri.org/humaid_dataset.html](https://crisisnlp.qcri.org/humaid_dataset.html)

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

This project is for academic purposes as part of the NLP course at IIT Palakkad. The HumAID dataset is used under their respective licensing terms.

## Contact

For questions or issues, contact any team member or raise an issue on this repository.

- Muhamed Rizwan Mehaboob - 142301026@smail.iitpkd.ac.in
- Anju Sasikumar - 142301004@smail.iitpkd.ac.in
- Kotha Adarsh Reddy - 102301018@smail.iitpkd.ac.in