# Multilingual Classification of Urgent Disaster Response Messages from Social Media

**Course:** Natural Language Processing  
**Instructor:** Dr. Swapnil Hingmire  

**Team:**
- Anju Sasikumar (142301004)
- Kotha Adarsh Reddy (102301018)
- Muhamed Rizwan Mehaboob (142301026)

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

## Datasets

### 1. HumAID

The [HumAID](https://crisisnlp.qcri.org/humaid_dataset.html) dataset contains ~77K manually annotated disaster-related tweets from 17 major natural disaster events (2016–2019), including earthquakes, hurricanes, wildfires, and floods. Each tweet is labeled with one of 11 humanitarian categories. This is used as the primary dataset for training and evaluating all models.

### 2. Multilingual Disaster Response Messages

The [Disaster Response Messages](https://github.com/rmunro/disaster_response_messages) dataset by Robert Munro contains ~25K messages from the Haiti earthquake (2010), Pakistan floods (2010), and Hurricane Sandy (2012). Messages are labeled with 38 binary categories. This dataset includes original messages in Haitian Creole and Urdu alongside English translations. It is used in separate experiments to study the effect of adding cross-domain training data on model performance.

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

```
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb            # Data loading, label mapping, preprocessing, EDA
│   └── Baseline_SVM.ipynb                  # TF-IDF + LinearSVC
│
├── datasets/
│   ├── raw/                                # Original dataset files
│   │   ├── HumAID_data_events_set1_47K/    # HumAID set1 — 11 events (~47K tweets)
│   │   ├── HumAID_data_events_set2_29K/    # HumAID set2 — 6 events (~29K tweets)
│   │   └── disaster_response_messages/     # Munro dataset (~25K messages)
│   ├── processed/
│   │   └── humaid_processed.csv            # Output of Data_Preprocessing.ipynb
│   └── indian_language/                    # Manually annotated regional language dataset
│
├── results/
│   ├── plots/
│   │   ├── eda/                            # EDA visualizations
│   │   └── svm/                            # SVM baseline plots
│   ├── svm_results.json
│   └── svm_errors.csv
│
└── docs/
    ├── project_proposal.pdf
    └── weekly_reports/
```

> **Note:** This structure will grow as more models are added. New plot subfolders and result files will be added for each model (Naive Bayes, Logistic Regression, transformer models, etc.).

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
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Update the Base Path

Open `notebooks/Data_Preprocessing.ipynb` and update `BASE_DIR` in the configuration cell to point to your project directory:

```python
BASE_DIR = r"C:\path\to\your\disaster-response-classification"
```

Update the same `BASE_DIR` in all other notebooks as well.

### 5. Run the Notebooks

Run the notebooks **in order**. Each notebook depends on the output of the previous one.

| Order | Notebook | Description | GPU Required |
|---|---|---|---|
| 1 | `Data_Preprocessing.ipynb` | Loads raw data, maps labels, preprocesses text, generates EDA plots, saves `humaid_processed.csv` | No |
| 2 | `Baseline_SVM.ipynb` | TF-IDF + LinearSVC baseline | No |

More notebooks will be added as the project progresses (Naive Bayes, Logistic Regression, transformer models, cross-lingual evaluation, error analysis).

## Reproducing Results

```bash
# 1. Clone and set up environment
git clone https://github.com/Rizzwan285/Multilingual-Classification-of-Urgent-Disaster-Response-Messages.git
cd disaster-response-classification
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Update BASE_DIR in all notebooks

# 3. Run notebooks in order
jupyter notebook notebooks/
```

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
- **Disaster Response Messages:** [https://github.com/rmunro/disaster_response_messages](https://github.com/rmunro/disaster_response_messages)

## Citation

```bibtex
@inproceedings{alam2021humaid,
  title={HumAID: Human-Annotated Disaster Incidents Data from Twitter},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  year={2021}
}

@phdthesis{munro12dissertation,
  author={Robert Munro},
  title={Processing short message communications in low-resource languages},
  school={Stanford University},
  year={2012},
  url={https://purl.stanford.edu/cg721hb0673}
}
```

## License

This project is for academic purposes as part of the NLP course at IIT Palakkad. The HumAID dataset and Disaster Response Messages dataset are used under their respective licensing terms.

## Contact

For questions or issues, contact any team member or raise an issue on this repository.