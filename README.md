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

## Dataset

We use the [HumAID](https://crisisnlp.qcri.org/humaid_dataset.html) dataset — a collection of ~77K manually annotated disaster-related tweets from 17 major natural disaster events (2016–2019), including earthquakes, hurricanes, wildfires, and floods.

Additionally, a small manually annotated dataset of disaster messages in an Indian regional language is used for cross-lingual evaluation.

### Label Mapping

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
│   ├── Data_Preprocessing.ipynb          # Data loading, label mapping, preprocessing, EDA
│   ├── Baseline_SVM.ipynb                # TF-IDF + LinearSVC
│   ├── Baseline_Naive_Bayes.ipynb        # TF-IDF + Multinomial Naive Bayes
│   ├── Baseline_Logistic_Regression.ipynb# TF-IDF + Logistic Regression
│   ├── XLM_RoBERTa.ipynb                # Fine-tuning XLM-RoBERTa
│   ├── Cross_Lingual_Evaluation.ipynb    # Zero-shot on Indian language data
│   └── Error_Analysis.ipynb              # Detailed error analysis across models
│
├── data/
│   ├── raw/                              # Original HumAID TSV files (not tracked in git)
│   │   ├── events_set1/                  # 11 events (~47K tweets)
│   │   └── events_set2/                  # 6 events (~29K tweets)
│   ├── processed/
│   │   └── humaid_processed.csv          # Output of Data_Preprocessing.ipynb
│   └── indian_language/                  # Manually annotated regional language dataset
│       └── annotation_guidelines.md
│
├── results/
│   ├── plots/
│   │   ├── eda/
│   │   ├── svm/
│   │   ├── naive_bayes/
│   │   ├── logistic_regression/
│   │   ├── xlm_roberta/
│   │   ├── cross_lingual/
│   │   └── error_analysis/
│   ├── svm_results.json
│   ├── nb_results.json
│   ├── lr_results.json
│   ├── xlmr_results.json
│   └── model_comparison.csv
│
└── docs/
    ├── project_proposal.pdf
    ├── weekly_reports/
    └── final_report.pdf
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/disaster-response-classification.git
cd disaster-response-classification
```

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For transformer notebooks (XLM-RoBERTa), install PyTorch separately based on your system:
- **With GPU (CUDA):** Visit [https://pytorch.org](https://pytorch.org) and select your CUDA version
- **CPU only:** `pip install torch torchvision`

### 4. Download the HumAID Dataset

1. Go to [https://crisisnlp.qcri.org/humaid_dataset.html](https://crisisnlp.qcri.org/humaid_dataset.html)
2. Download both event sets:
   - `HumAID_data_events_set1_47K` (11 events)
   - `HumAID_data_events_set2_29K` (6 events)
3. Extract and place them in the `data/raw/` directory:

```
data/raw/
├── events_set1/
│   ├── canada_wildfires_2016/
│   │   ├── canada_wildfires_2016_train.tsv
│   │   ├── canada_wildfires_2016_dev.tsv
│   │   └── canada_wildfires_2016_test.tsv
│   ├── cyclone_idai_2019/
│   │   └── ...
│   └── ... (11 events)
└── events_set2/
    ├── california_wildfires_2018/
    │   └── ...
    └── ... (6 events)
```

### 5. Update the Base Path

Open `notebooks/Data_Preprocessing.ipynb` and update `BASE_DIR` in the configuration cell to point to your project directory:

```python
BASE_DIR = r"C:\path\to\your\disaster-response-classification"
```

Update the same `BASE_DIR` in all other notebooks as well.

### 6. Run the Notebooks

Run the notebooks **in order**. Each notebook depends on the output of the previous one.

| Order | Notebook | Description | GPU Required |
|---|---|---|---|
| 1 | `Data_Preprocessing.ipynb` | Loads raw data, maps labels, preprocesses text, generates EDA plots, saves `humaid_processed.csv` | No |
| 2 | `Baseline_SVM.ipynb` | TF-IDF + LinearSVC baseline | No |

**Note:** Notebooks 2–4 (baselines) can be run in any order after notebook 1. Notebook 5 must be run before notebook 6. Notebook 7 should be run last after all models have been trained.

## Reproducing Results

To reproduce all results reported in the project report:

```bash
# 1. Set up environment
git clone https://github.com/<your-username>/disaster-response-classification.git
cd disaster-response-classification
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Download HumAID data into data/raw/ (see step 4 above)

# 3. Update BASE_DIR in all notebooks

# 4. Run notebooks in order (1 → 7)
#    Open each notebook in Jupyter and run all cells
jupyter notebook notebooks/
```



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

## Citation

If you use this work, please cite the HumAID dataset:

```bibtex
@inproceedings{alam2021humaid,
  title={HumAID: Human-Annotated Disaster Incidents Data from Twitter},
  author={Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  year={2021}
}
```

## License

This project is for academic purposes as part of the NLP course at TIET. The HumAID dataset is used under its original licensing terms.

## Contact

For questions or issues, contact any team member or raise an issue on this repository.