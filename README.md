# SENTIMENT-ANALYSIS#
Sentiment Analysis: TF-IDF Baseline vs Fine-Tuned DistilBERT

A 3-class sentiment classification project using the Twitter Entity Sentiment Analysis dataset. The project compares a classical machine learning baseline using TF-IDF + Logistic Regression with a fine-tuned DistilBERT transformer model.

The project covers data cleaning, exploratory data analysis, leakage checking, baseline modeling, transformer fine-tuning, Optuna hyperparameter optimization, evaluation, error analysis, and interactive local inference using Streamlit.

---

## Project Overview

The goal is to predict the sentiment expressed toward a specific entity in a tweet:

- **Negative**
- **Neutral**
- **Positive**

The project focuses on understanding the complete machine learning workflow rather than only training a model.

### Pipeline

```text
Twitter Entity Sentiment Dataset
            ↓
     Data Cleaning
            ↓
          EDA
            ↓
 Stratified Train/Val/Test Split
       + Leakage Check
            ↓
 ┌───────────────────────┐
 │ TF-IDF + Logistic     │
 │ Regression Baseline   │
 └───────────────────────┘
            ↓
 ┌───────────────────────┐
 │ Fine-Tuned DistilBERT │
 └───────────────────────┘
            ↓
   Optuna Hyperparameter
       Optimization
            ↓
       Model Evaluation
            ↓
       Error Analysis
            ↓
         Inference
            ↓
   Streamlit Local Demo


Dataset

This project uses the Twitter Entity Sentiment Analysis dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

The original dataset contains four sentiment labels:

Positive
Negative
Neutral
Irrelevant

For this project, Irrelevant is mapped to Neutral, resulting in three final classes.

The raw dataset contains:

tweet_id
entity
sentiment
text
Data Preprocessing

The preprocessing pipeline:

Removes rows with missing text or labels
Normalizes whitespace
Removes invalid/unmapped labels
Removes exact duplicate tweet texts
Removes pathologically long rows
Performs a stratified validation/test split
Checks for text overlap between train, validation, and test sets

The leakage check ensures that identical tweet texts do not appear across multiple splits.

Exploratory Data Analysis

The EDA examines:

Class distribution
Entity distribution
Tweet length
Sentiment distribution
Example tweets from each sentiment class
Potential data quality issues

The EDA notebook is available at:

notebooks/01_eda.ipynb
Models
1. TF-IDF + Logistic Regression

A classical machine learning baseline was implemented using:

TF-IDF vectorization
Unigrams and bigrams
Logistic Regression

This provides a computationally inexpensive baseline for comparison with the transformer model.

2. Fine-Tuned DistilBERT

The main model uses:

distilbert-base-uncased

DistilBERT was selected because it provides strong language understanding while being smaller and faster than full BERT.

The model was fine-tuned specifically for 3-class sentiment classification.

Training includes:

Hugging Face Transformers
Tokenization
Early stopping
Validation-based model selection
Mixed precision training
Fixed random seed for reproducibility
Hyperparameter Optimization

Optuna was used to systematically search for better transformer hyperparameters.

The optimization considers parameters such as:

Learning rate
Weight decay
Warmup ratio
Batch size
Number of epochs

The optimization objective is validation Macro F1, which provides a balanced evaluation across the three sentiment classes.

Results

The final DistilBERT model achieved:

Metric	Score
Accuracy	97.20%
Macro F1	97.20%
Weighted F1	97.21%
Classification Report
  Class	      Precision	Recall	F1
Negative	    98.50%	98.50%	98.50%
Neutral	      98.65%	96.07%	97.35%
Positive	    93.75%	97.83%	95.74%
Macro Avg	    96.97%	97.46%	97.20%
Confusion Matrix
                Predicted
              Neg  Neu  Pos
Actual Neg    131   0    2
Actual Neu      2  220    7
Actual Pos      0    3  135

The model performs strongly across all three classes, with most errors occurring between Neutral and Positive.

Error Analysis

The project includes dedicated error analysis to investigate where the model fails rather than relying only on aggregate metrics.

The analysis examines:

False positives
False negatives
Low-confidence predictions
Baseline vs transformer disagreements
Ambiguous tweets
Short tweets
Negation
Slang
Sarcasm
Domain-specific language

This helps identify limitations of the model and potential areas for improvement.

Inference

The trained DistilBERT model can be used to classify new text.

Example:

Input:
"I absolutely love this product! It is amazing and fantastic!"


Prediction:
Positive

The inference pipeline also returns confidence scores for each class:

Negative: 0.01%
Neutral: 0.02%
Positive: 99.97%

The inference implementation is available in:

src/inference/predict.py
Streamlit Local Demo

A Streamlit interface was created to interactively test the trained sentiment model.

The application allows users to:

Enter a tweet or text
Get the predicted sentiment
View the probability for each sentiment class
Test multiple examples interactively

The Streamlit application was tested locally with the trained DistilBERT model and successfully produced predictions.

Run the application

Install the dependencies:

pip install -r requirements.txt

Then start Streamlit:

streamlit run app.py

Open the local URL provided by Streamlit:

http://localhost:8501
Example
Input:
"I absolutely love this product! It is amazing and fantastic!"


Prediction:
Positive


Confidence:
99.97%

During development, a temporary Cloudflare Quick Tunnel was also used to access the local Streamlit application through a browser. This was used only for testing and is not a permanent deployment.

Project Structure
sentiment-analysis/
│
├── api/
│   └── main.py
│
├── app/
│   └── app.py
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   │   └── download.py
│   │
│   ├── preprocessing/
│   │   └── clean.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_transformer.py
│   │   └── tune_optuna.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── utils/
│       └── seed.py
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── Dockerfile
├── app.py
├── README.md
└── .gitignore
Installation

Clone the repository:

git clone https://github.com/shashankkachouli/SENTIMENT-ANALYSIS.git
cd SENTIMENT-ANALYSIS

Create a virtual environment:

python -m venv .venv
Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Reproducing the Project

Download the dataset and place the CSV files inside:

data/raw/

Then run preprocessing:

python src/preprocessing/clean.py

Train the baseline:

python src/training/train_baseline.py

Train the transformer:

python src/training/train_transformer.py

Run Optuna optimization:

python src/training/tune_optuna.py

Evaluate the models:

python src/evaluation/evaluate.py

Run error analysis:

python src/evaluation/error_analysis.py

Run inference:

python src/inference/predict.py
Technologies Used
Python
Pandas
NumPy
Scikit-learn
PyTorch
Hugging Face Transformers
DistilBERT
Optuna
NLTK
Matplotlib
Seaborn
Streamlit
FastAPI
Docker
Git / GitHub
Key Takeaways

This project demonstrates:

Data cleaning and validation
Exploratory data analysis
Stratified dataset splitting
Data leakage detection
Classical ML modeling
Transformer fine-tuning
Hyperparameter optimization with Optuna
Evaluation using precision, recall, F1 and confusion matrices
Model error analysis
Reproducible ML pipelines
Model inference
Interactive Streamlit application development
Limitations

This model performs sentiment classification toward a specific entity in a tweet and should not be considered a completely general-purpose sentiment classifier.

The dataset may contain noisy labels, slang, multilingual text, sarcasm, and ambiguous tweets.

The reported results are based on the held-out test set created during preprocessing and should not be interpreted as universal real-world accuracy.

The Streamlit interface was tested locally and is not permanently deployed.

Author

Shashank
