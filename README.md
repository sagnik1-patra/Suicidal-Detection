🧠 MindShield — Suicidal Thought Detection (Demo)

Research/education demo for binary classification of suicidal ideation in text.
Not a medical device. If someone may be in danger, contact local emergency services immediately.

✨ What this repo does

Ingests a CSV of Reddit-style texts and labels.

Cleans + normalizes labels to {0,1} → 0 = non-suicidal, 1 = suicidal.

Makes stratified train/valid/test splits.

Trains a TF-IDF + Logistic Regression baseline.

Exports artifacts: H5 / PKL / YAML / JSONL, plus metrics, learning-curve, confusion-matrix.

Provides inference scripts:

Use the trained model if available;

otherwise zero-shot (BART-MNLI) + small heuristic highlight/boost.

🗂 Project layout (suggested)
Suicidal Detection/
├─ archive/
│  └─ suicidal_ideation_reddit_annotated.csv      # your dataset
├─ build_artifacts.py                             # CSV → H5/PKL/YAML/JSONL
├─ ms_accuracy_heatmap.py                         # accuracy learning curve + confusion matrix
├─ mindshield_predict.py                          # train+save model, test predictions, external inference
├─ mindshield_identify.py                         # identify suicidal thoughts (yes/no), model or zero-shot
└─ (outputs written here)


The scripts assume Windows paths provided by you. Feel free to rename files; keep the constants updated at the top of each script.

✅ Requirements

Python 3.9–3.11 (tested on 3.11)

OS: Windows 10/11 (works on Linux/macOS with minor path changes)

Install deps (one-liner):

pip install pandas numpy scikit-learn matplotlib h5py pyyaml joblib transformers


If you don’t need charts/artifacts/zero-shot in a given run, you can omit those specific packages.

📥 Dataset

Expected input file (from your setup):
C:\Users\sagni\Downloads\Suicidal Detection\archive\suicidal_ideation_reddit_annotated.csv

Text column: auto-detected from candidates like text, message, content, body, post, comment, selftext, title.
If both title and selftext exist, they are concatenated.

Label column: auto-detected from candidates like label, class, target, is_suicidal, suicidal, suicide, risk, y.
Values are normalized to {0,1} using robust string mapping (e.g., "suicidal"→1, "non-suicidal"→0).

If auto-detection fails, rename your columns to text and label.

🚀 Quickstart
1) Build dataset artifacts

Run build_artifacts.py (already customized for your paths):

Reads the CSV

Splits: 70/15/15

Writes to C:\Users\sagni\Downloads\Suicidal Detection\:

Artifacts:

mindshield_dataset.h5 (HDF5; /train, /valid, /test groups with text, label)

mindshield_dataset.pkl (Python pickle dict with splits + meta)

mindshield_dataset.jsonl (all splits, one record per line)

mindshield_summary.json (dataset meta, sizes, class balance)

mindshield_config.yaml (same meta in YAML)

2) Train, save model, and generate predictions

Run mindshield_predict.py:

Trains TF-IDF + Logistic Regression on train+valid

Saves model to ms_model.joblib

Evaluates on test, writes:

ms_predictions.csv & ms_predictions.json (test set)

ms_infer_summary.json

ms_classification_report.txt (Precision/Recall/F1 per class)

Predict on your own file (TXT/CSV/JSON): set INFER_SOURCE = r"...\my_texts.csv" in the script and run again.
Outputs will be saved as ms_predictions_external.csv/.json.

3) Accuracy curve & confusion-matrix heatmap

Run ms_accuracy_heatmap.py:

Creates ms_accuracy_curve.png (learning curve: training vs. validation accuracy)

Creates ms_confusion_matrix.png (normalized; counts are overlaid)

Also writes ms_metrics.json and ms_classification_report.txt again for convenience.

4) Identify suicidal thoughts (model or zero-shot)

Run mindshield_identify.py:

If ms_model.joblib exists → uses it.

Else → zero-shot via facebook/bart-large-mnli, with heuristic boosts and phrase highlights.

Accepts TXT/CSV/JSON (INFER_SOURCE) or uses built-in sample texts.

Writes:

mindshield_suicide_pred.csv

mindshield_suicide_pred.json (with summary counts)

🧪 Example output (JSON)
{
  "items": [
    {
      "text": "Sometimes I just want to disappear.",
      "pred_label": 1,
      "pred_name": "suicidal",
      "prob_suicidal": 0.87,
      "highlights": [[12, 22, "ideation_soft"]]
    }
  ],
  "summary": {
    "total": 1,
    "pred_counts": {"non-suicidal": 0, "suicidal": 1}
  }
}

⚙️ Configuration knobs

Each script exposes constants at the top:

CSV_PATH
C:\Users\sagni\Downloads\Suicidal Detection\archive\suicidal_ideation_reddit_annotated.csv

OUT_DIR
C:\Users\sagni\Downloads\Suicidal Detection

MODEL_PATH
C:\Users\sagni\Downloads\Suicidal Detection\ms_model.joblib

SPLIT_FRAC = (0.70, 0.15, 0.15)
Change to tweak dataset split.

Zero-shot threshold (in mindshield_identify.py):
ZS_SUICIDAL_THRESHOLD = 0.50

📈 What the charts mean

Learning curve (ms_accuracy_curve.png)
Shows if the model benefits from more data and whether it’s over/under-fitting.

Confusion matrix (ms_confusion_matrix.png)
Normalized by true labels; counts included in parentheses.
Useful to see false negatives (missed risky messages) vs false positives.

🔍 Model details (baseline)

Vectorizer: TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200k)

Classifier: LogisticRegression(solver="saga", penalty="l2", class_weight="balanced")

Why this combo? Strong sparse baseline for short texts, fast to train, interpretable.

You can replace the classifier with LinearSVC, or wrap into CalibratedClassifierCV for calibrated probabilities.
![Confusion Matrix Heatmap](ms_accuracy_curve.png)
🔐 Privacy & Ethics

This project processes sensitive content. Treat all inputs as confidential.

Avoid uploading real personal data to public repos or third-party services.

Human review is essential for any “high-risk” outputs.

NOT a medical device and NOT clinical advice. For emergencies, contact local services immediately.

🛠 Troubleshooting

UnicodeDecodeError reading CSV
The scripts already retry with encoding="latin-1". If needed, specify your encoding in pd.read_csv.

“No text/label column found”
Rename to text and label, or add your column names to the candidate lists in the script.

Model not found
Run mindshield_predict.py once to train & save ms_model.joblib, or use mindshield_identify.py (zero-shot fallback).

Imbalanced dataset, poor recall
Class weight is set to "balanced". You can also tune C, try LinearSVC, or gather more positive examples.

Matplotlib figure not showing
The scripts save images to files; if you run in Jupyter, they’ll also display inline.

🧩 Extending

Swap baseline with a Hugging Face model (e.g., DistilBERT) using transformers.

Add PII scrubbing before saving predictions (emails, phones, names).

Add a Streamlit dashboard and/or a FastAPI microservice for UI/API.

Use explainability libs (e.g., eli5, shap) for feature attributions.
Author
SAGNIK PATRA
