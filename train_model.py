"""Train, compare, evaluate and save telecom complaint classification models."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
DATA_PATH = Path("data/fcc_complaints.csv")
ARTIFACTS = Path("artifacts")
OUTPUTS = Path("outputs")
ARTIFACTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
RANDOM_STATE = 42
TOP_N_CLASSES = 8

def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. First run: python download_data.py")
    df = pd.read_csv(DATA_PATH)
    required = {"ticket_created", "issue_type", "method", "issue", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.drop_duplicates().copy()
    df["ticket_created"] = pd.to_datetime(df["ticket_created"], errors="coerce", utc=True)
    df = df.dropna(subset=["issue", "ticket_created"])
    for col in ["issue_type", "method", "state", "issue"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df = df.dropna(subset=["issue", "issue_type", "method", "state"])
    # Use the eight most frequent issue labels to keep the student demo interpretable.
    top_labels = df["issue"].value_counts().head(TOP_N_CLASSES).index
    df = df[df["issue"].isin(top_labels)].copy()
    df["month"] = df["ticket_created"].dt.month.astype(int)
    df["year"] = df["ticket_created"].dt.year.astype(int)
    df["day_of_week"] = df["ticket_created"].dt.dayofweek.astype(int)
    df = df.rename(columns={"issue": "target_issue"})
    return df.reset_index(drop=True)

def make_pipeline(model):
    categorical = ["issue_type", "method", "state"]
    numeric = ["month", "year", "day_of_week"]
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)
    ])
    return Pipeline([("preprocessor", prep), ("classifier", model)])

def save_eda(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    df["target_issue"].value_counts().plot(kind="bar", color="#2563eb")
    plt.title("Most common complaint issue categories")
    plt.xlabel("Complaint issue")
    plt.ylabel("Number of records")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "01_issue_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["issue_type"].value_counts().plot(kind="bar", color="#0f766e")
    plt.title("Complaints by service form")
    plt.xlabel("Service form")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "02_service_form.png", dpi=160)
    plt.close()

    monthly = df.groupby(df["ticket_created"].dt.to_period("M")).size()
    plt.figure(figsize=(11, 5))
    monthly.plot(color="#7c3aed")
    plt.title("Complaint records by month")
    plt.xlabel("Month")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "03_monthly_trend.png", dpi=160)
    plt.close()

    cross = pd.crosstab(df["issue_type"], df["target_issue"])
    plt.figure(figsize=(11, 6))
    sns.heatmap(cross, cmap="Blues", annot=False)
    plt.title("Service form vs complaint issue")
    plt.xlabel("Complaint issue")
    plt.ylabel("Service form")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "04_service_issue_heatmap.png", dpi=160)
    plt.close()

def metric_row(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def main():
    print("Loading and cleaning FCC complaint records...")
    df = load_and_prepare()
    if df["target_issue"].nunique() < 2:
        raise ValueError("Not enough complaint categories to train a classifier.")
    save_eda(df)

    features = ["issue_type", "method", "state", "month", "year", "day_of_week"]
    target = "target_issue"
    X, y = df[features], df[target]

    # Keep only labels with at least two records so stratification is possible.
    counts = y.value_counts()
    keep = counts[counts >= 2].index
    X, y = X[y.isin(keep)], y[y.isin(keep)]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        # Experiment 1: compare a linear classifier with a tree ensemble.
        "Random Forest (baseline)": make_pipeline(
            RandomForestClassifier(
                n_estimators=150, max_depth=None, min_samples_leaf=1,
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
        # Experiment 2: tune tree depth and leaf size to control model complexity.
        "Random Forest (tuned)": make_pipeline(
            RandomForestClassifier(
                n_estimators=250, max_depth=18, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(metric_row(name, y_test, pred))
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values(
        ["f1_macro", "f1_weighted", "accuracy"], ascending=False
    )
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    # Save a model comparison chart.
    comparison.set_index("model")[["accuracy", "f1_weighted", "f1_macro"]].plot(
        kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0
    )
    plt.title("Model comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "05_model_comparison.png", dpi=160)
    plt.close()

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(11, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "06_confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    joblib.dump(best_model, ARTIFACTS / "telecom_complaint_model.joblib")
    metrics = {
        "best_model": best_name,
        "comparison": comparison.to_dict(orient="records"),
        "classification_report": report,
        "labels": labels,
        "feature_names": features,
        "dataset_rows_after_cleaning": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "class_distribution": y.value_counts().to_dict(),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "target": "target_issue",
        "note": "Scores describe this filtered FCC complaint sample and are not a guarantee of real-world performance."
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUTPUTS / "classification_report.txt").write_text(
        classification_report(y_test, best_pred, zero_division=0)
    )
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nClassification report:")
    print(classification_report(y_test, best_pred, zero_division=0))
    print("\nSaved model and metrics in artifacts/. EDA charts are in outputs/.")

if __name__ == "__main__":
    main()
