# Project Report: Telecom Customer Complaint Prediction & Classification

## 1. Introduction
Telecommunication services include phone, internet and television services. Consumers may submit complaints about different service issues. Classifying these complaints into meaningful issue categories can help organize complaint records and support basic complaint analytics.

## 2. Problem Statement
To develop a supervised machine learning application that predicts the likely issue category of a telecom consumer complaint using structured complaint information such as service form, service acquisition method, state and submission date.

## 3. Aim
To build and evaluate a multiclass classification system for telecom complaint issue categories and integrate the selected trained model into a user-friendly web dashboard.

## 4. Objectives
- Obtain a real public telecom complaint dataset.
- Understand the dataset structure and variables.
- Clean missing and duplicate records.
- Perform exploratory data analysis.
- Select relevant input features and target.
- Train at least two classification algorithms.
- Compare algorithms using suitable classification metrics.
- Perform model improvement experiments.
- Save the final model and build a prediction interface.

## 5. Dataset
Source: FCC CGB Consumer Complaints Data (official public dataset).
Link: https://opendata.fcc.gov/Consumer/CGB-Consumer-Complaints-Data/3xyp-aqkj

The project downloads up to 50,000 recent records for Phone, Internet and TV service forms. Each record represents an informal consumer complaint or inquiry submitted to the FCC. The FCC states that the complaint information is selected by consumers and is not verified by the FCC.

## 6. Variables
### Input features (X)
- issue_type: telecom service form
- method: manner in which the consumer obtains the service
- state: consumer's state
- month: month derived from ticket_created
- year: year derived from ticket_created
- day_of_week: weekday derived from ticket_created

### Target (y)
- issue: specific complaint issue category

This is a multiclass classification task because the target consists of multiple discrete issue categories.

## 7. Data Preprocessing
The training script:
- Removes duplicate rows.
- Parses the submission timestamp.
- Removes records missing the target, service type, method, state or valid timestamp.
- Retains the eight most frequent issue categories.
- Derives month, year and day of week from the timestamp.
- One-hot encodes categorical input features.
- Imputes missing categorical and numeric values where applicable.
- Scales numeric features for Logistic Regression.

The preprocessing steps are placed inside a scikit-learn pipeline to ensure that preprocessing is fitted on training data rather than the test data.

## 8. Exploratory Data Analysis
The program creates:
1. Complaint issue category distribution.
2. Complaint count by service form.
3. Monthly complaint trend.
4. Heatmap of service form against complaint issue.

Insert observations after running the project. Do not invent findings before examining the generated charts.

## 9. Train-Test Split
The dataset is divided into 80% training records and 20% testing records. `random_state=42` makes the split reproducible. Stratification preserves the approximate class proportions in both subsets.

## 10. Algorithms
### Logistic Regression
A supervised classification algorithm used as an interpretable baseline. It estimates class probabilities and supports multiclass classification.

### Random Forest Classifier
An ensemble of decision trees. It can learn non-linear relationships and interactions among input variables.

## 11. Evaluation
The models are compared using:
- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Macro F1-score
- Confusion matrix

Macro F1 is given priority during model selection because it gives equal importance to each retained class. Weighted metrics also account for class frequency.

## 12. Model Improvement Experiments
Experiment 1 — Algorithm comparison: compare balanced Logistic Regression against a baseline Random Forest using the same train-test split and input features.

Experiment 2 — Hyperparameter tuning: compare a baseline Random Forest (150 trees, unrestricted depth, minimum leaf size 1) with a tuned Random Forest (250 trees, maximum depth 18, minimum leaf size 2). Record the measured results.

| Experiment | Change | Result | Observation | Conclusion |
|---|---|---|---|---|
| 1 | Logistic Regression vs Random Forest | Fill from metrics.json | Compare macro F1 and confusion matrices | Select based on test evidence |
| 2 | Change Random Forest depth / remove a feature | Run and record | Check whether score changes | Keep only justified changes |

## 13. Front-End Application
The Streamlit application provides:
- Overview dashboard
- Complaint category prediction form
- Model performance and comparison
- Dataset explorer
- Project methodology and limitations

The app loads the saved model and does not retrain it on each prediction.

## 14. Conclusion
This project demonstrates an end-to-end multiclass classification workflow using public telecom consumer complaint records. The trained model, evaluation metrics and dashboard provide a practical demonstration of data preprocessing, model comparison, prediction and deployment. The model's usefulness is limited by the coverage and representativeness of the FCC complaint dataset.

## 15. Future Scope
- Use complaint text and NLP methods if an appropriate public text dataset becomes available.
- Validate on a later time period.
- Include more complaint categories when enough examples are available.
- Add an API using Flask or FastAPI.
- Monitor category drift and retrain with new data.
- Evaluate probability calibration and per-class errors.
