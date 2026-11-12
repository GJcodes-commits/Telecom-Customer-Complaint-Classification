from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Complaint Intelligence", page_icon="📡", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "artifacts" / "telecom_complaint_model.joblib"
METRICS_PATH = BASE / "artifacts" / "metrics.json"
DATA_PATH = BASE / "data" / "fcc_complaints.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.6rem; max-width: 1450px; }
.hero { background: linear-gradient(120deg,#101d3a,#153e75); padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px; }
.hero h1 { font-family: Manrope,sans-serif; font-size: 30px; margin: 0 0 8px 0; color: white; }
.hero p { color: #d6e5ff; margin: 0; font-size: 14px; }
.kpi { background: white; border: 1px solid #e7ebf2; padding: 18px 20px; border-radius: 14px; }
.kpi-label { color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { color: #12213e; font-family: Manrope,sans-serif; font-size: 27px; font-weight: 800; margin-top: 6px; }
.panel { background: white; border: 1px solid #e7ebf2; padding: 20px; border-radius: 14px; }
.small-note { color: #64748b; font-size: 12px; }
div.stButton > button { border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

def load_assets():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())

model, metrics = load_assets()

with st.sidebar:
    st.markdown("## 📡 Telecom AI")
    st.caption("Customer Complaint Intelligence")
    page = st.radio("NAVIGATION", ["Overview", "Predict Complaint", "Model Performance", "Dataset Explorer", "About Project"])
    st.divider()
    st.caption("Machine Learning • Multiclass Classification")
    if not MODEL_PATH.exists():
        st.warning("Model not trained yet. Follow the setup instructions in README.")

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

if page == "Overview":
    hero("Telecom Complaint Intelligence", "An ML-powered application for classifying telecom consumer complaint categories.")
    if metrics:
        comparison = pd.DataFrame(metrics["comparison"])
        best = comparison.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        cards = [
            ("Records analysed", f'{metrics["dataset_rows_after_cleaning"]:,}'),
            ("Complaint categories", str(len(metrics["labels"]))),
            ("Selected algorithm", metrics["best_model"]),
            ("Macro F1-score", f'{best["f1_macro"]:.3f}')
        ]
        for col,(label,value) in zip([c1,c2,c3,c4],cards):
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
        st.write("")
        left,right = st.columns([1.15,1])
        with left:
            st.subheader("Model comparison")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Complaint categories")
            st.bar_chart(pd.Series(metrics["class_distribution"], name="Complaints"))
        st.info("These statistics describe the downloaded FCC sample and the held-out test split. They should not be interpreted as guaranteed performance on all telecom customers.")
    else:
        st.info("Welcome! Download the dataset and train the models first. Follow the README's Quick Start steps.")
    st.subheader("What this application does")
    a,b,c = st.columns(3)
    a.markdown('<div class="panel"><h3>01 · Understand</h3><p>Explore complaint patterns, service forms and categories.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="panel"><h3>02 · Predict</h3><p>Enter complaint context and receive a predicted issue category.</p></div>', unsafe_allow_html=True)
    c.markdown('<div class="panel"><h3>03 · Evaluate</h3><p>Compare algorithms and inspect classification performance.</p></div>', unsafe_allow_html=True)

elif page == "Predict Complaint":
    hero("Predict a Complaint Category", "Enter the available complaint details to classify the likely issue category.")
    if model is None:
        st.error("The trained model is not available. Run `python download_data.py` and `python train_model.py` first.")
    else:
        labels = metrics["labels"]
        with st.form("prediction_form"):
            left,right = st.columns(2)
            with left:
                service = st.selectbox("Service form", ["Phone","Internet","TV"])
                method = st.selectbox("Service acquisition method", ["Mobile","Wireline","Cable","Satellite","Fiber","Other","Unknown"])
                state = st.selectbox("Consumer state", ["CA","NY","TX","FL","IL","PA","OH","GA","NC","MI","NJ","VA","WA","AZ","MA","Other/Unknown"])
            with right:
                month = st.select_slider("Complaint month", options=list(range(1,13)), value=6)
                year = st.number_input("Complaint year", min_value=2014, max_value=2035, value=2026, step=1)
                day = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            submitted = st.form_submit_button("✨ Predict complaint category", use_container_width=True)
        if submitted:
            day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)
            row = pd.DataFrame([{
                "issue_type": service, "method": method,
                "state": state if state != "Other/Unknown" else "Unknown",
                "month": int(month), "year": int(year), "day_of_week": day_num
            }])
            prediction = model.predict(row)[0]
            st.divider()
            st.success("Prediction generated")
            st.markdown("### Predicted complaint category")
            st.markdown(f"## 🏷️ {prediction}")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                prob_df = pd.DataFrame({"Category": model.classes_, "Estimated probability": probs})
                prob_df = prob_df.sort_values("Estimated probability", ascending=False).head(5)
                st.subheader("Top category estimates")
                st.bar_chart(prob_df.set_index("Category"))
                st.caption("Probabilities are model estimates, not calibrated guarantees or proof of the actual complaint cause.")
            st.subheader("Input summary")
            st.dataframe(row, use_container_width=True, hide_index=True)

elif page == "Model Performance":
    hero("Model Performance", "Compare the trained classifiers and inspect how well the selected model performs.")
    if not metrics:
        st.warning("Train the model first.")
    else:
        comparison = pd.DataFrame(metrics["comparison"])
        st.subheader("Algorithm comparison")
        st.dataframe(comparison.style.format({c:"{:.3f}" for c in comparison.select_dtypes("number").columns}), use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("model")[["accuracy","precision_weighted","recall_weighted","f1_weighted","f1_macro"]])
        st.markdown(f"**Selected final model:** {metrics['best_model']}")
        st.markdown(f"Training records: **{metrics['training_rows']:,}** · Testing records: **{metrics['testing_rows']:,}** · Split: **80:20**")
        for img,caption in [
            ("05_model_comparison.png","Model comparison"),
            ("06_confusion_matrix.png","Confusion matrix"),
        ]:
            path = BASE / "outputs" / img
            if path.exists():
                st.subheader(caption)
                st.image(str(path), use_container_width=True)
        report = metrics.get("classification_report", {})
        if report:
            st.subheader("Per-class evaluation")
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Inspect the downloaded public complaint sample and explore its structure.")
    if not DATA_PATH.exists():
        st.warning("Dataset not found. Run `python download_data.py`.")
    else:
        df = pd.read_csv(DATA_PATH)
        a,b,c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Duplicate rows", f"{df.duplicated().sum():,}")
        st.subheader("First records")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("Missing values")
        st.dataframe(df.isna().sum().rename("Missing values"), use_container_width=True)
        st.subheader("Service form distribution")
        st.bar_chart(df["issue_type"].value_counts())
        st.subheader("Most frequent complaint issues")
        st.bar_chart(df["issue"].value_counts().head(15))
        st.download_button("Download the dataset CSV", df.to_csv(index=False), "fcc_complaints.csv", "text/csv")

elif page == "About Project":
    hero("About the Project", "A student-built end-to-end supervised machine learning application.")
    st.markdown("""
    ### Problem statement
    Classify the likely issue category of a telecom consumer complaint using structured information available at intake.

    ### Input features (X)
    - Service form (Phone, Internet or TV)
    - Service acquisition method
    - Consumer state
    - Complaint month, year and day of week

    ### Target variable (y)
    - `issue`: the specific complaint issue recorded by the FCC

    ### Algorithms
    - Logistic Regression
    - Random Forest Classifier

    ### Evaluation
    Accuracy, weighted precision, weighted recall, weighted F1-score, macro F1-score and confusion matrix.

    ### Important limitation
    The FCC dataset consists of complaints submitted to the FCC. It does not represent every telecom customer or every customer-support interaction. The model classifies issue labels within the sampled data; it does not predict which customers will complain.
    """)
    st.markdown("[Official FCC dataset](https://opendata.fcc.gov/Consumer/CGB-Consumer-Complaints-Data/3xyp-aqkj)")
