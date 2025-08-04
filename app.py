import streamlit as st
import os
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.exceptions import NotFittedError

MODEL_DIR = "models"
IMAGE_DIR = "images"

model_labels = {
    "blood_sugar": "🡸 Blood Sugar",
    "blood_pressure": "💉 Blood Pressure",
    "diabetes": "🍬 Diabetes",
    "fever": "🌡️ Fever",
    "heart": "❤️ Heart Disease",
    "readmission": "🏥 Readmission"
}

dropdown_options = {
    "gender": ["Male", "Female"],
    "smoking_history": ["never", "No Info", "current", "former", "not current", "ever"],
    "Fever_Severity": ["Normal", "Mild Fever", "High Fever"],
    "Smoking_Status": ["Never", "Current", "Former"],
    "Alcohol_Consumption": ["Yes", "No"],
    "Physical_Activity_Level": ["Low", "Moderate", "High"],
    "Physical_Activity": ["Active", "Sedentary"],
    "Diet_Type": ["Vegan", "Vegetarian", "Non-Vegetarian"],
    "bp_level": ["High", "Low", "Normal"],
    "Education_Level": ["Primary", "Secondary", "Higher"],
    "Employment_Status": ["Unemployed", "Employed", "Retired"],
    "readmitted": ["NO", "<30", ">30"],
    "change": ["Ch", "No"],
    "diabetesMed": ["Yes", "No"]
}

def inject_css():
    st.markdown("""
    <style>
    .title-container {
        background:black;
        color: white;
        padding: 40px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .abnormal {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    return joblib.load(path) if os.path.exists(path) else None

def load_sample_input(model_key):
    file = model_key.replace("blood_sugar", "bloodsugar").replace("blood_pressure", "bp") + ".csv"
    path = os.path.join("datasets", file)
    if os.path.exists(path):
        df = pd.read_csv(path).dropna()
        return df.iloc[0][:-1].to_dict() if not df.empty else {}
    return {}

def display_sample_inputs(sample_inputs):
    st.markdown("### 📜 Sample Input Values")
    if not sample_inputs:
        st.info("No sample data found.")
    else:
        for key, value in sample_inputs.items():
            st.markdown(f"- **{key}**: `{value}`")

def get_user_input(model_key):
    file = model_key.replace("blood_sugar", "bloodsugar").replace("blood_pressure", "bp") + ".csv"
    path = os.path.join("datasets", file)
    df = pd.read_csv(path).dropna()

    data = {}
    with st.form(f"{model_key}_form"):
        for col in df.columns[:-1]:
            if col in dropdown_options:
                data[col] = st.selectbox(col, dropdown_options[col])
            elif df[col].dtype == 'object':
                data[col] = st.text_input(col)
            else:
                data[col] = st.text_input(col, placeholder=str(df[col].mean()))
        submitted = st.form_submit_button("🔍 Predict")
    if submitted:
        try:
            for col in data:
                if df[col].dtype != 'object' and col not in dropdown_options:
                    data[col] = float(data[col])
        except ValueError as e:
            st.error(f"Numeric input required: {e}")
            return None, None
        return pd.DataFrame([data]), df
    return None, df

def predict(model, input_df):
    try:
        result = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max() if hasattr(model, "predict_proba") else 1.0
        return result, confidence
    except NotFittedError:
        st.error("Model not trained.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None, None

def show_prediction_result(result):
    if result in [0, "0", "Normal"]:
        st.markdown("<div class='prediction-box normal'>✅ Prediction Result : NORMAL — You are safe and healthy! 😊</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box abnormal'>❌ Prediction Result : ABNORMAL — Please consult a doctor. 🚑</div>", unsafe_allow_html=True)

def get_model_description(model_key):
    descriptions = {
        "blood_sugar": "🡸 Blood sugar prediction helps assess glucose levels in the body.",
        "blood_pressure": "💉 Blood pressure prediction identifies hypertension risks.",
        "diabetes": "🍬 Diabetes prediction estimates diabetes risk from health and lifestyle.",
        "fever": "🌡️ Fever prediction evaluates severity based on symptoms.",
        "heart": "❤️ Heart disease prediction assesses cardiovascular risk.",
        "readmission": "🏥 Readmission prediction detects risk of hospital readmission."
    }
    return descriptions.get(model_key, "")

# 🔍 Visualizations

def show_confidence_gauge(confidence):
    st.markdown("#### 📿 Model Confidence Score")
    st.caption("Shows how confident the model is about its prediction.")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if confidence >= 0.7 else "orange"}}))
    st.plotly_chart(fig, use_container_width=True)

def show_feature_distribution(df, user_input):
    st.markdown("#### 📊 Your Input vs Dataset Distribution")
    st.caption("Visual comparison of your input values against dataset distributions.")
    for feature in user_input.columns:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax, color='skyblue', label='Dataset')
            ax.axvline(user_input[feature].values[0], color='red', linestyle='--', label='Your Input')
            ax.set_title(f"Distribution of {feature}")
            ax.legend()
            st.pyplot(fig, use_container_width=True)

def show_class_distribution(df, target_column):
    st.markdown("#### 📈 Class Distribution in Dataset")
    st.caption("Pie chart of target classes in the training dataset.")
    fig, ax = plt.subplots()
    df[target_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig, use_container_width=True)

def show_correlation_heatmap(df):
    st.markdown("#### 🧬 Correlation Heatmap")
    st.caption("Shows correlation among all numeric features in the dataset.")
    numeric_df = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig, use_container_width=True)

def show_shap_explanation(model, input_df):
    st.markdown("#### 🧠 SHAP Feature Importance")
    st.caption("Explanation of how each feature influenced the model's decision.")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], input_df.iloc[0], ax=ax)
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

# 🚀 Main App
def main():
    st.set_page_config("Smart Health Predictor", layout="wide")
    inject_css()

    st.markdown("""
    <div class='title-container'>
        <h1>Welcome to Smart Health Predictor</h1>
        <p>Smart care starts with smart prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🦠 Choose a Prediction Model")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    for key in model_labels:
        if st.button(model_labels[key]):
            st.session_state.selected_model = key

    if st.session_state.selected_model:
        st.markdown("---")
        key = st.session_state.selected_model
        st.markdown(f"## 🧪 Predicting: {model_labels[key]}")
        st.info(get_model_description(key))

        cols = st.columns([1, 2])

        with cols[0]:
            image_path = os.path.join(IMAGE_DIR, f"{key}.png")
            if os.path.exists(image_path):
                st.image(image_path, use_column_width=True)
            display_sample_inputs(load_sample_input(key))

        with cols[1]:
            model = load_model(key)
            if model:
                input_df, df = get_user_input(key)
                if input_df is not None:
                    result, confidence = predict(model, input_df)
                    if result is not None:
                        show_prediction_result(result)

                        st.markdown("## 📊 Data Distribution for Your Given Input")

                        show_confidence_gauge(confidence)
                        show_class_distribution(df, df.columns[-1])
                        show_feature_distribution(df, input_df)
                        show_correlation_heatmap(df)
                        show_shap_explanation(model, input_df)
            else:
                st.warning("⚠️ Model not found. Please train and save it.")

    st.markdown("""<hr><div style='text-align:center; padding:20px; font-weight:bold;'>Created by Saran Kambala · ML Developer</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
