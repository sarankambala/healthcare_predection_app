import streamlit as st
import os
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

MODEL_DIR = "models"
IMAGE_DIR = "images"

# Emoji-enhanced labels for models
model_labels = {
    "blood_sugar": "🩸 Blood Sugar",
    "blood_pressure": "💉 Blood Pressure",
    "diabetes": "🍬 Diabetes",
    "fever": "🌡️ Fever",
    "heart": "❤️ Heart Disease",
    "readmission": "🏥 Readmission"
}

# Predefined dropdown options
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

# CSS Styling
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
    .model-list li {
        margin: 8px 0;
        font-size: 18px;
    }
    .model-list li:hover {
        color: black;
        cursor: pointer;
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
    st.markdown("### 🧾 Sample Input Values")
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
                data[col] = st.text_input(col)
        submitted = st.form_submit_button("🔍 Predict")
    if submitted:
        try:
            for col in data:
                if df[col].dtype != 'object' and col not in dropdown_options:
                    data[col] = float(data[col])
        except ValueError as e:
            st.error(f"Numeric input required: {e}")
            return None
        return pd.DataFrame([data])
    return None

def predict(model, input_df):
    try:
        return model.predict(input_df)[0]
    except NotFittedError:
        st.error("Model not trained.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None

def show_prediction_result(result):
    if result in [0, "0", "Normal"]:
        st.markdown("<div class='prediction-box normal'>✅ Prediction Result : NORMAL — You are safe and healthy! 😊</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box abnormal'>❌ Prediction Result : ABNORMAL — Please consult a doctor. 🚑</div>", unsafe_allow_html=True)

def get_model_description(model_key):
    descriptions = {
        "blood_sugar": "🩸 Blood sugar prediction helps assess glucose levels in the body, useful for monitoring diabetes and overall metabolic health.",
        "blood_pressure": "💉 Blood pressure prediction helps identify hypertension risks and cardiovascular health.",
        "diabetes": "🍬 Diabetes prediction estimates the likelihood of having or developing diabetes based on medical and lifestyle features.",
        "fever": "🌡️ Fever prediction evaluates symptoms and severity to help detect underlying infections or conditions.",
        "heart": "❤️ Heart disease prediction analyzes patient data to assess the risk of cardiovascular issues.",
        "readmission": "🏥 Readmission prediction identifies patients likely to be readmitted soon after hospital discharge."
    }
    return descriptions.get(model_key, "")

# Main
def main():
    st.set_page_config("Smart Health Predictor", layout="wide")
    inject_css()

    st.markdown("""
    <div class='title-container'>
        <h1>Welcome to Smart Health Predictor</h1>
        <p>Smart care starts with smart prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🩺 Choose a Prediction Model")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    for key in model_labels:
        if st.button(model_labels[key]):
            st.session_state.selected_model = key

    # If a model is selected
    if st.session_state.selected_model:
        st.markdown("---")
        key = st.session_state.selected_model
        st.markdown(f"## 🧪 Predicting: {model_labels[key]}")
        st.info(get_model_description(key))

        cols = st.columns([1, 2])  # Left: image/sample | Right: form/prediction

        with cols[0]:
            image_path = os.path.join(IMAGE_DIR, f"{key}.png")
            if os.path.exists(image_path):
                st.image(image_path, use_column_width=True)
            display_sample_inputs(load_sample_input(key))

        with cols[1]:
            model = load_model(key)
            if model:
                input_df = get_user_input(key)
                if input_df is not None:
                    result = predict(model, input_df)
                    if result is not None:
                        show_prediction_result(result)
            else:
                st.warning("⚠️ Model not found. Please train and save it.")

    # Footer
    st.markdown("""
<hr><div style='text-align:center; padding:20px; color:black; font-weight:bold;'>
    <strong>Created by Saran Kambala · ML Developer</strong>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
