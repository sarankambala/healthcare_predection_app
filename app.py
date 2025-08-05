import streamlit as st
import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.exceptions import NotFittedError

MODEL_DIR = "models"
IMAGE_DIR = "images"

model_labels = {
    "blood_sugar": "🩸 Blood Sugar",
    "blood_pressure": "💉 Blood Pressure",
    "diabetes": "🍬 Diabetes",
    "fever": "🌡 Fever",
    "heart": "❤ Heart Disease",
    "readmission": "🏥 Readmission"
}

model_details = {
    "blood_sugar": {
        "description": "Predicts glucose levels in your bloodstream",
        "inputs": "Age, BMI, genetic factors, lifestyle habits",
        "benefits": "Early detection of prediabetes, helps manage diet and exercise"
    },
    "blood_pressure": {
        "description": "Assesses hypertension risk",
        "inputs": "Blood pressure readings, age, weight, family history",
        "benefits": "Prevents cardiovascular complications through early intervention"
    },
    "diabetes": {
        "description": "Estimates diabetes risk based on health markers",
        "inputs": "Glucose levels, insulin resistance, BMI, age",
        "benefits": "Early lifestyle changes can prevent or delay onset"
    },
    "fever": {
        "description": "Evaluates fever severity and potential causes",
        "inputs": "Temperature, symptoms, duration, medical history",
        "benefits": "Helps determine when medical attention is needed"
    },
    "heart": {
        "description": "Predicts cardiovascular disease risk",
        "inputs": "Cholesterol, blood pressure, age, lifestyle factors",
        "benefits": "Identifies at-risk individuals for preventive care"
    },
    "readmission": {
        "description": "Estimates hospital readmission likelihood",
        "inputs": "Treatment history, vitals, comorbidities",
        "benefits": "Helps healthcare providers improve discharge planning"
    }
}

dropdown_options = {
    "gender": ["Male", "Female"],
    "Gender": ["Male", "Female"],
    "smoking_history": ["never", "No Info", "current", "former", "not current", "ever"],
    "Smoking_Status": ["Never", "Current", "Former"],
    "Fever_Severity": ["Normal", "Mild Fever", "High Fever"],
    "Alcohol_Consumption": ["Yes", "No"],
    "Physical_Activity_Level": ["Low", "Moderate", "High"],
    "Physical_Activity": ["Active", "Sedentary"],
    "Diet_Type": ["Vegan", "Vegetarian", "Non-Vegetarian"],
    "bp_level": ["High", "Low", "Normal"],
    "Education_Level": ["Primary", "Secondary", "Higher"],
    "Employment_Status": ["Unemployed", "Employed", "Retired"],
    "readmitted": ["NO", "<30", ">30"],
    "change": ["Ch", "No"],
    "diabetesMed": ["Yes", "No"],
    "Diabetes": ["Yes", "No"],
    "Family_History": ["Yes", "No"],
    "Country": ["India", "USA", "UK", "Other"],
}

def inject_css():
    st.markdown("""
    <style>
    :root {
        --primary: #4361ee;
        --secondary: #3a0ca3;
        --success: #4cc9f0;
        --danger: #f72585;
        --light: #f8f9fa;
        --dark: #212529;
        --text: #2b2d42;
        --background: #f8f9fa;
        --card-bg: #ffffff;
        --button-bg: #4895ef;
        --button-hover: #3a0ca3;
    }
    
    body {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .title-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .model-btn {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        background-color: var(--button-bg);
        color: white;
        border: none;
        font-weight: 600;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .model-btn:hover {
        background-color: var(--button-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .model-btn.active {
        background-color: var(--secondary);
        box-shadow: 0 0 0 2px var(--primary);
    }
    
    .model-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary);
    }
    
    .input-section {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .prediction-container {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    .normal-prediction {
        border-left: 6px solid var(--success);
        background: linear-gradient(90deg, rgba(76,201,240,0.08) 0%, rgba(255,255,255,1) 100%);
    }
    
    .abnormal-prediction {
        border-left: 6px solid var(--danger);
        background: linear-gradient(90deg, rgba(247,37,133,0.08) 0%, rgba(255,255,255,1) 100%);
    }
    
    .prediction-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .prediction-icon {
        font-size: 2rem;
    }
    
    .normal-icon {
        color: var(--success);
    }
    
    .abnormal-icon {
        color: var(--danger);
    }
    
    .prediction-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        color: var(--dark);
    }
    
    .normal-title {
        color: var(--success);
    }
    
    .abnormal-title {
        color: var(--danger);
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(67,97,238,0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 1rem 0;
        color: var(--primary);
        border: 1px solid rgba(67,97,238,0.2);
    }
    
    .suggestion-box {
        background: rgba(248,249,250,0.8);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .input-values-box {
        background: rgba(248,249,250,0.8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-left: 4px solid var(--primary);
    }
    
    .chart-container {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    .chart-description {
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
    }
    
    .feature-table th {
        background-color: var(--primary);
        color: white;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
    }
    
    .feature-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    .feature-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    .feature-table tr:hover {
        background-color: #f1f3f5;
    }
    
    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 3rem 0 1.5rem;
        color: var(--primary);
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-title:after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 60px;
        height: 4px;
        background: var(--primary);
        border-radius: 2px;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        background-color: var(--primary);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stSelectbox, .stTextInput, .stNumberInput {
        margin-bottom: 1.25rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .title-container {
            padding: 1.75rem;
        }
        
        .prediction-container {
            padding: 1.5rem;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
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
    st.markdown("#### 📜 Sample Input Values")
    if not sample_inputs:
        st.info("No sample data found.")
    else:
        st.markdown("""
        <table class="feature-table">
            <thead>
                <tr>
                    <th style="width: 50%;">Feature</th>
                    <th style="width: 50%;">Value</th>
                </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)
        
        for key, value in sample_inputs.items():
            st.markdown(f"""
                <tr>
                    <td style="width: 100%; border:none ">{key}:</td>
                    <td style="width: 100%; border:none "><code>{value}</code></td>
                </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("</tbody></table>", unsafe_allow_html=True)

def get_user_input(model_key):
    file = model_key.replace("blood_sugar", "bloodsugar").replace("blood_pressure", "bp") + ".csv"
    path = os.path.join("datasets", file)
    df = pd.read_csv(path).dropna()

    data = {}
    with st.form(f"{model_key}_form"):
        st.markdown("#### 📝 Enter Your Health Information")
        cols = st.columns(2)
        col_idx = 0
        for i, col in enumerate(df.columns[:-1]):
            with cols[col_idx]:
                if col in dropdown_options:
                    data[col] = st.selectbox(col, dropdown_options[col])
                elif df[col].dtype == 'object':
                    data[col] = st.text_input(col)
                else:
                    data[col] = st.number_input(col, value=None, format="%.2f", placeholder="Enter value")
            col_idx = 1 if col_idx == 0 else 0
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
    if submitted:
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

def get_prediction_suggestions(model_key, result, input_data):
    suggestions = {
        "blood_sugar": {
            "normal": [
                "Your blood sugar levels are within the normal range (70-99 mg/dL fasting).",
                "Maintain a balanced diet with complex carbohydrates like whole grains and vegetables.",
                "Engage in at least 150 minutes of moderate exercise per week.",
                "Consider annual checkups if you're over 40 or have family history of diabetes."
            ],
            "abnormal": [
                "Your blood sugar levels are elevated (fasting ≥126 mg/dL indicates diabetes).",
                "Reduce intake of simple sugars and processed carbohydrates.",
                "Aim for at least 30 minutes of physical activity daily.",
                "Schedule a consultation with your doctor for proper evaluation.",
                "Monitor your blood glucose levels regularly."
            ]
        },
        "blood_pressure": {
            "normal": [
                "Your blood pressure is within the healthy range (<120/80 mmHg).",
                "Continue with a low-sodium diet (<2,300 mg per day).",
                "Practice stress-reduction techniques like meditation or deep breathing.",
                "Maintain regular physical activity for cardiovascular health."
            ],
            "abnormal": [
                "Your blood pressure is elevated (≥130/80 mmHg indicates hypertension).",
                "Reduce sodium intake to less than 1,500 mg per day.",
                "Limit alcohol consumption to 1 drink per day for women, 2 for men.",
                "Consult a healthcare professional for proper evaluation and monitoring.",
                "Consider regular home blood pressure monitoring."
            ]
        },
        "diabetes": {
            "normal": [
                "You show no signs of diabetes risk based on current parameters.",
                "Maintain healthy weight through balanced nutrition.",
                "Include fiber-rich foods like vegetables, fruits, and whole grains.",
                "Get regular physical activity to maintain insulin sensitivity."
            ],
            "abnormal": [
                "You show potential diabetes risk factors (HbA1c ≥6.5% indicates diabetes).",
                "Consult a doctor for HbA1c testing and complete evaluation.",
                "Focus on low glycemic index foods to manage blood sugar.",
                "Aim for at least 150 minutes of moderate exercise weekly.",
                "Monitor for symptoms like increased thirst, frequent urination."
            ]
        },
        "fever": {
            "normal": [
                "No fever detected (normal body temperature 97°F-99°F).",
                "Maintain good hygiene practices to prevent infections.",
                "Stay hydrated with water and electrolyte solutions.",
                "Get adequate rest to support immune function."
            ],
            "abnormal": [
                "Fever detected (≥100.4°F indicates possible infection).",
                "Rest well and stay hydrated with electrolyte solutions.",
                "Monitor temperature every 4-6 hours.",
                "Consult a doctor if fever persists beyond 48 hours or exceeds 103°F.",
                "Watch for accompanying symptoms like rash or difficulty breathing."
            ]
        },
        "heart": {
            "normal": [
                "No signs of heart disease risk detected.",
                "Continue cardiovascular exercises like walking or swimming.",
                "Maintain a heart-healthy diet rich in omega-3s and antioxidants.",
                "Avoid smoking and limit alcohol consumption."
            ],
            "abnormal": [
                "Potential cardiovascular risk factors detected.",
                "Consult a cardiologist for complete evaluation.",
                "Reduce saturated fats and increase fiber intake.",
                "Establish a regular exercise routine (start with 30 minutes daily).",
                "Monitor blood pressure and cholesterol levels regularly."
            ]
        },
        "readmission": {
            "normal": [
                "Low risk of hospital readmission identified.",
                "Follow your treatment plan diligently as prescribed.",
                "Attend all scheduled follow-up appointments.",
                "Monitor symptoms carefully and report any changes."
            ],
            "abnormal": [
                "High readmission risk identified.",
                "Contact your healthcare provider immediately for care plan review.",
                "Ensure proper medication adherence and understanding.",
                "Arrange for follow-up within 7 days of discharge.",
                "Identify support systems for post-hospital care."
            ]
        }
    }
    
    status = "normal" if result in [0, "0", "Normal"] else "abnormal"
    base_suggestions = suggestions[model_key][status]
    
    # Add personalized suggestions
    personalized = []
    if model_key == "blood_sugar" and input_data.get("age", 0) > 40:
        personalized.append("Regular glucose monitoring is recommended for your age group.")
    if model_key == "heart" and input_data.get("smoking_history", "") in ["current", "former"]:
        personalized.append("Smoking cessation would significantly improve your cardiovascular health.")
    if model_key == "diabetes" and input_data.get("bmi", 0) > 25:
        personalized.append("Weight reduction of 5-10% can significantly improve insulin sensitivity.")
    
    return base_suggestions + personalized

def format_input_values(input_data):
    formatted = []
    for key, value in input_data.items():
        formatted.append(f"{key}: {value}")
    return formatted

def show_class_distribution(df, target_column):
    # Get the prediction class counts
    class_counts = df[target_column].value_counts()
    
    # Create a smaller figure (reduced from 5 to 4 inches)
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Use colors that match your prediction output (green/red)
    colors = ['#27ae60', '#e74c3c']  # Green for normal, red for abnormal
    
    # Create the pie chart with smaller percentage font size
    wedges, texts, autotexts = ax.pie(
        class_counts,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 3, 'color': 'white', 'weight': 'bold'}  # Smaller font size
    )
    
    # Add a legend with meaningful labels
    ax.legend(wedges, 
              ['Normal', 'Abnormal'],
              title="Categories",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Equal aspect ratio ensures the pie is drawn as a circle
    ax.axis('equal')  
    ax.set_title("Dataset Class Distribution", 
                 pad=8, 
                 fontsize=6, 
                 fontweight='bold', 
                 color='#4361ee')
    
    # Add a colored border that matches the prediction result
    prediction_color = '#27ae60' if class_counts.idxmax() in [0, "0", "Normal"] else '#e74c3c'
    for wedge in wedges:
        wedge.set_edgecolor(prediction_color)
        wedge.set_linewidth(2)
    
    # Add a colored circle around the pie chart to match prediction
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Add prediction indicator text in the center
    plt.text(0, 0, 
             "Most Common:\n" + ("Normal" if class_counts.idxmax() in [0, "0", "Normal"] else "Abnormal"), 
             ha='center', 
             va='center', 
             fontsize=10,
             color=prediction_color,
             weight='bold')
    
    with st.container():
        st.markdown("### Dataset Outcome Distribution")
        st.markdown("""
        <div style='background-color: black; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                    border-left: 4px solid #4361ee;'>
            <p style='margin: 0; font-size: 14px;'>
                This shows the proportion of outcomes in our training data. The colored border indicates the most common prediction.
                <br><strong>Green:</strong> Normal cases &nbsp;|&nbsp; <strong>Red:</strong> Abnormal cases
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)

def show_prediction_result(model_key, result, confidence, input_data):
    # Determine prediction status
    pred_class = "normal" if result in [0, "0", "Normal"] else "abnormal"
    result_text = "NORMAL" if pred_class == "normal" else "ABNORMAL"
    color = "#27ae60" if pred_class == "normal" else "#e74c3c"  # Bright green/red
    
    # Main container
    with st.container():
        st.markdown("---")
        
        # PREDICTION HEADER (only colored element)
        st.markdown(f"""
        <div style='background-color: {color}; 
                    color: white; 
                    padding: 20px; 
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
            <h1 style='margin: 0;'>Prediction Result: {result_text}</h1>
            <p style='margin: 5px 0 0; font-size: 16px;'>
                {'All parameters within healthy range' if pred_class == 'normal' else 'Potential health concern detected'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # CONFIDENCE METER (with matching color indicator)
        confidence_color = "#27ae60" if confidence >= 0.8 else ("#f39c12" if confidence >= 0.6 else "#e74c3c")
        st.markdown("### Assessment Confidence")
        st.markdown(f"""
        <div style='background-color: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin: 10px 0 20px;
                    border-left: 4px solid {confidence_color};'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span>Model Confidence Level:</span>
                <span style='font-weight: bold; color: {confidence_color};'>{confidence*100:.1f}%</span>
            </div>
            <div style='height: 8px; background-color: #ecf0f1; border-radius: 4px;'>
                <div style='height: 100%; width: {confidence*100}%; background-color: {confidence_color}; border-radius: 4px;'></div>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <small style='color: #7f8c8d;'>Low</small>
                <small style='color: #7f8c8d;'>Medium</small>
                <small style='color: #7f8c8d;'>High</small>
            </div>
            <p style='font-size: 13px; color: #7f8c8d; margin: 5px 0 0;'>
                {'High reliability' if confidence >= 0.8 else ('Moderate reliability' if confidence >= 0.6 else 'Low reliability')} • 
                Based on {model_labels[model_key]} model
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # INPUT PARAMETERS (neutral display)
        with st.expander("📋 Your Health Parameters", expanded=True):
            cols = st.columns(2)
            for i, (key, value) in enumerate(input_data.items()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa;
                                padding: 12px;
                                border-radius: 8px;
                                margin-bottom: 10px;
                                border-left: 3px solid #3498db;'>
                        <div style='font-weight: 600; color: #2c3e50;'>{key.replace('_', ' ').title()}</div>
                        <div style='font-size: 15px;'>{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # RECOMMENDATIONS SECTION (neutral with color-coded bullets)
        st.markdown("---")
        st.markdown("### Health Recommendations")
        
        suggestions = get_prediction_suggestions(model_key, result, input_data)
        
        # Add personalized suggestions based on inputs
        if model_key == "diabetes" and input_data.get("bmi", 0) > 25:
            suggestions.append(f"Aim to reduce weight by {max(5, min(10, int(input_data['bmi']-22)))}% to improve insulin sensitivity")
        if model_key == "blood_pressure" and input_data.get("Physical_Activity_Level", "") == "Low":
            suggestions.append("Begin with 15-minute walks twice daily, gradually increasing to 30 minutes")
        
        for suggestion in suggestions:
            st.markdown(f"""
            <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
                <div style='color: {color}; font-weight: bold; margin-right: 10px;'>•</div>
                <div style='flex: 1;'>{suggestion}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # NEXT STEPS (neutral with colored action buttons)
        st.markdown("---")
        st.markdown("### Next Steps")
        
        if pred_class == "normal":
            st.success("Maintain your health with these actions:")
            st.markdown("- Continue regular health screenings")
            st.markdown("- Monitor your parameters monthly")
            st.markdown("- Maintain current healthy habits")
        else:
            st.error("Consider these important actions:")
            st.markdown(f"- Schedule a consultation with your { 'endocrinologist' if model_key == 'diabetes' else 'cardiologist' if model_key == 'heart' else 'physician'}")
            st.markdown("- Begin implementing recommendations above")
            st.markdown("- Follow up with testing in 2-4 weeks")
        
        # Add urgency for critical cases
        if pred_class == "abnormal" and confidence > 0.85:
            st.markdown(f"""
            <div style='background-color: #fde8e8;
                        padding: 12px;
                        border-radius: 8px;
                        margin: 15px 0;
                        border-left: 4px solid #e74c3c;'>
                <div style='display: flex; align-items: center;'>
                    <span style='color: #e74c3c; font-size: 20px; margin-right: 10px;'>⚠</span>
                    <strong>Important:</strong> Consider seeking medical evaluation within 48 hours
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
def show_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Confidence Level", 'font': {'size': 18, 'color': '#4361ee'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkblue'},
            'bar': {'color': '#4361ee'},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 80], 'color': '#fff2cc'},
                {'range': [80, 100], 'color': '#d5e8d4'}],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100}}))
    
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=80),
        height=350,
        width=500
    )
    
    with st.container():
        st.markdown("### Model Confidence Level")
        st.markdown("This gauge shows the model's confidence in its prediction (0-100%). Higher values indicate more reliable results. Confidence above 80% is considered strong.")
        st.plotly_chart(fig, use_container_width=True)

def show_class_distribution(df, target_column):
    # Get value counts and labels
    value_counts = df[target_column].value_counts()
    labels = ['Normal', 'Abnormal']  # Custom labels for legend
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)  # Small size but high DPI for clarity
    fig.subplots_adjust(left=0.1, right=0.75)  # Make space for legend
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        value_counts,
        autopct='%1.1f%%',
        colors=['#4361ee', '#4cc9f0'],
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'},
        startangle=90,
        pctdistance=0.85  # Push percentages inward
    )
    
    # Add legend outside the pie
    ax.legend(
        wedges,
        labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8
    )
    
    # Add title and adjust layout
    ax.set_title(
        "Dataset Class Distribution",
        pad=10,
        fontsize=10,
        fontweight='bold',
        color='#4361ee'
    )
    
    # Equal aspect ratio ensures the pie is drawn as a circle
    ax.axis('equal')
    
    with st.container():
        st.markdown("### Outcome Distribution in Dataset")
        st.markdown("""
        <div style='background-color: #f8f9fa; 
                    padding: 12px; 
                    border-radius: 8px; 
                    margin-bottom: 15px;
                    border-left: 3px solid #4361ee;'>
            <p style='margin: 0; font-size: 14px;'>
                Shows the proportion of outcomes in training data. <br>
                <span style='color: #4361ee; font-weight: bold;'>Blue:</span> Normal cases | 
                <span style='color: #4cc9f0; font-weight: bold;'>Light Blue:</span> Abnormal cases
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        
def show_feature_distribution(df, user_input):
    st.markdown("## 📊 Feature Distribution Analysis")
    
    numeric_features = [
        feature for feature in user_input.columns
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
    ]
    
    for i in range(0, len(numeric_features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(numeric_features):
                feature = numeric_features[i + j]
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.histplot(df[feature], kde=True, ax=ax, color='#4361ee', label='Dataset')
                    ax.axvline(user_input[feature].values[0], color='#f72585', linestyle='--', linewidth=2, label='Your Value')
                    ax.set_title(f"Distribution of {feature}", pad=15, fontsize=14, fontweight='bold', color='#4361ee')
                    ax.legend()
                    plt.tight_layout()
                    
                    st.markdown(f"#### Your {feature} Value")
                    st.markdown("The red line shows where your value falls within our dataset distribution. Values in the extreme ends may indicate potential health risks.")
                    st.pyplot(fig, use_container_width=True)

def show_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            ax=ax,
            fmt=".2f",
            linewidths=.5,
            annot_kws={"size": 10, 'weight': 'bold'},
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix", pad=20, fontsize=14, fontweight='bold', color='#4361ee')
        
        st.markdown("## Feature Relationships")
        st.markdown("Shows how different health factors relate to each other. Strong correlations (close to 1 or -1) indicate important relationships that influence predictions.")
        st.pyplot(fig, use_container_width=True)

def main():
    st.set_page_config("Smart Health Predictor", layout="wide", page_icon="🧬")
    inject_css()

    st.markdown("""
    <div class='title-container'>
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">Smart Health Predictor</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">AI-powered health assessment for better preventive care</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🦠 Choose a Prediction Model")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(model_labels.keys())[0]

    # Model selection section
    cols = st.columns([1, 2])

    with cols[0]:
        st.markdown("#### Select Health Model")
        for key in model_labels:
            if st.button(model_labels[key], key=key):
                st.session_state.selected_model = key

    with cols[1]:
        selected_key = st.session_state.selected_model
        st.markdown(f"""
        <div class='model-card'>
            <h3 style="color: #4361ee; margin-bottom: 1rem;">{model_labels[selected_key]}</h3>
            <p><strong style="color: #4361ee;">What it does:</strong> {model_details[selected_key]['description']}</p>
            <p><strong style="color: #4361ee;">Key inputs:</strong> {model_details[selected_key]['inputs']}</p>
            <p><strong style="color: #4361ee;">Benefits:</strong> {model_details[selected_key]['benefits']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    input_cols = st.columns([1, 2])

    with input_cols[0]:
        image_path = os.path.join(IMAGE_DIR, f"{selected_key}.png")
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        display_sample_inputs(load_sample_input(selected_key))

    with input_cols[1]:
        model = load_model(selected_key)
        if model:
            input_df, df = get_user_input(selected_key)
            if input_df is not None:
                result, confidence = predict(model, input_df)

    # Prediction output (full width)
    if input_df is not None and result is not None:
        show_prediction_result(selected_key, result, confidence, input_df.iloc[0].to_dict())
        
        # Visualizations
        st.markdown("""
<div style='text-align: center; margin: 20px 0;'>
    <h2 style='color: #4361ee; 
               background: linear-gradient(to right, #f8f9fa, #4361ee20, #f8f9fa);
               padding: 10px;
               border-radius: 8px;
               display: inline-block;
               width: 100%;'>
        📈 Prediction Analysis
    </h2>
</div>
""", unsafe_allow_html=True)
        
        # Confidence gauge centered
        show_confidence_gauge(confidence)
        
        # Pie chart centered (reduced size)
        show_class_distribution(df, df.columns[-1])
        
        # Feature distributions
        show_feature_distribution(df, input_df)
        
        # Correlation heatmap
        show_correlation_heatmap(df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem; margin-top: 2rem;">
        Created with  by Saran Kambala !
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
