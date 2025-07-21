def recommend_medicines(condition):
    recs = {
        "diabetes": ["Metformin", "Insulin"],
        "heart disease": ["Aspirin", "Beta Blockers", "Statins"],
        "blood pressure": ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"],
        "blood sugar": ["Glipizide", "Metformin"],
        "fever": ["Paracetamol", "Ibuprofen", "Hydration"],
        "readmission": ["Scheduled follow-ups", "Medication adherence", "Diet control"]
    }
    return recs.get(condition, ["Consult a physician"])
