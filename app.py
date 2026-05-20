import streamlit as st
import pickle
import pandas as pd
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
df = pd.read_csv("Training.csv")
columns = list(df.drop("prognosis", axis=1).columns)
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction System")
st.write("Select your symptoms:")
selected_symptoms = st.multiselect(
    "Search and select symptoms",
    columns
)
input_data = [1 if col in selected_symptoms else 0 for col in columns]
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:
        input_df = pd.DataFrame([input_data], columns=columns)
        prediction = model.predict(input_df)
        result = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Disease: {result}")