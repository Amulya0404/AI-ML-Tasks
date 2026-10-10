"""Streamlit app. Run from the project root: streamlit run app.py"""
import os
import sys
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_passenger  # noqa: E402

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.caption("A machine-learning model trained on historical Titanic passenger records. "
           "Enter a passenger's details to see the predicted chance of survival.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2, help="1 = first class, 3 = third class")
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    age_known = st.checkbox("Age is known", value=True)
    age = st.slider("Age", 0, 80, 30, disabled=not age_known)
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 6, 0)
    fare = st.number_input("Ticket fare (£)", 0.0, 520.0, 15.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                            format_func=lambda p: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[p])

if st.button("Predict survival", type="primary"):
    r = predict_passenger(pclass, sex, age if age_known else np.nan, sibsp, parch, fare, embarked)
    p = r["survival_probability"]
    if r["survived"]:
        st.success(f"Predicted: **SURVIVED**  —  survival probability {p:.1%}")
    else:
        st.error(f"Predicted: **DID NOT SURVIVE**  —  survival probability {p:.1%}")
    st.progress(p)
    st.caption("Educational demo. The model reflects patterns in 1912 data (mainly sex, ticket class and fare) "
               "and its test accuracy is about 80%.")
