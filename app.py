import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Model
model = pickle.load(open("student_model.pkl", "rb"))

# Title
st.title("🎓 Student Performance Predictor")
st.markdown("### Predict Student Math Score using Machine Learning")

st.markdown("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    race = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"]
    )

    parental_education = st.selectbox(
        "Parental Level of Education",
        [
            "high school",
            "some high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]
    )

with col2:
    lunch = st.selectbox(
        "Lunch Type",
        ["standard", "free/reduced"]
    )

    test_prep = st.selectbox(
        "Test Preparation Course",
        ["none", "completed"]
    )

    reading = st.slider(
        "Reading Score",
        0,
        100,
        50
    )

    writing = st.slider(
        "Writing Score",
        0,
        100,
        50
    )

# Encoding
gender_val = 1 if gender == "Male" else 0

race_map = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

parent_map = {
    "high school": 0,
    "some high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

lunch_val = 1 if lunch == "standard" else 0
test_val = 1 if test_prep == "completed" else 0

race_val = race_map[race]
parent_val = parent_map[parental_education]

if st.button("🔍 Predict Score"):

    input_data = pd.DataFrame(
        [[
            gender_val,
            race_val,
            parent_val,
            lunch_val,
            test_val,
            reading,
            writing
        ]],
        columns=[
            'gender',
            'race/ethnicity',
            'parental level of education',
            'lunch',
            'test preparation course',
            'reading score',
            'writing score'
        ]
    )

    prediction = model.predict(input_data)

    score = float(prediction[0])

    st.markdown("---")

    st.metric(
        "Predicted Math Score",
        f"{score:.2f}"
    )

    st.progress(min(int(score), 100))

    if score >= 80:
        st.success("🌟 Excellent Student")
    elif score >= 60:
        st.warning("📚 Average Student")
    else:
        st.error("📈 Needs Improvement")

    st.subheader("📊 Performance Chart")

    chart_data = pd.DataFrame(
        {
            "Score": [
                reading,
                writing,
                score
            ]
        },
        index=[
            "Reading",
            "Writing",
            "Predicted Math"
        ]
    )

    st.bar_chart(chart_data)

    st.subheader("📋 Analysis")

    st.write(f"Reading Score: {reading}")
    st.write(f"Writing Score: {writing}")
    st.write(f"Predicted Math Score: {score:.2f}")

st.markdown("---")
st.caption("Built with Streamlit and Scikit-Learn")