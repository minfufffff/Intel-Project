import streamlit as st
import pickle
import numpy as np

# load models
wine_model = pickle.load(open('models/wine_model.pkl','rb'))
spam_model = pickle.load(open('models/spam_model.pkl','rb'))
tfidf = pickle.load(open('models/tfidf.pkl','rb'))

st.title("AI Prediction System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Wine Model Info","Spam Model Info","Test Wine Model","Test Spam Model"]
)

# -----------------------
if menu == "Wine Model Info":
    st.header("Wine Quality Prediction")
    st.write("""
    โมเดลนี้ใช้ Ensemble Learning (Logistic Regression + Random Forest)
    เพื่อจำแนกคุณภาพไวน์ (Good / Bad)
    """)
    st.info("Accuracy ≈ ดูใน console ตอน train")

# -----------------------
elif menu == "Spam Model Info":
    st.header("Spam Detection")
    st.write("""
    โมเดลนี้ใช้ TF-IDF + Logistic Regression
    เพื่อจำแนกข้อความ Spam
    """)
    st.info("Accuracy ≈ ดูใน console ตอน train")

# -----------------------
elif menu == "Test Wine Model":
    st.header("Predict Wine Quality")

    inputs = []
    features = [
        "fixed acidity","volatile acidity","citric acid","residual sugar",
        "chlorides","free sulfur dioxide","total sulfur dioxide",
        "density","pH","sulphates","alcohol"
    ]

    for f in features:
        inputs.append(st.number_input(f))

    if st.button("Predict"):
        data = np.array([inputs])
        result = wine_model.predict(data)

        if result[0] == 1:
            st.success("Good Quality 🍷")
        else:
            st.error("Bad Quality ❌")

# -----------------------
elif menu == "Test Spam Model":
    st.header("Spam Detection")

    text = st.text_area("Enter message")

    if st.button("Check"):
        vec = tfidf.transform([text]).toarray()
        result = spam_model.predict(vec)

        if result[0] == 1:
            st.error("Spam ❌")
        else:
            st.success("Not Spam ✅")