import streamlit as st
import pickle
import re
import numpy as np

wine_model = pickle.load(open('models/wine_model.pkl','rb'))
spam_model = pickle.load(open('models/spam_model.pkl','rb'))
tfidf = pickle.load(open('models/tfidf.pkl','rb'))

st.title("AI Prediction System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Wine Model Info","Spam Model Info","Test Wine Model","Test Spam Model"]
)

if menu == "Wine Model Info":
    st.header("Wine Quality Prediction 🍷")

    st.write("""
    โมเดลนี้ใช้ Machine Learning เพื่อจำแนกคุณภาพของไวน์ (Good / Bad) โดยใช้ข้อมูลทางเคมีของไวน์ เช่น Acidity, Alcohol, Sulphates
    """)

    st.subheader("Model Details")
    st.write("""
    - ใช้ Logistic Regression + Random Forest
    - ใช้ StandardScaler กับ Logistic Regression
    - เลือกโมเดลที่มี Accuracy สูงสุด
    """)

    st.subheader("Features")
    st.write("""
    - Fixed Acidity
    - Volatile Acidity
    - Citric Acid
    - Residual Sugar
    - Chlorides
    - Free Sulfur Dioxide
    - Total Sulfur Dioxide
    - Density
    - pH
    - Sulphates
    - Alcohol
    """)

    st.subheader("Model Performance")
    st.success("Random Forest Accuracy ≈ 80%")
    st.info("Logistic Regression Accuracy ≈ 74%")

    st.subheader("Dateset")
    st.write("""
    ใช้ชุดข้อมูล Wine Quality Dataset จาก UCI Machine Learning Repository
    """)

elif menu == "Spam Model Info":
    st.header("Spam Detection 📩")

    st.write("""
    โมเดลนี้ใช้ Natural Language Processing (NLP)
    เพื่อจำแนกข้อความว่าเป็น Spam หรือไม่
    """)

    st.subheader("Model Details")
    st.write("""
    - ใช้ TF-IDF Vectorization (รองรับ unigram + bigram)
    - ใช้ Logistic Regression (class_weight balanced)
    - เพิ่ม feature พิเศษ เช่น:
        • มีลิงก์ (URL)
        • มีคำเกี่ยวกับเงิน/รางวัล
    """)

    st.subheader("Text Processing")
    st.write("""
    - แปลงข้อความเป็น lowercase
    - แทน URL → 'URL'
    - แทนตัวเลข → 'NUM'
    - ลบสัญลักษณ์พิเศษ
    """)

    st.subheader("Model Performance")
    st.success("Accuracy ≈ 0.90")

    st.subheader("Dateset")
    st.write("""
    ใช้ชุดข้อมูล SMS Spam Collection Dataset จาก UCI Machine Learning Repository
    """)

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
            st.success("Good Quality 👍")
        else:
            st.error("Bad Quality 👎")

elif menu == "Test Spam Model":
    st.header("Spam Detection")

    text = st.text_area("Enter message")

    if st.button("Check"):

        def clean(text):
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', ' URL ', text)
            text = re.sub(r'\d+', ' NUM ', text)
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            return text

        clean_text = clean(text)

        def has_link(text):
            return int('url' in text)

        def has_money(text):
            keywords = ['cash','prize','win','free','reward']
            return int(any(k in text for k in keywords))

        link_feat = has_link(clean_text)
        money_feat = has_money(clean_text)

        vec = tfidf.transform([clean_text]).toarray()

        final_input = np.hstack((vec, [[link_feat, money_feat]]))

        result = spam_model.predict(final_input)
        prob = spam_model.predict_proba(final_input)[0][1]

        # output
        if result[0] == 1:
            st.error(f"Spam ❌")
        else:
            st.success(f"Not Spam ✅")