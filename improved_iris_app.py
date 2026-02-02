import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("iris-species.csv")

iris_df = load_data()

iris_df["Label"] = iris_df["Species"].map({
    "Iris-setosa": 0,
    "Iris-virginica": 1,
    "Iris-versicolor": 2
})

X = iris_df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]
y = iris_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# ---------- Train Models ----------
@st.cache_resource
def train_models():
    svc = SVC(kernel="linear")
    svc.fit(X_train, y_train)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train, y_train)

    return svc, log_reg, rf

svc_model, log_reg_model, rf_model = train_models()

# ---------- Prediction ----------
def predict_species(model, sl, sw, pl, pw):
    label = model.predict([[sl, sw, pl, pw]])[0]
    return {
        0: "Iris-setosa",
        1: "Iris-virginica",
        2: "Iris-versicolor"
    }[label]

# ---------- UI ----------
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter flower measurements and choose a classifier.")

col1, col2 = st.columns(2)

with col1:
    s_len = st.slider("Sepal Length", float(X["SepalLengthCm"].min()), float(X["SepalLengthCm"].max()))
    p_len = st.slider("Petal Length", float(X["PetalLengthCm"].min()), float(X["PetalLengthCm"].max()))

with col2:
    s_wid = st.slider("Sepal Width", float(X["SepalWidthCm"].min()), float(X["SepalWidthCm"].max()))
    p_wid = st.slider("Petal Width", float(X["PetalWidthCm"].min()), float(X["PetalWidthCm"].max()))

classifier = st.selectbox(
    "Choose Classifier",
    ("Support Vector Machine", "Logistic Regression", "Random Forest")
)

if st.button("🔮 Predict"):
    if classifier == "Support Vector Machine":
        model = svc_model
        accuracy = model.score(X_test, y_test)
    elif classifier == "Logistic Regression":
        model = log_reg_model
        accuracy = model.score(X_test, y_test)
    else:
        model = rf_model
        accuracy = model.score(X_test, y_test)

    species = predict_species(model, s_len, s_wid, p_len, p_wid)

    st.success(f"🌼 Predicted Species: **{species}**")
    st.info(f"📊 Model Accuracy: **{accuracy:.2f}**")
