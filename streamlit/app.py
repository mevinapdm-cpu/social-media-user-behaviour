import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Social Media Usage Prediction", layout="wide")

st.title("📱 Social Media Daily Time Spent Prediction")

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Social Media Users (1).csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Data preprocessing
# -------------------------
data = df.copy()

label_encoders = {}
categorical_cols = ['Platform', 'Owner', 'Primary Usage', 'Country', 'Verified Account']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Drop Date Joined (not useful for prediction)
data.drop('Date Joined', axis=1, inplace=True)

# -------------------------
# Features & Target
# -------------------------
X = data.drop('Daily Time Spent (min)', axis=1)
y = data['Daily Time Spent (min)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model training
# -------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ Model trained successfully | R² Score: {r2:.2f}")

# -------------------------
# User Input Section
# -------------------------
st.subheader("🧑‍💻 Enter User Details")

col1, col2 = st.columns(2)

with col1:
    platform = st.selectbox("Platform", df['Platform'].unique())
    owner = st.selectbox("Owner", df['Owner'].unique())
    primary_usage = st.selectbox("Primary Usage", df['Primary Usage'].unique())

with col2:
    country = st.selectbox("Country", df['Country'].unique())
    verified = st.selectbox("Verified Account", df['Verified Account'].unique())

# Encode user input
input_data = pd.DataFrame({
    'Platform': [label_encoders['Platform'].transform([platform])[0]],
    'Owner': [label_encoders['Owner'].transform([owner])[0]],
    'Primary Usage': [label_encoders['Primary Usage'].transform([primary_usage])[0]],
    'Country': [label_encoders['Country'].transform([country])[0]],
    'Verified Account': [label_encoders['Verified Account'].transform([verified])[0]],
})

# -------------------------
# Prediction
# -------------------------
if st.button("🔮 Predict Daily Time Spent"):
    prediction = model.predict(input_data)[0]
    st.metric(
        label="⏱ Predicted Daily Time Spent (minutes)",
        value=f"{prediction:.2f} min"
    )
