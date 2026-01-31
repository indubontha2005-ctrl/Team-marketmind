import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🧠 Why Customers Don’t Buy – AI System")

# Load data
data = pd.read_csv("data/customer_data.csv")

# Convert sentiment to numbers
data["sentiment_num"] = data["sentiment"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

X = data[[
    "time_on_site",
    "pages_viewed",
    "cart_abandoned",
    "price_viewed",
    "sentiment_num"
]]

y = data["purchased"]

# Train model
model = LogisticRegression()
model.fit(X, y)

st.subheader("Customer Data")
st.dataframe(data)

# Select customer
customer_id = st.selectbox("Select Customer Index", data.index)

row = data.loc[customer_id]

input_data = [[
    row["time_on_site"],
    row["pages_viewed"],
    row["cart_abandoned"],
    row["price_viewed"],
    row["sentiment_num"]
]]

prediction = model.predict(input_data)[0]

st.subheader("AI Prediction")

if prediction == 1:
    st.success("✅ Customer WILL BUY")
else:
    st.error("❌ Customer WILL NOT BUY")

    reasons = []
    if row["price_viewed"] == 1:
        reasons.append("Price is too high")
    if row["cart_abandoned"] == 1:
        reasons.append("Checkout problem")
    if row["sentiment"] == "negative":
        reasons.append("Negative experience")

    st.write("### AI Explanation")
    st.info(", ".join(reasons))
