import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# -----------------------
# App Title
# -----------------------
st.set_page_config(page_title="Customer Intelligence Website", layout="wide")
st.title("🧠 Customer Purchase Intelligence Website")

# -----------------------
# Simple Login
# -----------------------
def login():
    st.sidebar.subheader("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "admin":
            st.sidebar.success("Login Successful!")
            return True
        else:
            st.sidebar.error("Invalid Credentials")
            return False
    return False

logged_in = login()

if not logged_in:
    st.stop()

# -----------------------
# Load Data
# -----------------------
data = pd.read_csv("data/customer_data.csv")
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

# -----------------------
# Sidebar Menu
# -----------------------
menu = st.sidebar.selectbox(
    "Select Option",
    ["Dashboard", "Predict Customer", "Why Not Buy", "Insights", "Export"]
)

# -----------------------
# DASHBOARD
# -----------------------
if menu == "Dashboard":
    st.subheader("📊 Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Total Customers")
        st.write(len(data))

        st.write("### Customers who bought")
        st.write(data["purchased"].sum())

        st.write("### Customers who did NOT buy")
        st.write(len(data) - data["purchased"].sum())

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="purchased", data=data, ax=ax)
        ax.set_title("Purchase Distribution")
        ax.set_xlabel("Purchased (0=No, 1=Yes)")
        st.pyplot(fig)

    st.write("### Data Preview")
    st.dataframe(data)

# -----------------------
# PREDICT CUSTOMER
# -----------------------
elif menu == "Predict Customer":
    st.subheader("🔮 Predict Customer Purchase")

    customer_id = st.selectbox("Select Customer", data.index)
    row = data.loc[customer_id]

    input_data = pd.DataFrame([{
        "time_on_site": row["time_on_site"],
        "pages_viewed": row["pages_viewed"],
        "cart_abandoned": row["cart_abandoned"],
        "price_viewed": row["price_viewed"],
        "sentiment_num": row["sentiment_num"]
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediction: WILL BUY")
    else:
        st.error("❌ Prediction: WILL NOT BUY")

# -----------------------
# WHY NOT BUY
# -----------------------
elif menu == "Why Not Buy":
    st.subheader("🧾 Explanation for Non-Purchase")

    customer_id = st.selectbox("Select Customer", data.index)
    row = data.loc[customer_id]

    if row["purchased"] == 1:
        st.success("Customer bought the product.")
    else:
        reasons = []
        if row["price_viewed"] == 1:
            reasons.append("Price sensitivity")
        if row["cart_abandoned"] == 1:
            reasons.append("Checkout friction")
        if row["sentiment"] == "negative":
            reasons.append("Negative experience")

        st.error("Customer did NOT buy")
        st.write("### Reasons Identified:")
        for r in reasons:
            st.write("•", r)

# -----------------------
# INSIGHTS
# -----------------------
elif menu == "Insights":
    st.subheader("💡 Business Insights")

    st.write("### Most common reasons for not buying")
    st.write("""
    - Price sensitivity
    - Checkout friction
    - Negative experience
    """)

    st.write("### Recommendation")
    st.write("""
    - Offer discount coupons for price-sensitive customers
    - Improve checkout speed
    - Add customer support for trust issues
    """)

# -----------------------
# EXPORT
# -----------------------
elif menu == "Export":
    st.subheader("📤 Export Data")

    st.write("You can export the dataset as CSV.")
    csv = data.to_csv(index=False)
    st.download_button("Download CSV", csv, "customer_data.csv", "text/csv")
