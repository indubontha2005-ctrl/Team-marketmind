import pandas as pd
from sklearn.linear_model import LogisticRegression

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

# Test on existing customers
for index, row in data.iterrows():
    input_data = [[
        row["time_on_site"],
        row["pages_viewed"],
        row["cart_abandoned"],
        row["price_viewed"],
        row["sentiment_num"]
    ]]

    prediction = model.predict(input_data)[0]

    print("\nCustomer", index)
    if prediction == 1:
        print("Prediction: WILL BUY ✅")
    else:
        print("Prediction: WILL NOT BUY ❌")

        reasons = []
        if row["price_viewed"] == 1:
            reasons.append("Price concern")
        if row["cart_abandoned"] == 1:
            reasons.append("Checkout issue")
        if row["sentiment"] == "negative":
            reasons.append("Bad experience")

        print("AI Explanation:", ", ".join(reasons))
