import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data/customer_data.csv")

# Convert sentiment text to numbers
data["sentiment"] = data["sentiment"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

# Input features (what the AI looks at)
X = data[[
    "time_on_site",
    "pages_viewed",
    "cart_abandoned",
    "price_viewed",
    "sentiment"
]]

# Output (what we want to predict)
y = data["purchased"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

print("AI model trained successfully!")
