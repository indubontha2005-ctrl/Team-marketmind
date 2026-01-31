import pandas as pd

data = pd.read_csv("data/customer_data.csv")

for index, row in data.iterrows():
    if row["purchased"] == 0:
        print("\nCustomer did NOT buy.")
        reasons = []

        if row["price_viewed"] == 1:
            reasons.append("Price is too high")

        if row["cart_abandoned"] == 1:
            reasons.append("Customer left during checkout")

        if row["sentiment"] == "negative":
            reasons.append("Negative experience")

        print("Reasons:", ", ".join(reasons))
