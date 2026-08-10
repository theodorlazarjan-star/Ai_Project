from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and threshold

model = joblib.load("churn_model.pkl")
threshold = joblib.load("churn_threshold.pkl")

print("Model loaded successfully!")
print("Threshold:", threshold)

# Home

@app.route("/")
def home():
    return render_template("Home.html")

# Model page

@app.route("/model")
def model_page():
    return render_template("Model.html")

# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form

    tenure = float(request.form["tenure"])
    multiplelines = request.form["multiplelines"]
    internet = request.form["internet"]
    security = request.form["security"]
    backup = request.form["backup"]
    tech = request.form["tech"]
    paperless = request.form["paperless"]
    contract = request.form["contract"]
    payment = request.form["payment"]
    monthly = float(request.form["monthly"])

    # Create DataFrame

    data = pd.DataFrame([{
        "tenure": tenure,
        "MultipleLines": multiplelines,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "TechSupport": tech,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment
    }])

    # Convert Yes / No features

    yes_no_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "TechSupport",
        "PaperlessBilling"
    ]

    for column in yes_no_columns:
        data[column] = data[column].map({
            "Yes": True,
            "No": False,
            "No internet service": False
        })

    # Multiple Lines

    data["MultipleLines"] = data["MultipleLines"].map({
        "Yes": True,
        "No": False,
        "No phone service": False
    })

    # One-hot encoding

    data = pd.get_dummies(
        data,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod"
        ]
    )

    # Make sure all expected features exist

    expected_features = model.feature_names_in_

    for feature in expected_features:

        if feature not in data.columns:
            data[feature] = False

    # Keep exact model features and order

    data = data[expected_features]

    # Convert boolean values to integers

    boolean_columns = data.select_dtypes(
        include=["bool"]
    ).columns

    data[boolean_columns] = data[boolean_columns].astype(int)

    # Prediction probability

    probability = model.predict_proba(data)[0][1]

    # Apply threshold

    prediction = probability >= threshold

    if prediction:
        result = "Churn"
    else:
        result = "No Churn"

    # Return to SAME model page

    return render_template(
        "Model.html",
        prediction=result,
        probability=probability,
        threshold=threshold
    )

# Start Flask

if __name__ == "__main__":
    app.run(debug=True)