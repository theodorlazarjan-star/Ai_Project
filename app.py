from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load trained model and threshold

#model = joblib.load("churn_model.pkl")
#threshold = joblib.load("churn_threshold.pkl")

MODEL_FOLDER = "models"

#print("Model loaded successfully!")
#print("Threshold:", threshold)

DATASET_PATH = "Telco-Customer-Churn.csv"

df = pd.read_csv(DATASET_PATH)

os.makedirs("static/plots", exist_ok=True)

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

    print("PREDICT ROUTE STARTED")

    # Get selected model
    model_name = request.form["model"]

    print("Selected model:", model_name)

    # Load selected model
    model = joblib.load(
        os.path.join(MODEL_FOLDER, f"{model_name}_model.pkl")
    )

    print("MODEL LOADED")

    # Load corresponding threshold
    threshold = joblib.load(
        os.path.join(MODEL_FOLDER, f"{model_name}_threshold.pkl")
    )

    print("THRESHOLD LOADED:", threshold)

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

    print("FORM DATA LOADED")


    # Create customer plot

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="tenure",
        y="MonthlyCharges",
        hue="Churn"
    )

    plt.scatter(
        tenure,
        monthly,
        color="red",
        s=180,
        marker="X",
        label="Current Customer"
    )

    plt.title("Tenure vs Monthly Charges")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Monthly Charges")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "static/plots/customer_tenure_monthly.png"
    )

    plt.close()


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

    print("DATAFRAME CREATED")

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

    print("BOOLEAN FEATURES CONVERTED")

    # One-hot encoding

    data = pd.get_dummies(
        data,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod"
        ]
    )

    print("ONE-HOT ENCODING DONE")

    # Make sure all expected features exist

    expected_features = model.feature_names_in_

    for feature in expected_features:

        if feature not in data.columns:
            data[feature] = False

    print("EXPECTED FEATURES ADDED")

    # Keep exact model features and order

    data = data[expected_features]

    print("FEATURE ORDER SET")

    # Convert boolean values to integers

    boolean_columns = data.select_dtypes(
        include=["bool"]
    ).columns

    data[boolean_columns] = data[boolean_columns].astype(int)

    print("BOOLEAN VALUES CONVERTED")

    # Prediction probability

    probability = model.predict_proba(data)[0][1]

    print("PROBABILITY:", probability)

    # Apply threshold

    prediction = probability >= threshold

    if prediction:
        result = "Churn"
    else:
        result = "No Churn"

    print("RESULT:", result)

    # Return to SAME model page

    return render_template(
        "Model.html",
        prediction=result,
        probability=probability,
        threshold=threshold
    )

# Start Flask

if __name__ == "__main__":
    app.run(debug=False)