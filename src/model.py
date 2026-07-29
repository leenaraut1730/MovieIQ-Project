# ==========================================
# MovieIQ Project - Machine Learning Model
# ==========================================

# Import Libraries

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ------------------------------------------
# Create Models Folder
# ------------------------------------------

os.makedirs("models", exist_ok=True)


# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("data/cleaned_movies.csv")

print("\nDataset Loaded Successfully")
print("-" * 50)

print("Dataset Shape :", df.shape)


# ------------------------------------------
# Select Features and Target
# ------------------------------------------

X = df[[
        "budget",
        "popularity",
        "runtime",
        "vote_average"
      ]]


y = df["success"]


# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)


# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ------------------------------------------
# Random Forest Model
# ------------------------------------------

model = RandomForestClassifier(

    n_estimators=100,
    random_state=42

)


# ------------------------------------------
# Train Model
# ------------------------------------------

model.fit(

    X_train,
    y_train

)


print("\nModel Training Completed Successfully")


# ------------------------------------------
# Prediction
# ------------------------------------------

prediction = model.predict(X_test)


# ------------------------------------------
# Accuracy Score
# ------------------------------------------

accuracy = accuracy_score(

    y_test,
    prediction

)


print("\nAccuracy Score : ", accuracy)


# ------------------------------------------
# Classification Report
# ------------------------------------------

print("\nClassification Report\n")

print(

    classification_report(

        y_test,
        prediction

    )

)


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y_test,
        prediction

    )

)


# ------------------------------------------
# Save Model
# ------------------------------------------

joblib.dump(

    model,

    "models/random_forest.pkl"

)


# ------------------------------------------
# Save Scaler
# ------------------------------------------

joblib.dump(

    scaler,

    "models/scaler.pkl"

)


# ------------------------------------------
# Final Output
# ------------------------------------------

print("\nModel Saved Successfully")

print("Scaler Saved Successfully")

print("-" * 50)

print("\nMovieIQ Model Training Completed.")