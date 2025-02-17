{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfdf727f-5d13-4e79-b54d-d5a46d9837b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load the trained model and scaler\n",
    "model = joblib.load(\"C:/Users/Owner/OneDrive/Documents/diabetes_model.pkl\")\n",
    "scaler = joblib.load(\"C:/Users/Owner/OneDrive/Documents/scaler.pkl\")\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Diabetes Prediction App\")\n",
    "st.write(\"Enter the patient details below to predict diabetes.\")\n",
    "\n",
    "# Input fields\n",
    "pregnancies = st.number_input(\"Number of Pregnancies\", 0, 20, 1)\n",
    "glucose = st.number_input(\"Glucose Level\", 0, 300, 100)\n",
    "blood_pressure = st.number_input(\"Blood Pressure\", 0, 200, 70)\n",
    "skin_thickness = st.number_input(\"Skin Thickness\", 0, 100, 20)\n",
    "insulin = st.number_input(\"Insulin Level\", 0, 900, 79)\n",
    "bmi = st.number_input(\"BMI\", 0.0, 100.0, 25.0)\n",
    "dpf = st.number_input(\"Diabetes Pedigree Function\", 0.0, 2.5, 0.5)\n",
    "age = st.number_input(\"Age\", 1, 120, 25)\n",
    "\n",
    "# Predict function\n",
    "if st.button(\"Predict\"):\n",
    "    # Create a DataFrame for model input\n",
    "    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,\n",
    "                           insulin, bmi, dpf, age]])\n",
    "    \n",
    "    # Scale the input data\n",
    "    user_data_scaled = scaler.transform(user_data)\n",
    "\n",
    "    # Predict\n",
    "    prediction = model.predict(user_data_scaled)\n",
    "    result = \"Diabetic\" if prediction[0] == 1 else \"Not Diabetic\"\n",
    "\n",
    "    # Display result\n",
    "    st.success(f\"Prediction: {result}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
