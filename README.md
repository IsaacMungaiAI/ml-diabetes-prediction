# 🩺 Diabetes Prediction with Machine Learning

This is a machine learning project that predicts whether a patient is likely to have diabetes based on their medical data. The model is trained on the popular Pima Indians Diabetes dataset and deployed using [Streamlit](https://streamlit.io/) for easy interaction.

---

## 📊 Dataset

The dataset contains medical diagnostic measurements of female patients and includes the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (1 = diabetic, 0 = not diabetic)

Source: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## ⚙️ Model Building

- **Preprocessing**: Handled scaling using `StandardScaler`
- **Class Imbalance**: Addressed using `SMOTE`
- **Model**: `RandomForestClassifier` with hyperparameter tuning using `GridSearchCV`
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Precision, Recall, F1-score

---

## 🚀 Streamlit App

The app allows users to input patient data manually and get a real-time prediction.

### Run the app locally:

```bash

Make sure you have the following files in the project:

app.py (Streamlit UI)

diabetes_model2.pkl (Trained model)

scaler.pkl (Saved scaler)


 Project Structure

diabetes-prediction/
│
├── app.py                # Streamlit app
├── diabetes_model2.pkl   # Trained Random Forest model
├── scaler.pkl            # Saved StandardScaler
├── diabetes.csv          # Dataset
├── requirements.txt      # Project dependencies
└── README.md             # This file


Installation
pip install -r requirements.txt

Dependencies include:
pandas
numpy
scikit-learn
seaborn
matplotlib
streamlit
imblearn
joblib


streamlit run app.py
⭐️ License
This project is open-source and free to use for educational purposes.


---

### ✅ How to Add This in PyCharm or VS Code:

1. In your project folder, **right-click > New File > name it `README.md`**
2. Paste the above content
3. Commit & push to GitHub:
   ```bash
   git add README.md
   git commit -m "Added README file"
   git push


🙋‍♂️ Author
Isaac Mungai
Computer Science AI/ML 
Project powered by Python, Machine Learning and Streamlit.
