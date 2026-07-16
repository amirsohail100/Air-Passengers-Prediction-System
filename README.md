# ✈️ Air Passengers Prediction System

An interactive Machine Learning web application that predicts the estimated count of monthly air passengers based on the target **Year** and **Month**. This project utilizes advanced regression algorithms to achieve high-precision forecasting, featuring an intuitive frontend dashboard built with Streamlit.

---

## 🚀 Live Demo & Deployment

You can interact with the live dashboard application here:
🔗 **[Live Streamlit Application Link](PASTE_YOUR_STREAMLIT_DEPLOYMENT_LINK_HERE)**

---

## 📸 User Interface (UI) Dashboard

Here is a preview of the active web dashboard interface where users can select inputs and get real-time forecasting numbers:

![Dashboard Interface](ui.png)
_(Note: Replace `ui.png` in the repository root folder with your actual dashboard screenshot to render it here)_

---

## 🛠️ Data Preprocessing & Pipeline

To maintain data structure integrity between training and inference phases, the following pipeline operations were carried out:

1. **Numerical Standardization:** The `year` column is preprocessed using a `StandardScaler` to bring variance to scale.
2. **Categorical Encoding:** The `month` feature is transformed using **One-Hot Encoding** to effectively evaluate seasonal trends.
3. **Artifact Serialization:** Preprocessing pipelines and columns layout structure are safely serialized into `scaler.pkl` and `columns.pkl` respectively for smooth runtime execution.

---

## 📊 Model Evaluation & Benchmarking

Multiple regression variants were trained and thoroughly benchmarked based on $R^2$ and Adjusted $R^2$ matrices. The comparative evaluations are summarized below:

| Model Index | Trained Models             | $R^2$ Score | Adjusted $R^2$ Score |           Status           |
| :---------: | :------------------------- | :---------: | :------------------: | :------------------------: |
|      0      | Linear Regression (LR)     |    0.94     |         0.88         |         Evaluated          |
|      1      | K-Nearest Neighbors (KNN)  |    0.92     |         0.86         |         Evaluated          |
|      2      | Decision Tree (DT)         |    0.89     |         0.79         |         Evaluated          |
|      3      | Random Forest (RF)         |    0.92     |         0.85         |         Evaluated          |
|      4      | AdaBoost (ADA)             |    0.86     |         0.74         |         Evaluated          |
|    **5**    | **Gradient Boosting (GD)** |  **0.98**   |       **0.96**       | **🏆 Best Fit (Selected)** |
|      6      | XGBoost (XG)               |    0.97     |         0.95         |         Evaluated          |

_Our production system utilizes the **Gradient Boosting (GD)** architecture (`model.pkl`) which delivered an outstanding benchmark performance of **98% accuracy ($R^2 = 0.98$)**._

---

## 🗂️ Project Repository Architecture

```text
├── app.py                 # Core Streamlit interface script with built-in Exception Handling
├── requirements.txt       # Dependencies (streamlit, scikit-learn, joblib, pandas, numpy)
├── model.pkl              # Pickled Gradient Boosting model layout
├── scaler.pkl             # Serialized StandardScaler instance
├── columns.pkl            # Structural column validation checklist array
├── ui.png                 # Dashboard preview display asset
├── Flights.ipynb   # Jupyter Notebook containing Data Analysis & Model Training pipeline
└── README.md              # Document portfolio profile


An interactive Machine Learning web app built with Streamlit to forecast monthly air passengers. Features a 98% accurate Gradient Boosting Regressor pipeline with robust preprocessing (StandardScaler &amp; One-Hot Encoding) and complete exception handling. Includes the core training Jupyter Notebook
```
