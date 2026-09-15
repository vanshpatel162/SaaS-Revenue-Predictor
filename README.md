# 🚀 SaaS Startup Revenue Predictor

An end-to-end Machine Learning pipeline and interactive web dashboard that predicts the future profitability of SaaS startups based on departmental expenditures and geographical location.

## 🎯 Project Overview
The objective of this project is to assist investors and startup founders in understanding which expenses actually drive profit. Using **Lasso Regression (L1 Regularization)**, the model performs automated Feature Selection—mathematically proving that R&D and Marketing spends are crucial, while shrinking irrelevant features (like Administration costs) to exactly zero.

## 🛠️ Tech Stack Used
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Linear, Ridge, Lasso Regression, GridSearchCV, StandardScaler)
* **Web UI:** Streamlit
* **Deployment:** GitHub

## 📊 Key Highlights & Results
* Built a complete ML pipeline avoiding data leakage (`fit_transform` on train, `transform` on test).
* Achieved **~90% accuracy** with standard Linear Regression.
* Improved accuracy to **91.66%** using Lasso Regression via Hyperparameter Tuning (Alpha = 1000).
* Deployed a responsive, interactive web application using Streamlit.

## 💻 How to Run This Project Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/vanshpate162/SaaS-Revenue-Predictor.git](https://github.com/vanshpatel62/SaaS-Revenue-Predictor.git)

2. Open the folder in your terminal and install Streamlit:
   `pip install streamlit`

3. Run the application:
   `streamlit run app.py`
