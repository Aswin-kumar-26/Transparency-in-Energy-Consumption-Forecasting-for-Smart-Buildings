# Smart Building Energy Forecaster & XAI Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP-green.svg)

## Introduction & Abstract

Energy prediction in smart buildings can save significant money and energy. While this is highly possible using machine learning models, user trust remains low because these models often operate as a "black box." People are unaware of how the AI reaches its conclusions. 

In this project, we built a transparent, highly accurate model for energy forecasting using Random Forest and XGBoost regressors. 

We compared our project with a recent base paper published in 2025. While their research focused on data privacy, we identified a critical methodology flaw: Data Leakage. They used current appliance data to predict current total energy—essentially giving the model the answers to the test before it took it. 

To solve this, we engineered "Time-Series Lag" features, utilizing past data from exactly one minute ago to legitimately forecast the future. Finally, for complete transparency, we integrated SHAP (SHapley Additive exPlanations) to visually explain exactly why the model generates a certain output.

## Objectives

Here is what we aimed to achieve:
* **Identify & Resolve Flaws:** Prove and fix the target data leakage found in the 2025 baseline literature.
* **Feature Engineering:** Prepare a raw CSV dataset using autoregressive (t-1 lag) features for valid time-series forecasting.
* **Model Training:** Develop and hyper-tune Random Forest and XGBoost models to beat the base paper's accuracy.
* **Deployment:** Package these models into a transparent Flask API & Dashboard, making it straightforward for users to test inputs and see live XAI visualizations.

## Development Workflow

Below is a Mermaid Diagram outlining our project's development workflow from raw data to API deployment:

```mermaid
graph LR
    A[EDA & Data Prep<br>Notebook] --> B[Machine Learning<br>Notebook]
    B --> C[Evaluation & XAI<br>Notebook]
    C --> D[Flask API & Dashboard<br>Development]
    
    D -- Virtual Env --> E[Local Environment]
    E --> F[Test and Use Dashboard]

    style A fill:#2b3e50,stroke:#1a252f,stroke-width:2px,color:#fff
    style B fill:#2b3e50,stroke:#1a252f,stroke-width:2px,color:#fff
    style C fill:#2b3e50,stroke:#1a252f,stroke-width:2px,color:#fff
    style D fill:#34495e,stroke:#1a252f,stroke-width:2px,color:#fff
    style E fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    style F fill:#2980b9,stroke:#1c598a,stroke-width:2px,color:#fff
```

## Project Structure

After prototyping the ML models in a Jupyter Notebook, here is how the final project structure was designed:

```text
Smart-Building-Forecaster/
│
├── data/                                   # Unprocessed initial Kaggle data
│   └── HomeC.csv                 
│
├── notebooks/                              # Jupyter notebooks for faculty review
│   └── Model_Training_EDA.ipynb            # EDA, ML Experimentation, and Graph Generation
│
├── images/                                 # High-resolution (600-1000 DPI) report visuals
│   ├── 1_dataset/
│   ├── 2_random_forest/
│   ├── 3_xgboost/
│   └── 4_comparisons/
│
├── models/                                 # Serialized machine learning assets
│   ├── rf_model.pkl                        
│   ├── xgb_model.pkl                       
│   ├── scaler.pkl                          
│   └── feature_names.pkl                   
│
├── templates/                              # Source code for the Frontend UI
│   └── index.html                          # Dashboard built with Chart.js
│
├── app.py                                  # Main Flask Backend API
├── requirements.txt                        # Project dependencies
└── README.md                               # Comprehensive project guide
```

## Experimental Results

Our Random Forest regressor model performed the best overall. By utilizing the strict lagged features, our models successfully outperformed the 2025 base paper models while remaining mathematically valid.

* **Highest Accuracy:** R² = 0.8085
* **Lowest Error:** RMSE = 0.3231 (Over a 50% error reduction compared to the baseline).

| Dataset Phase | Model Architecture | MAE | RMSE | R-Squared |
|---------------|--------------------|-----|------|-----------|
| **Testing** | **Proposed Lagged Random Forest** | **0.1287** | **0.3231** | **0.8085** |
| **Testing** | **Proposed Lagged XGBoost** | **0.1769** | **0.3518** | **0.7729** |
| *Testing* | *Base Paper Random Forest* | *0.6396* | *0.8158* | *0.8036* |
| *Testing* | *Base Paper XGBoost* | *0.6149* | *0.8781* | *0.7725* |

## Run the Code

You are a few steps away from forecasting energy usage with Explainable AI.

**1. Environment Setup**
Confirm Python is installed and ready. Open your terminal in the project's main folder.

**2. Dependency Installation**
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

**3. Running the Application**
Your environment is now primed for action. Start the Flask backend server:
```bash
python app.py
```

**4. Access the Dashboard**
Open your preferred web browser and navigate to:
```text
http://127.0.0.1:5000
```
You can now input environmental data and lagged historical usage to watch the models predict live, complete with SHAP waterfall explanations.