
# Smart Building Energy Forecaster & XAI Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP-green.svg)

## Introduction & Abstract

Energy prediction in smart buildings can save significant money and energy. While this is highly possible using machine learning models, user trust remains low because these models often operate as a "black box." People are unaware of how the AI reaches its conclusions. 

In this project, we built a transparent, highly accurate model for energy forecasting using Random Forest and XGBoost regressors. 

A critical challenge in energy forecasting is avoiding "Data Leakage"—using current appliance data to predict current total energy, which artificially inflates model accuracy by giving the AI the answers before the test. To solve this and build a mathematically robust forecasting engine, we engineered "Time-Series Lag" features. By utilizing past data from exactly one minute ago, the model legitimately forecasts future energy states. Finally, for complete transparency, we integrated SHAP (SHapley Additive exPlanations) to visually explain exactly why the model generates a certain output.

## Objectives

Here is what we aimed to achieve:
* **Feature Engineering:** Prepare a raw CSV dataset using autoregressive (t-1 lag) features to ensure valid, leakage-free time-series forecasting.
* **Model Training:** Develop and hyper-tune Random Forest and XGBoost models to achieve high predictive accuracy.
* **Explainability:** Integrate Explainable AI (XAI) techniques to decode the model's decision-making process.
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
│   └── [https://www.kaggle.com/code/offmann/smart-home-dataset/input]                 
│
├── notebooks/                              # Jupyter notebooks for faculty review
│   └── Model_Training_EDA.ipynb            # EDA, ML Experimentation, and Graph Generation
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
*(Note: To run this project, please download the HomeC dataset from Kaggle and place it in the `data/` directory).*

## Experimental Results

Our Random Forest regressor model performed the best overall. By utilizing strict features, our ensemble models successfully achieved high predictive performance while remaining mathematically valid and robust for real-world deployment. The tight correlation between training and testing scores proves the models did not over-fit the data.

* **Highest Accuracy:** R² = 0.8085
* **Lowest Error:** RMSE = 0.3231 

| Dataset Phase | Models | MAE | RMSE | R-Squared |
|---------------|--------------------|-----|------|-----------|
| **Training** | Random Forest Regressor | 0.1012 | 0.1952 | 0.9782 |
| **Testing** | **Random Forest Regressor** | **0.1287** | **0.3231** | **0.8085** |
| **Training** | XGBoost Regressor | 0.1504 | 0.2610 | 0.9511 |
| **Testing** | **XGBoost Regressor** | **0.1769** | **0.3518** | **0.7729** |

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


## Contributors


| Name             | GitHub Profile                                        |
|------------------|--------------------------------------------------------|
| Aswin Kumar      | [@Aswin-kumar-26](https://github.com/Aswin-kumar-26)  |
| Sharun Kumar    | [@SharunKumarD](https://github.com/SharunKumarD) |

---
