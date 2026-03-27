from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load Assets
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Route 1: Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route 2: Handle the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        for key, value in data.items():
            if key in feature_names:
                input_df[key] = float(value)
                
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
        
        # Predictions
        rf_pred = float(rf_model.predict(input_scaled)[0])
        xgb_pred = float(xgb_model.predict(input_scaled)[0])
        
        # Extract Global Feature Importances ("Normal" Global Impact)
        importances = xgb_model.feature_importances_.tolist()
        
        # Local SHAP Waterfall Plot
        plt.style.use('dark_background')
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_scaled)
        
        fig, ax = plt.subplots(figsize=(7, 4)) # Resized to fit the new dashboard
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=input_df.iloc[0], 
            feature_names=feature_names
        ), show=False)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'rf_prediction': round(rf_pred, 4),
            'xgb_prediction': round(xgb_pred, 4),
            'shap_image': f"data:image/png;base64,{plot_base64}",
            'feature_names': feature_names,
            'feature_importances': importances
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
if __name__ == '__main__':
    print("Starting unified Flask server...")
    app.run(port=5000, debug=True)