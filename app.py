# app.py

from flask import Flask, request, render_template, jsonify
from src.CreditCardDefaultPrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.CreditCardDefaultPrediction.components.model_trainer import ModelTrainer

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    """
    Prediction method
    """
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(            
            LIMIT_BAL = float(request.form.get('LIMIT_BAL')),
            SEX = int(request.form.get('SEX')),
            EDUCATION = int(request.form.get('EDUCATION')),
            MARRIAGE = int(request.form.get('MARRIAGE')),
            AGE = int(request.form.get('AGE')),
            PAY_0 = int(request.form.get('PAY_0')),
            PAY_2 = int(request.form.get('PAY_2')),
            PAY_3 = int(request.form.get('PAY_3')),
            PAY_4 = int(request.form.get('PAY_4')),
            PAY_5 = int(request.form.get('PAY_5')),
            PAY_6 = int(request.form.get('PAY_6')),
            BILL_AMT1 = float(request.form.get('BILL_AMT1')),
            BILL_AMT2 = float(request.form.get('BILL_AMT2')),
            BILL_AMT3 = float(request.form.get('BILL_AMT3')),
            BILL_AMT4 = float(request.form.get('BILL_AMT4')),
            BILL_AMT5 = float(request.form.get('BILL_AMT5')),
            BILL_AMT6 = float(request.form.get('BILL_AMT6')),
            PAY_AMT1 = float(request.form.get('PAY_AMT1')),
            PAY_AMT2 = float(request.form.get('PAY_AMT2')),
            PAY_AMT3 = float(request.form.get('PAY_AMT3')),
            PAY_AMT4 = float(request.form.get('PAY_AMT4')),
            PAY_AMT5 = float(request.form.get('PAY_AMT5')),
            PAY_AMT6 = float(request.form.get('PAY_AMT6'))
        )        
        final_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        pred, prob_not_default, prob_default = predict_pipeline.predict(final_data)
        result = "DEFAULT" if pred[0] == 1 else "NOT DEFAULT"
        probability = prob_default if pred[0] == 1 else prob_not_default

        return render_template("result.html", final_result=result, probability=probability)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)