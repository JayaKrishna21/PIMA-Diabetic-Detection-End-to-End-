from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application = Flask(__name__)

app = application

# Route to Home page

@app.route('/')

def index():
    return render_template('abc.html')

@app.route('/predict data',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('abc.html')
    else:
        data = CustomData(
            Pregnancies=request.form.get('Pregnancies'),
            Glucose=request.form.get('Glucose'),
            BloodPressure=request.form.get('BloodPressure'),
            SkinThickness=request.form.get('SkinThicknesss'),
            Insulin=request.form.get('Insulin'),
            BMI=request.form.get('BMI'),
            DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction'),
            Age=request.form.get('Age'),
        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('abc.html',results = results[0])
    

if __name__ == "__main__":
    app.run(host = "0.0.0.0",debug = True)


