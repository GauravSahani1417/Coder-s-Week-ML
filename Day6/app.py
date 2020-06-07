# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:05:40 2020

@author: gaurav sahani
"""


from flask import Flask, render_template, request
import joblib
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            Gender = float(request.form['Gender'])
            Age = float(request.form['Age'])
            EstimatedSalary = float(request.form['EstimatedSalary'])
            pred_args = [Gender, Age, EstimatedSalary]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            IEEE_Task = open("IEEE.pkl", "rb")
            ml_model = joblib.load(IEEE_Task)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        
        except ValueError:
            return "Please Check if values are written correctly"
    return render_template('predict.html', prediction=model_prediction)

if __name__ == "__main__":
    app.run()